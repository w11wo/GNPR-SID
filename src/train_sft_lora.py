import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--codebook_path", type=Path, required=True)
    parser.add_argument("--max_codebook_length", type=int, default=4)
    parser.add_argument("--codebook_embedding_dim", type=int, default=64)
    parser.add_argument("--pid_mapping_path", type=Path, required=True)
    return parser.parse_args()


def encode_samples(datum, venue2codebook):
    inputs, targets = datum["inputs"], datum["targets"]
    target_prompt, label = targets.split("POI id ")
    prompt = inputs + " " + target_prompt + "POI id "
    completion = label.replace(".", "").strip()

    def replace_id(match):
        vid_str = match.group(1)
        vid_int = int(vid_str)
        if vid_int in venue2codebook:
            return f"POI id {''.join(venue2codebook[vid_int])}"
        else:
            return f"POI id {vid_str}"

    prompt = re.sub(r"POI id (\d+)\b", replace_id, prompt)
    completion_semantic_ids = venue2codebook[int(completion)]
    completion_semantic_ids = "".join(completion_semantic_ids)

    return {"prompt": prompt, "completion": completion_semantic_ids}


def main():
    args = parse_args()

    codebook_df = pd.read_csv(args.codebook_path)
    codebook_df["Codebook"] = codebook_df["Codebook"].apply(eval)
    codebook_df["Codebook"] = codebook_df["Codebook"].apply(lambda x: x + [0] * (args.max_codebook_length - len(x)))
    codebook_df = codebook_df[["Pid", "Codebook"]]

    max_collision_token = max(code for codebook in codebook_df["Codebook"] for code in codebook)
    codebook_indices = [
        range(args.codebook_embedding_dim),
        range(args.codebook_embedding_dim),
        range(args.codebook_embedding_dim),
        range(max_collision_token + 1),
    ]
    vocab = [f"<{chr(97 + idx)}_{code}>" for idx, indices in enumerate(codebook_indices) for code in indices]

    poi_map_df = pd.read_csv(args.pid_mapping_path)
    poi_map_df = poi_map_df.rename(columns={"Mapped_Pid": "Pid", "Original_Pid": "venue_id"})

    venue_codebook_df = poi_map_df.merge(codebook_df, on="Pid", how="left")
    venue2codebook = {
        venue_id: [f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(codebook)]
        for venue_id, codebook in zip(venue_codebook_df["venue_id"], venue_codebook_df["Codebook"])
    }

    model_id = Path("models") / f"{args.model_checkpoint.split('/')[-1]}-{args.dataset_id.split('/')[-1]}"

    dataset = load_dataset(args.dataset_id)
    dataset = dataset.map(
        encode_samples,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"venue2codebook": venue2codebook},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_new_tokens = tokenizer.add_tokens(vocab, special_tokens=True)
    print(f"Added {num_new_tokens} new tokens to the tokenizer.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        use_cache=True if args.gradient_checkpointing else False,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )

    args = SFTConfig(
        output_dir=model_id,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        dataloader_num_workers=16,
        num_train_epochs=args.num_epochs,
        optim="adamw_torch",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    trainer.train()

    trainer.save_model(model_id)
    trainer.create_model_card()
    tokenizer.save_pretrained(model_id)


if __name__ == "__main__":
    main()
