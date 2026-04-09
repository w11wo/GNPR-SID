import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from train_sft_lora import encode_samples


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--codebook_path", type=Path, required=True)
    parser.add_argument("--max_codebook_length", type=int, default=4)
    parser.add_argument("--codebook_embedding_dim", type=int, default=64)
    parser.add_argument("--pid_mapping_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_allowed_transitions(tokenizer, vocab_list):
    allowed_transitions = {"start": set(), "a_to_b": {}, "ab_to_c": {}, "abc_to_d": {}}
    for segment in vocab_list:
        t_ids = tokenizer.convert_tokens_to_ids(segment)
        if any(tid == tokenizer.unk_token_id for tid in t_ids):
            continue
        a, b, c, d = t_ids[0], t_ids[1], t_ids[2], t_ids[3]
        allowed_transitions["start"].add(a)
        allowed_transitions["a_to_b"].setdefault(a, set()).add(b)
        allowed_transitions["ab_to_c"].setdefault((a, b), set()).add(c)
        allowed_transitions["abc_to_d"].setdefault((a, b, c), set()).add(d)

    return {k: (list(v) if isinstance(v, set) else v) for k, v in allowed_transitions.items()}


def main():
    args = parse_args()
    set_seed(args.seed)

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
    indices = list(venue2codebook.values())

    dataset = load_dataset(args.dataset_id)
    dataset = dataset.map(
        encode_samples,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"venue2codebook": venue2codebook},
    )

    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        completions = [item["completion"] for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        lengths = inputs.input_ids.shape[1]
        return inputs, completions, lengths

    dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=collate_fn)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_new_tokens = tokenizer.add_tokens(vocab, special_tokens=True)
    assert num_new_tokens == 0

    peft_config = PeftConfig.from_pretrained(args.model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.model_checkpoint).eval()
    model = torch.compile(model, mode="reduce-overhead")

    generation_config = GenerationConfig(
        max_new_tokens=4,
        min_new_tokens=4,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=5,
    )

    transitions = get_allowed_transitions(tokenizer, indices)

    predictions, targets = [], []
    for inputs, completions, prompt_lengths in tqdm(dataloader):

        def prefix_allowed_tokens_fn(batch_id, sentence):
            completion = sentence[prompt_lengths:].tolist()
            comp_len = len(completion)

            if comp_len == 0:
                res = transitions["start"]
            elif comp_len == 1:
                res = transitions["a_to_b"].get(completion[-1], [])
            elif comp_len == 2:
                res = transitions["ab_to_c"].get(tuple(completion[-2:]), [])
            elif comp_len == 3:
                res = transitions["abc_to_d"].get(tuple(completion[-3:]), [])
            else:
                res = [tokenizer.eos_token_id]

            if not res:
                return [tokenizer.eos_token_id]
            return list(res)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        batch_size = inputs.input_ids.shape[0]
        n_seq = generation_config.num_return_sequences
        reshaped_outputs = outputs.view(batch_size, n_seq, -1)

        for i in range(batch_size):
            decoded_batch = tokenizer.batch_decode(reshaped_outputs[i, :, prompt_lengths:], skip_special_tokens=False)
            predictions.append(decoded_batch)
            targets.append(completions[i])

    acc_1, acc_5, ndcg_5 = 0, 0, 0
    for prediction, target in zip(predictions, targets):
        if target == prediction[0]:
            acc_1 += 1

        if target in prediction:
            acc_5 += 1
            ndcg_5 += 1 / np.log2(prediction.index(target) + 2)

    acc_1 /= len(targets)
    acc_5 /= len(targets)
    ndcg_5 /= len(targets)

    print(f"Acc@1: {acc_1:.4f}\tAcc@5: {acc_5:.4f}\tNDCG@5: {ndcg_5:.4f}")

    with open(Path(args.model_checkpoint) / "eval_metrics.json", "w") as f:
        json.dump({"acc_1": acc_1, "acc_5": acc_5, "ndcg_5": ndcg_5}, f, indent=4)


if __name__ == "__main__":
    main()
