for city in Melbourne New-York Sao-Paulo; do
    city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    python src/train_sft_lora.py \
        --model_checkpoint meta-llama/Llama-3.1-8B \
        --max_length 1024 \
        --batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --gradient_checkpointing \
        --dataset_id CRUISEResearchGroup/Massive-STEPS-$city \
        --codebook_path data/$city_key/codebooks_0.25.csv \
        --pid_mapping_path data/$city_key/pid_mapping.csv
done