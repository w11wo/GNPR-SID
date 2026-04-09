for city in Melbourne New-York Sao-Paulo; do
    city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    python src/eval_next_poi.py \
        --model_checkpoint models/Llama-3.1-8B-Massive-STEPS-$city \
        --dataset_id CRUISEResearchGroup/Massive-STEPS-$city \
        --codebook_path data/$city_key/codebooks_0.25.csv \
        --pid_mapping_path data/$city_key/pid_mapping.csv \
        --temperature 0.15 \
        --top_k 50 \
        --top_p 0.95 \
        --typical_p 1.0 \
        --repetition_penalty 1.0
done