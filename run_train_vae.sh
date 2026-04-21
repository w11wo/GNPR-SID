for city in bandung beijing istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya sao_paulo shanghai sydney tangerang tokyo; do
    python src/preprocess_massive_steps.py --city $city

    python code/train_rqvae.py \
        --city $city \
        --data_path data/$city/poi_info.csv \
        --num_emb_list 64 64 64 \
        --lamda 0.25

    python code/codebook.py \
        --city $city \
        --data_path data/$city/poi_info.csv \
        --num_emb_list 64 64 64 \
        --lamda 0.25
done