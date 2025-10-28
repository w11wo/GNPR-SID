city=nyc

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