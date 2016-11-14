#!/bin/bash
source ../venv/bin/activate

python -m spinn.models.fat_classifier \
    --batch_size 10 \
    --connect_tracking_comp \
    --data_type arithmetic \
    --eval_data_path ../simple.tsv \
    --eval_seq_length 25 \
    --experiment_name arithmetic \
    --log_path ../logs \
    --model_dim 50 \
    --model_type SingleSPINN \
    --seq_length 5 \
    --training_data_path ../simple.tsv \
    --word_embedding_dim 50 \
    --eval_interval_steps 50000 \
    --learning_rate 0.01 \
    --make_logits
