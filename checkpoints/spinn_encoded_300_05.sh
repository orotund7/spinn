#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -N spinn_encoded_300_05
#PBS -j oe
#PBS -M apd283@nyu.edu
#PBS -l mem=6GB
#PBS -l walltime=24:00:00

module load cuda/7.5.18
module load cudnn/7.0v4.0
module load numpy/intel/1.10.1

MODEL_NAME="spinn_encoded_300_05"

cd spinn
. .venv-hpc/bin/activate
cd checkpoints

export PYTHONPATH=../python:$PYTHONPATH
export THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32

export MODEL_FLAGS=" \
--batch_size 32 \
--ckpt_path $MODEL_NAME.ckpt_best   \
--connect_tracking_comp \
--data_type snli \
--embedding_data_path ../glove/glove.840B.300d.txt \
--eval_data_path ../snli_1.0/snli_1.0_dev.jsonl \
--eval_seq_length 50 \
--experiment_name $MODEL_NAME \
--init_range 0.005 \
--log_path ../logs \
--model_dim 600 \
--model_type Model1 \
--predict_use_cell \
--seq_length 50 \
--training_data_path ../snli_1.0/snli_1.0_train.jsonl \
--use_tracking_lstm  \
--use_encoded_embeddings \
--enc_embedding_dim 300 \
--word_embedding_dim 300 \
\
 --semantic_classifier_keep_rate 0.805687783361 \
 --tracking_lstm_hidden_dim 108 \
 --num_sentence_pair_combination_layers 3 \
 --embedding_keep_rate 0.842455761471 \
 --learning_rate 0.000531754201224 \
 --l2_lambda 1.66053326417e-06 \
 --scheduled_sampling_exponent_base 0.999959450491 \
 --transition_cost_scale 3.89626323815 \
"

echo "THEANO_FLAGS: $THEANO_FLAGS"
echo "python -m spinn.models.fat_classifier $MODEL_FLAGS"

python -m spinn.models.fat_classifier $MODEL_FLAGS
