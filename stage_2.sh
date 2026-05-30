export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=./data
DATASET=Instruments
BASE_MODEL=
INTER=./data/Instruments/Instruments.inter.json
INDEX=./data/Instruments/Instruments.RQK.index.json
HEAD_ITEMS=./data/Instruments/Instruments.head.json
EMBED_FILE=./data/Instruments/Instruments.item.embedding.pt
OUTPUT_DIR=./ckpt/Instruments/stage_2

mkdir -p $OUTPUT_DIR

LR=3e-5
ALPHA=0.1
K=5

torchrun --nproc_per_node=2 --master_port=33324 finetune.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate $LR \
    --epochs 8 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 100 \
    --deepspeed ./config/ds_z2_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file $INDEX \
    --inter_file $INTER \
    --ul_alpha $ALPHA \
    --head_items $HEAD_ITEMS \
    --embed_file $EMBED_FILE \
    --negative_k $K \
    