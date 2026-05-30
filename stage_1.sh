export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=./data
DATASET=Instruments
BASE_MODEL=
INTER=./data/Instruments/Instruments.inter.json
INDEX=./data/Instruments/Instruments.RQK.index.json
OUTPUT_DIR=./ckpt/Instruments/stage_1

mkdir -p $OUTPUT_DIR

LR=3e-5

torchrun --nproc_per_node=2 --master_port=33324 warm_up.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate $LR \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z2_bf16.json \
    --bf16 \
    --only_train_response \
    --tasks item2index,index2item,fusionseqrec,itemsearch,preferenceobtain \
    --train_prompt_sample_num 1,1,1,1,1 \
    --train_data_sample_num 0,0,0,0,0 \
    --index_file $INDEX \
    --inter_file $INTER \
