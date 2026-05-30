export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Instruments

DATA_PATH=./data
BASE_MODEL=
INTER=./data/Instruments/Instruments.inter.json

CKPT_PATH=
RESULTS_FILE=./ckpt/Instruments/stage_2/results.json

INDEX=./data/Instruments/Instruments.RQK.index.json

torchrun --nproc_per_node=2 --master_port=23320 test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --base_model $BASE_MODEL \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file $INDEX \
    --test_task seqrec \
    --inter_file $INTER \
    --length_penalty 1.0 \
    --head_items ./data/Instruments/Instruments.head.json \
    --tail_items ./data/Instruments/Instruments.tail.json \
