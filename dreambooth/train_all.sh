#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/user1/jittor2024/JDiffusion/cached_path"

root="/home/user1/jittor2024/jittor-B-commit"
save_root="${root}/dreambooth/results/prompt_v1_cosine_test"
style_file="${root}/dreambooth/settings/style.json"
texture_file="${root}/dreambooth/settings/texture.json"
color_file="${root}/dreambooth/settings/color.json"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="${root}/B"
OUTPUT_DIR_PREFIX="${save_root}/style/style_"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_EPOCHS=25
LEARNING_RATE=1e-4
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=0
MAX_TRAIN_EPOCHS=500
SEED=0
GPU_COUNT=1
MAX_NUM=27

for ((folder_number = 0; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        key=$(printf "%02d" $current_folder_number)
        style_prompt=$(python read_json.py "$style_file" "$key")
        texture_prompt=$(python read_json.py "$texture_file" "$key")
        color_prompt=$(python read_json.py "$color_file" "$key")
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        CUDA_VISIBLE_DEVICES=$gpu_id
#        CUDA_VISIBLE_DEVICES="1,0"
#        PROMPT=$(printf "style_%02d" $current_folder_number)
        PROMPT=" in $style_prompt style, with a texture of $texture_prompt and with a color style of $color_prompt."
        echo "current_folder_number: $current_folder_number, PROMPT: $PROMPT"

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --instance_prompt='$PROMPT' \
            --resolution=$RESOLUTION \
            --train_batch_size=$TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --learning_rate=$LEARNING_RATE \
            --lr_scheduler=$LR_SCHEDULER \
            --lr_warmup_steps=$LR_WARMUP_STEPS \
            --num_train_epochs=$MAX_TRAIN_EPOCHS \
            --seed=$SEED \
            --checkpoint_save_epochs=$CHECKPOINTING_EPOCHS"
        eval $COMMAND &
        sleep 3
    done
    wait
done
