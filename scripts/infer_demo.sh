export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置 GPU 数量
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=1  # 默认值
fi

echo "GPU number: $gpu_count"

# 获取脚本路径
current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code=$(realpath "$current_dir/../SLAM-LLM")
echo "Code path: ${code}"
cd ${code}

# 设置路径
beam=1

validnum=-2

encoder_path_hf=${code}/../models/whisper-large-v3
ckpt_name=${code}/../models/srt-large/srt_large_27b.pt
llm_path=${code}/../models/gemma-3-27b-it
val_data_path=${code}/../data/s2tt/srt_demo_70.jsonl

models_dir="${code}/../models"
data_dir="${code}/../data"

# 创建目录
mkdir -p "$models_dir" "$data_dir"

# 检查并下载函数
check_and_download() {
    local path="$1"
    local url="$2"
    local name="$3"
    
    if [ ! -e "$path" ]; then
        echo "下载: $name"
        cd "$(dirname "$path")"
        git lfs clone "$url" "$(basename "$path")"
    else
        echo "已存在: $name"
    fi
}

# 检查模型和数据
check_and_download "${models_dir}/whisper-large-v3" "https://huggingface.co/openai/whisper-large-v3" "Whisper模型"
check_and_download "${models_dir}/srt-large" "https://huggingface.co/yxdu/srt-large" "SRT模型"
check_and_download "${models_dir}/gemma-3-27b-it" "https://huggingface.co/google/gemma-3-27b-it" "Gemma模型"

# 检查数据
if [ ! -f "${data_dir}/s2tt/srt_demo_70.jsonl" ]; then
    check_and_download "${data_dir}/s2tt" "https://huggingface.co/datasets/yxdu/srt-demo-s2tt-70" "演示数据"
    # 解压音频
    if [ -f "${data_dir}/s2tt/audio.tar.gz" ]; then
        echo "解压音频文件..."
        tar -zxvf "${data_dir}/s2tt/audio.tar.gz" -C "${data_dir}/s2tt"
    fi
else
    echo "已存在: 演示数据"
fi

echo "=== 检查完成 ==="




echo "${val_data_path}"
mode=srt
encoder_projector=qqm
source=all
peft=true
freeze_llm="false"

# 根据 encoder_projector 设置参数
if [ "$encoder_projector" = "qqm" ]; then
  query_len=150
  encoder_projector_ds_rate=5
  fix_length_audio=30
  peft=true
fi






echo "找到的最新 .pt 文件为: $ckpt_name"

# 设置 decode log 路径
decode_log=${code}/../output/${mode}_${source}_${encoder_projector}_test.jsonl
echo "Decode log saved to: ${decode_log}"

llm_name=$(basename "$llm_path")


# 执行推理任务
torchrun \
    --nnodes 1 \
    --nproc_per_node ${gpu_count} \
    --master_port=29503 \
    ${code}/examples/st_covost2/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=5376 \
    ++model_config.query_len=$query_len \
    ++model_config.encoder_name=whisper \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_path_hf=$encoder_path_hf \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=$encoder_projector \
    ++model_config.beam=$beam \
    ++dataset_config.dataset=st_dataset \
    ++dataset_config.file=examples/st_covost2/dataset/st_dataset.py:get_speech_dataset \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.fix_length_audio=$fix_length_audio \
    ++dataset_config.mel_size=128 \
    ++dataset_config.inference_mode=true \
    ++dataset_config.source=$source \
    ++dataset_config.mode=$mode \
    ++dataset_config.validnum=$validnum \
    ++train_config.model_name=asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=$freeze_llm \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=50 \
    ++train_config.num_workers_dataloader=25 \
    ++log_config.decode_log=$decode_log \
    ++ckpt_path=$ckpt_name \
    ++train_config.use_peft=${peft} 
done
