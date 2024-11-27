#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/workspace/tools/SLAM-LLM/examples/qwen2audio
cd $run_dir
code_dir=examples/qwen2audio

speech_encoder_path=
llm_path=/workspace/tools/qwen2-audio/qwen2-audio-7b-instruct
#train_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/librispeech/train_960h.jsonl
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/librispeech/dev_other.jsonl
#train_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/train-drama1k-kor-punc/train.jsonl
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/train-drama1k-kor-punc/valid.jsonl
#train_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/train-drama1k-kor-nopunc/train_tmp.jsonl
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/train-drama1k-kor-nopunc/valid.jsonl
train_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/kspon-drama1k-kor-nopunc-20sec/train.jsonl
val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/kspon-drama1k-kor-nopunc-20sec/valid.jsonl

#output_dir=${run_dir}/output/qwen2audio-lora-drama1k-kor-nopunc.bf16
output_dir=${run_dir}/output/qwen2audio-lora-kspon-drama1k-kor-nopunc-20sec

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=qwen2-audio-7b-instruct \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=3072 \
++model_config.encoder_name=qwen2-audio \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=6 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=2 \
++train_config.val_batch_size=2 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++train_config.use_peft=true \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
#if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
#   python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
#        --config-path "conf" \
#        --config-name "prompt.yaml" \
#        $hydra_args
#else
if true; then
    torchrun \
        --nnodes 1 \
        --nproc_per_node 8 \
        --master_port=29503 \
        $run_dir/finetune_asr_qwen2audio.py \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        ++train_config.mixed_precision=false \
        $hydra_args
fi
