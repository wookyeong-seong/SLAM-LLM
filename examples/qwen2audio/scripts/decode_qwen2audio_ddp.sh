#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/workspace/tools/SLAM-LLM/examples/qwen2audio
cd $run_dir
code_dir=examples/qwen2audio

speech_encoder_path=
llm_path=/workspace/tools/qwen2-audio/qwen2-audio-7b-instruct

### youtube drama1k finetuning with no punctuation ###
#output_dir=/workspace/tools/SLAM-LLM/examples/asr_librispeech/output/llama-3.2-korean-phi-3.5-mini-instruct-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-202410310016
#ckpt_path=$output_dir/asr_epoch_3_step_17316

### youtube drama1k finetuning with no punctuation with LoRA ###
#output_dir=/workspace/tools/SLAM-LLM/examples/asr_librispeech/output/phi-3.5-mini-instruct-librispeech-LoRA-linear-steplrwarmupkeep1e-4-wavlm-large-202410312302
#ckpt_path=$output_dir/asr_epoch_3_step_17316

### subset of youtube drama1k finetuning with no punctuation with LoRA ###
#output_dir=/workspace/tools/SLAM-LLM/examples/qwen2audio/output/qwen2audio-202411180036
#ckpt_path=$output_dir/asr_epoch_2_step_375

### ksponspeech + youtube drama1k (< 20secs) finetuning with no punctuation with LoRA ###
#output_dir=/workspace/tools/SLAM-LLM/examples/qwen2audio/output/qwen2audio-lora-kspon-drama1k-kor-nopunc-20sec
#ckpt_path=$output_dir/asr_epoch_2_step_35728

### AAC(audiocaps + clotho + musiccaps) + ksponspeech + youtube drama1k (< 20secs) finetuning with no punctuation with LoRA ###
output_dir=/workspace/tools/SLAM-LLM/examples/qwen2audio/output/qwen2audio-lora-aac-kspon-drama1k-kor-nopunc-20sec
ckpt_path=$output_dir/asr_epoch_3_step_1428

split=test
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/kspon-eval-clean/${split}.jsonl
#decode_log=$ckpt_path/decode_kspon-eval-clean_${split}_greedy
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/crash_05-long2/${split}.jsonl
#decode_log=$ckpt_path/decode_crash_05-long2_${split}_greedy
#val_data_path=/workspace/tools/SLAM-LLM/examples/qwen2audio/data/crash_08-long2/${split}.jsonl
#decode_log=$ckpt_path/decode_crash_08-long2_${split}_greedy
#val_data_path=/workspace/tools/SLAM-LLM/examples/qwen2audio/data/audiocaps/${split}_kor_exist.jsonl
#decode_log=$ckpt_path/decode_audiocaps_${split}_greedy
#val_data_path=/workspace/tools/SLAM-LLM/examples/qwen2audio/data/clotho/${split}_kor.jsonl
#decode_log=$ckpt_path/decode_clotho_${split}_greedy
val_data_path=/workspace/tools/SLAM-LLM/examples/qwen2audio/data/musiccaps/${split}_kor.jsonl
decode_log=$ckpt_path/decode_musiccaps_${split}_greedy

# -m debugpy --listen 5678 --wait-for-client
#python $run_dir/inference_asr_batch_qwen.py \
torchrun \
	--nnodes 1 \
	--nproc_per_node 8 \
	--master_port=29503 \
	$run_dir/inference_asr_batch_qwen_ddp.py \
        ++train_config.enable_ddp=true \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="qwen2-audio-7b-instruct" \
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
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=6 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++train_config.use_peft=true \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
