#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/workspace/tools/SLAM-LLM/examples/asr_librispeech
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/download/wavlm/WavLM-Large.pt
llm_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/download/llama-3.2-korean-bllossom-3b

### librispeech finetuning ###
#output_dir=/workspace/tools/SLAM-LLM/examples/asr_librispeech/output/llama-3.2-korean-bllossom-3b-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-20241029
#ckpt_path=$output_dir/asr_epoch_2_step_5212
#split=test_other
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/librispeech/${split}.jsonl

### youtube drama1k finetuning with punctuation ###
#output_dir=/workspace/tools/SLAM-LLM/examples/asr_librispeech/output/llama-3.2-korean-bllossom-3b-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-202410300210
#ckpt_path=$output_dir/asr_epoch_2_step_658

### youtube drama1k finetuning with no punctuation ###
output_dir=/workspace/tools/SLAM-LLM/examples/asr_librispeech/output/llama-3.2-korean-bllossom-3b-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-202410300427
ckpt_path=$output_dir/asr_epoch_3_step_16316
split=test
#val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/crash_05-long2/${split}.jsonl
val_data_path=/workspace/tools/SLAM-LLM/examples/asr_librispeech/data/kspon-eval-clean/${split}.jsonl
decode_log=$ckpt_path/decode_kspon-eval-clean_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
python $run_dir/inference_asr_batch_bllossom.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="llama-3.2-korean-bllossom-3b" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=3072 \
        ++model_config.encoder_name=wavlm \
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
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
