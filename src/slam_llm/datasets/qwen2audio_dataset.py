import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d


class QwenAudioDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 processor=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.processor = processor
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        #self.prompt = dataset_config.get("prompt", None)
        
        #self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        # self.prompt_library = [
        #     "Begin by converting the spoken words into written text. ",
        #     "Can you transcribe the speech into a written format? ",
        #     "Focus on translating the audible content into text. ",
        #     "Transcribe the speech by carefully listening to it. ",
        #     "Would you kindly write down the content of the speech? ",
        #     "Analyze the speech and create a written transcription. ",
        #     "Engage with the speech to produce a text-based version. ",
        #     "Can you document the speech in written form? ",
        #     "Transform the spoken words into text accurately. ",
        #     "How about putting the speech's content into writing? "
        # ]
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

        self.data_list = []
        if split == "train":
            for dpath in dataset_config.train_data_path.split(','):
                with open(dpath, encoding='utf-8') as fin:
                    for line in fin:
                        data_dict = json.loads(line.strip())
                        self.data_list.append(data_dict)
        else:
            for dpath in dataset_config.val_data_path.split(','):
                with open(dpath, encoding='utf-8') as fin:
                    for line in fin:
                        data_dict = json.loads(line.strip())
                        self.data_list.append(data_dict)

        # # debug
        # with open(dataset_config.train_data_path, encoding='utf-8') as fin:
        #         for line in fin:
        #             data_dict = json.loads(line.strip())
        #             self.data_list.append(data_dict)
        # if split == "train":
        #     self.data_list = self.data_list[:80]
        # else:
        #     self.data_list = self.data_list[80:100]

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
    
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)

        if not self.inference_mode:
            messages = [
                {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
                #{"role": "user", "content": [{"type": "audio", "audio_url": audio_path},{"type": "text", "text": "이 소리가 음성이면, 전사해주세요."}]},
                {"role": "user", "content": [{"type": "audio", "audio_url": audio_path},{"type": "text", "text": "음성이면 전사해주고, 그렇지 않으면 자세히 묘사해주세요."}]},
                {"role": "assistant", "content": target}
            ]
        else:
            #messages = [
            #    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
            #    {"role": "user", "content": [{"type": "audio", "audio_url": audio_path},{"type": "text", "text": "이 소리가 음성이면, 전사해주세요."}]}
            #]
            messages = [
                {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
                #{"role": "user", "content": [{"type": "audio", "audio_url": audio_path},{"type": "text", "text": "이 소리를 묘사해주세요."}]}
                {"role": "user", "content": [{"type": "audio", "audio_url": audio_path},{"type": "text", "text": "음성이면 전사해주고, 그렇지 않으면 자세히 묘사해주세요."}]}
                #{"role": "user", "content": [{"type": "audio", "audio_url": audio_path},{"type": "text", "text": "Describe this audio in detail."}]}
            ]


        if self.inference_mode:
            return {
                "text": messages,
                "audio": audio_path,
                "key": key,
                "target": target,
            }
        return {
            "text": messages,
            "audio": audio_path,
            "key": key,
        }

        '''
        audios = []
        import librosa
        audios.append(
                librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[0])

        inputs = self.processor(
                text = messages,
                audios=audios,
                return_tensors="pt",
                padding=True
        )
        # Qwen2AudioForConditionalGeneration will automatically shift the input_ids for you
        inputs["labels"] = inputs["input_ids"]
        '''


        audio_raw = whisper.load_audio(audio_path)
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        prompt = self.prompt
        if prompt is None:
            # prompt = random.choice(self.prompt_library)
            # prompt = "Transcribe speech to text. "
            prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        answer = self.answer_template.format(target)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64
        )
        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 

        conversations = [s["text"] for s in samples]

        text = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False if not self.inference_mode else True,
                tokeni_in_conversationze=False
            )
            for conversation in conversations
        ]
        #print(f"text: {text}", flush=True)

        audios = []
        audio_paths = [s["audio"] for s in samples]
        import librosa
        for audio_path in audio_paths:
            audios.append(
                librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)[0])

        inputs = self.processor(
                text=text,
                audios=audios,
                return_tensors="pt",
                padding=True
        )
        # Qwen2AudioForConditionalGeneration will automatically shift the input_ids for you
        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]
            inputs["keys"] = keys
            inputs["targets"] = targets
            return inputs

        inputs["labels"] = inputs["input_ids"]
        return inputs

        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples] #[120, 48, 82, 42]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]  #[0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])


        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask
        }



def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = QwenAudioDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
