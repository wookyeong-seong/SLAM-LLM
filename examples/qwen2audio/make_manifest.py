# --- Building Manifest Files --- #
import json
import os

# Function to build a manifest
def build_manifest(wav_path, manifest_path):
    with open(wav_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in fin:
                elements = line.strip().split('\t')
                key = elements[0]
                audio_path = elements[1]
                transcription = elements[2].lower()
                # Write the metadata to the manifest
                metadata = {
                    "key": key,
                    "source": audio_path,
                    "target": transcription
				}
                json.dump(metadata, fout)
                fout.write('\n')
                
# Building Manifests
print("******")
#data_dir = 'data'
#data_dir = 'data/train-738h-kor-punc'
#data_dir = 'data/train-drama1k-kor-punc'
#data_dir = 'data/train-drama1k-kor-nopunc'
#data_dir = 'data/kspon-aicap/ksponspeech'
#data_dir = 'data/kspon-drama1k-kor-nopunc-20sec'
#data_dir = 'data/transcription/test_nia22_consulting3000'
data_dir = 'data/crash_08-long2'
#train_wavs = data_dir + '/train_960.scp'
#train_manifest = data_dir + '/train_960.jsonl'
#train_wavs = data_dir + '/train.scp'
#train_manifest = data_dir + '/train.jsonl'
#if not os.path.isfile(train_manifest):
#    build_manifest(train_wavs, train_manifest)
#    print("Training manifest created.")

#valid_wavs = data_dir + '/dev_other.scp'
#valid_manifest = data_dir + '/dev_other.jsonl'
#valid_wavs = data_dir + '/valid.scp'
#valid_manifest = data_dir + '/valid.jsonl'
#if not os.path.isfile(valid_manifest):
#    build_manifest(valid_wavs, valid_manifest)
#    print("Valid manifest created.")

#eval_wavs = data_dir + '/test_other.scp'
#eval_manifest = data_dir + '/test_other.jsonl'
eval_wavs = data_dir + '/test.scp'
eval_manifest = data_dir + '/test.jsonl'
if not os.path.isfile(eval_manifest):
    build_manifest(eval_wavs, eval_manifest)
    print("Eval manifest created.")
print("***Done***")
