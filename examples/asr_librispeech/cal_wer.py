import sys
import difflib
import editdistance

def load_transcriptions(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    transcriptions = {line.split('\t')[0]: line.split('\t')[1] for line in lines}
    return transcriptions

def calculate_cer(reference, hypothesis):
    ref = load_transcriptions(reference)
    hyp = load_transcriptions(hypothesis)

    total_dist = 0
    total_length = 0

    for key in ref.keys():
        if key in hyp:
            r = ref[key]
            h = hyp[key]

            dist = editdistance.eval(r.split(), h.split())
            total_dist += dist
            total_length += len(r.split())

            cer = dist / len(r)

            #print(f"\nFor line with key: {key}\nReference: {r}\nHypothesis: {h}")
            #print(f'Character Error Rate: {cer}')

            d = difflib.Differ()
            diff = list(d.compare(r, h))

            #print("Differences:")
            #for d in diff:
            #    if d[0] == ' ':
            #        print(d, end='')
            #    elif d[0] == '-':
            #        print('\033[94m' + d + '\033[0m', end='') # blue for deletion
            #    elif d[0] == '+':
            #        print('\033[92m' + d + '\033[0m', end='') # green for insertion
            #    elif d[0] == '?':
            #        print('\033[91m' + d + '\033[0m', end='') # red for substitution
            #print()

        else:
            print(f"Key {key} not found in hypothesis.")

    print(f"total_dist: {total_dist}, total_length: {total_length}")
    return total_dist / total_length

#cer = calculate_cer('ref.txt', 'hyp.txt')
cer = calculate_cer(sys.argv[1], sys.argv[2])

print("\nOverall Word Error Rate(%): ", cer*100)
