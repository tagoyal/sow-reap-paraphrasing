import torch, time, sys, argparse, os, codecs, h5py, csv, random
import pickle as pk
import numpy as np
from subwordnmt.apply_bpe import BPE, read_vocabulary


def read_next_rule(input_file):
    s1 = str(input_file.readline().decode().strip())
    if s1 == "":
        return None
    s2 = str(input_file.readline().decode().strip())
    pos1 = str(input_file.readline().decode().strip())
    pos2 = str(input_file.readline().decode().strip())
    input_file.readline()

    return (s1, s2, pos1, pos2)


def get_new_reordering(order, toks, bpe_toks):
    bpe_idx = 0
    order_new = []
    for tok_idx, tok in enumerate(toks):
        try:
            bpe_curr = bpe_toks[bpe_idx].split("@@")[0]
            assert tok.startswith(bpe_curr), "bpe doesn't split at spaces"
            current = bpe_curr
            order_new.append(order[tok_idx])
            while current != tok:
                order_new.append(order[tok_idx])
                bpe_idx += 1
                bpe_curr = bpe_toks[bpe_idx].split("@@")[0]
                current = current + bpe_curr
            bpe_idx += 1
        except:
            return None

    return order_new


def pad_to_length(sent, max_length):
    out = np.zeros((max_length))
    out[:len(sent)] = sent[:max_length]
    return out


def encode_data(input_file, max_length):
    global COUNT

    paraphrases = {"inputs": [], "input_lens": [], "input_pos": [],
                   "outputs": [], "output_lens": [], "output_pos": [],
                   "reordering_input": [], "reordering_output": []}

    while True:
        temp = read_next_rule(input_file)
        if temp is None:
            break
        else:

            s1, s2, pos1, pos2 = temp

            if s1 == s2:
                continue

            toks1 = s1.split(" ")
            toks2 = s2.split(" ")

            pos1 = pos1.split(" ")
            pos2 = pos2.split(" ")

            for p in pos1 + pos2:
                if p not in pos_vocab.keys():
                    pos_vocab[p] = len(pos_vocab)
                    rev_pos_vocab[pos_vocab[p]] = p

            pos1 = [pos_vocab[p] for p in pos1]
            pos2 = [pos_vocab[p] for p in pos2]

            seg_sent1 = bpe.segment(s1).split()
            seg_sent2 = bpe.segment(s2).split()

            x1 = seg_sent1.index("X") + 1
            x2 = seg_sent1.index("Y") + 1

            y1 = seg_sent2.index("X") + 1
            y2 = seg_sent2.index("Y") + 1

            sign1 = x1 - x2 < 0
            sign2 = y1 - y2 < 0

            pos1 = get_new_reordering(pos1, toks1, seg_sent1)
            pos2 = get_new_reordering(pos2, toks2, seg_sent2)

            pos1 = [0] + pos1 + [0]
            pos2 = [0] + pos2 + [0]

            seg_sent1 = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent1 if w in pp_vocab] + [pp_vocab['EOS']]
            seg_sent2 = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent2 if w in pp_vocab] + [pp_vocab['EOS']]

            if len(seg_sent1) < len(pos1):
                continue

            if len(seg_sent2) < len(pos2):
                continue

            reordering1 = [0] * len(seg_sent1)
            reordering2 = [0] * len(seg_sent2)

            if sign1 == sign2:
                reordering1[min(x1, x2)] = 1
                reordering1[max(x1, x2)] = 2
                reordering2[min(y1, y2)] = 1
                reordering2[max(y1, y2)] = 2
            else:
                reordering1[max(x1, x2)] = 1
                reordering1[min(x1, x2)] = 2
                reordering2[max(y1, y2)] = 1
                reordering2[min(y1, y2)] = 2

            paraphrases["inputs"].append(pad_to_length(seg_sent1, max_length))
            paraphrases["outputs"].append(pad_to_length(seg_sent2, max_length))
            paraphrases["input_lens"].append(len(seg_sent1))
            paraphrases["output_lens"].append(len(seg_sent2))
            paraphrases["reordering_input"].append(pad_to_length(reordering1, max_length))
            paraphrases["reordering_output"].append(pad_to_length(reordering2, max_length))
            paraphrases["input_pos"].append(pad_to_length(pos1, max_length))
            paraphrases["output_pos"].append(pad_to_length(pos2, max_length))

    return paraphrases


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rule generator pre-processor')
    parser.add_argument('--out_path', type=str, default='../data',
                        help='data save path')
    parser.add_argument('--input_folder', type=str, default='sow_intermediate/',
                        help='input load path')
    parser.add_argument('--vocab', type=str, default='../resources/parse_vocab_rules.pkl',
                        help='word vocabulary')
    parser.add_argument('--pos_vocab', type=str, default='../resources/pos_vocab.pkl',
                        help='pos tag vocabulary')
    parser.add_argument('--max_length', type=int, default=70,
                        help='max sentence length')
    parser.add_argument('--split_data', type=bool, default=True,
                        help='max sentence length')

    parser.add_argument('--bpe_codes', type=str, default='../resources/bpe.codes')
    parser.add_argument('--bpe_vocab', type=str, default='../resources/vocab.txt')
    parser.add_argument('--bpe_vocab_thresh', type=int, default=50)

    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--dev_size', type=float, default=0.1)

    args = parser.parse_args()

    ## instantiate BPE segmentor
    bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
    bpe_vocab.add("X")
    bpe_vocab.add("Y")
    bpe = BPE(bpe_codes, separator='@@', vocab=bpe_vocab)

    ## load vocabulary
    pp_vocab, rev_pp_vocab = pk.load(open(args.vocab, 'rb'))
    pos_vocab, rev_pos_vocab = pk.load(open(args.pos_vocab, "rb"))

    paraphrases = {"inputs": [], "input_lens": [], "input_pos": [],
                   "outputs": [], "output_lens": [], "output_pos": [],
                   "reordering_input": [], "reordering_output": []}

    input_files = [args.input_folder + 'rules_with_reordering.out',
                     args.input_folder + 'rules_no_reordering.out']

    for file in input_files:
        input_file = open(file, 'rb')
        paraphrases_temp = encode_data(input_file, args.max_length)
        paraphrases["inputs"] += paraphrases_temp["inputs"]
        paraphrases["outputs"] += paraphrases_temp["outputs"]
        paraphrases["input_lens"] += paraphrases_temp["input_lens"]
        paraphrases["output_lens"] += paraphrases_temp["output_lens"]
        paraphrases["reordering_input"] += paraphrases_temp["reordering_input"]
        paraphrases["reordering_output"] += paraphrases_temp["reordering_output"]
        paraphrases["input_pos"] += paraphrases_temp["input_pos"]
        paraphrases["output_pos"] += paraphrases_temp["output_pos"]

    print(len(paraphrases["inputs"]))

    paraphrases["inputs"] = np.array(paraphrases["inputs"], dtype="int32")
    paraphrases["outputs"] = np.array(paraphrases["outputs"], dtype="int32")
    paraphrases["input_lens"] = np.array(paraphrases["input_lens"], dtype="int32")
    paraphrases["output_lens"] = np.array(paraphrases["output_lens"], dtype="int32")
    paraphrases["reordering_input"] = np.array(paraphrases["reordering_input"], dtype="int32")
    paraphrases["reordering_output"] = np.array(paraphrases["reordering_output"], dtype="int32")
    paraphrases["input_pos"] = np.array(paraphrases["input_pos"], dtype="int32")
    paraphrases["output_pos"] = np.array(paraphrases["output_pos"], dtype="int32")

    out_path = args.out_path
    f = h5py.File(os.path.join(out_path, 'sow_train.hdf5'), 'w')
    train_size = int(args.train_size * len(paraphrases["inputs"]))

    f.create_dataset("inputs", data=paraphrases["inputs"][:train_size], dtype="int32")
    f.create_dataset("outputs", data=paraphrases["outputs"][:train_size], dtype="int32")
    f.create_dataset("input_lens", data=paraphrases["input_lens"][:train_size], dtype="int32")
    f.create_dataset("output_lens", data=paraphrases["output_lens"][:train_size], dtype="int32")
    f.create_dataset("reordering_input", data=paraphrases["reordering_input"][:train_size], dtype="int32")
    f.create_dataset("reordering_output", data=paraphrases["reordering_output"][:train_size], dtype="int32")
    f.create_dataset("input_pos", data=paraphrases["input_pos"][:train_size], dtype="int32")
    f.create_dataset("output_pos", data=paraphrases["output_pos"][:train_size], dtype="int32")
    f.close()

    f = h5py.File(os.path.join(out_path, 'sow_dev.hdf5'), 'w')

    f.create_dataset("inputs", data=paraphrases["inputs"][train_size:], dtype="int32")
    f.create_dataset("outputs", data=paraphrases["outputs"][train_size:], dtype="int32")
    f.create_dataset("input_lens", data=paraphrases["input_lens"][train_size:], dtype="int32")
    f.create_dataset("output_lens", data=paraphrases["output_lens"][train_size:], dtype="int32")
    f.create_dataset("reordering_input", data=paraphrases["reordering_input"][train_size:], dtype="int32")
    f.create_dataset("reordering_output", data=paraphrases["reordering_output"][train_size:], dtype="int32")
    f.create_dataset("input_pos", data=paraphrases["input_pos"][train_size:], dtype="int32")
    f.create_dataset("output_pos", data=paraphrases["output_pos"][train_size:], dtype="int32")
    f.close()
