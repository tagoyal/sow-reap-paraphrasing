import torch, time, sys, argparse, os, codecs, h5py, csv, random
import pickle as pk
import numpy as np
from subwordnmt.apply_bpe import BPE, read_vocabulary


def pad_to_length(sent, max_length):
    out = np.zeros((max_length))
    out[:len(sent)] = sent[:max_length]
    return out


def reverse_bpe(sent, reorder):
    x = []
    r = []
    cache = ''

    for w, re in zip(sent, reorder):
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
            r.append(re)
        else:
            x.append(w)
            r.append(re)

    return ' '.join(x), r


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


def encode_data(inputs, max_length):
    count = 0
    paraphrases = {"inputs": [], "outputs": [], "input_lens": [], "output_lens": [],
                   "reordering_input": [], "reordering_output": []}

    for (s1, s2, r1, r2) in inputs:

        tokens1 = s1.split(' ')
        tokens2 = s2.split(' ')

        reordering1 = r1.split(' ')
        reordering2 = r2.split(' ')

        line1 = ' '.join(tokens1)
        line2 = ' '.join(tokens2)

        seg_sent1 = bpe.segment(line1.lower()).split()
        seg_sent2 = bpe.segment(line2.lower()).split()

        len_reordering1 = len(reordering1)
        len_reordering2 = len(reordering2)
        reordering1 = get_new_reordering(reordering1, tokens1, seg_sent1)
        reordering2 = get_new_reordering(reordering2, tokens2, seg_sent2)

        if reordering1 is None or reordering2 is None:
            continue

        reordering1 = ['0'] + reordering1 + [str(len_reordering1)]
        reordering2 = ['0'] + reordering2 + [str(len_reordering2)]

        seg_sent1 = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent1 if w in pp_vocab] + [pp_vocab['EOS']]
        seg_sent2 = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent2 if w in pp_vocab] + [pp_vocab['EOS']]

        if len(seg_sent1) > max_length or len(seg_sent2) > max_length:
            "missed this one"
            continue

        seg_sent1 = seg_sent1[:max_length]
        seg_sent2 = seg_sent2[:max_length]

        if len(seg_sent1) != len(reordering1) or len(seg_sent2) != len(
                reordering2):  ####usually true went sentence contains special characters like accent or special symbols
            continue

        paraphrases["inputs"].append(pad_to_length(seg_sent1, max_length))
        paraphrases["outputs"].append(pad_to_length(seg_sent2, max_length))
        paraphrases["input_lens"].append(len(seg_sent1))
        paraphrases["output_lens"].append(len(seg_sent2))
        paraphrases["reordering_input"].append(pad_to_length(reordering1, max_length))
        paraphrases["reordering_output"].append(pad_to_length(reordering2, max_length))

        count += 1

    return paraphrases


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Seq2Seq paraphrase generator pre-processor')

    ## paraphrase model args
    parser.add_argument('--out_path', type=str, default='../data/custom',
                        help='data save path')
    parser.add_argument('--input_folder', type=str,
                        default='reap_intermediate',
                        help='input load path')
    parser.add_argument('--vocab', type=str, default='../resources/parse_vocab.pkl',
                        help='word vocabulary')
    parser.add_argument('--max_length', type=int, default=70,
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
    bpe = BPE(bpe_codes, separator='@@', vocab=bpe_vocab)

    ## load vocabulary
    pp_vocab, rev_pp_vocab = pk.load(open(args.vocab, 'rb'))

    ## load input paraphrases
    paraphrases = {"inputs": [], "outputs": [], "input_lens": [], "output_lens": [],
                   "reordering_input": [], "reordering_output": []}

    input_file = open(args.input_folder + '/reap.order')
    sentences = []
    for line in input_file.readlines():
        sentences.append(line.strip())

    sentences_final = []
    for i in range(0, len(sentences), 5):
        sentences_final.append(sentences[i: i + 4])

    paraphrases_temp = encode_data(sentences_final, args.max_length)
    paraphrases["inputs"] += paraphrases_temp["inputs"]
    paraphrases["outputs"] += paraphrases_temp["outputs"]
    paraphrases["input_lens"] += paraphrases_temp["input_lens"]
    paraphrases["output_lens"] += paraphrases_temp["output_lens"]
    paraphrases["reordering_input"] += paraphrases_temp["reordering_input"]
    paraphrases["reordering_output"] += paraphrases_temp["reordering_output"]

    paraphrases["inputs"] = np.array(paraphrases["inputs"], dtype="int32")
    paraphrases["outputs"] = np.array(paraphrases["outputs"], dtype="int32")
    paraphrases["input_lens"] = np.array(paraphrases["input_lens"], dtype="int32")
    paraphrases["output_lens"] = np.array(paraphrases["output_lens"], dtype="int32")
    paraphrases["reordering_input"] = np.array(paraphrases["reordering_input"], dtype="int32")
    paraphrases["reordering_output"] = np.array(paraphrases["reordering_output"], dtype="int32")

    print(paraphrases["inputs"].shape)

    out_path = args.out_path
    f = h5py.File(os.path.join(out_path, 'reap_train.hdf5'), 'w')
    train_size = int(args.train_size * len(paraphrases["inputs"]))

    f.create_dataset("inputs", data=paraphrases["inputs"][:train_size], dtype="int32")
    f.create_dataset("outputs", data=paraphrases["outputs"][:train_size], dtype="int32")
    f.create_dataset("input_lens", data=paraphrases["input_lens"][:train_size], dtype="int32")
    f.create_dataset("output_lens", data=paraphrases["output_lens"][:train_size], dtype="int32")
    f.create_dataset("reordering_input", data=paraphrases["reordering_input"][:train_size], dtype="int32")
    f.create_dataset("reordering_output", data=paraphrases["reordering_output"][:train_size], dtype="int32")
    f.close()

    f = h5py.File(os.path.join(out_path, 'reap_dev.hdf5'), 'w')
    f.create_dataset("inputs", data=paraphrases["inputs"][train_size:], dtype="int32")
    f.create_dataset("outputs", data=paraphrases["outputs"][train_size:], dtype="int32")
    f.create_dataset("input_lens", data=paraphrases["input_lens"][train_size:], dtype="int32")
    f.create_dataset("output_lens", data=paraphrases["output_lens"][train_size:], dtype="int32")
    f.create_dataset("reordering_input", data=paraphrases["reordering_input"][train_size:], dtype="int32")
    f.create_dataset("reordering_output", data=paraphrases["reordering_output"][train_size:], dtype="int32")
    f.close()
