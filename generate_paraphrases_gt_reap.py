from processing.subwordnmt.apply_bpe import BPE, read_vocabulary
import torch, time, sys, argparse, os, codecs, h5py, csv, random
import pickle as pk
from reap.models import transformer, seq2seq_base
from torch.autograd import Variable
import numpy as np
import itertools
import string

parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--results_file', default='./outputs/test_reap.out',
                    help='results file')
parser.add_argument('--input_file', default='./sample_test_gt_reap.txt',
                    help='input file')
parser.add_argument('--model_reap', default='./models/reap.pt',
                    help='model path reap')
parser.add_argument('--model-config', default="{'hidden_size':256,'num_layers':2}",
                    help='architecture configuration')
parser.add_argument('--device_ids', default=1,
                    help='device ids assignment')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--reap_vocab', type=str, default='./resources/parse_vocab.pkl',
                    help='reap word vocabulary')
parser.add_argument('--bpe_codes', type=str, default='./resources/bpe.codes')
parser.add_argument('--bpe_vocab', type=str, default='./resources/vocab.txt')
parser.add_argument('--bpe_vocab_thresh', type=int, default=50)


def reverse_bpe(sent):
    x = []
    cache = ''

    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)

    return ' '.join(x)


def pad_one(vector, size, padding_idx=0):
    vec_out = np.zeros(size)
    vec_out[:len(vector)] = vector[:size]
    return vec_out


def get_new_reordering(order, toks, bpe_toks):
    bpe_idx = 0
    order_new = []

    for tok_idx, tok in enumerate(toks):
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

    return order_new


def main(args):
    bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, 50)
    bpe = BPE(bpe_codes, separator='@@', vocab=bpe_vocab)

    device_id = args.device_ids
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    ## load vocabulary and model
    pp_vocab, rev_pp_vocab = pk.load(open(args.reap_vocab, 'rb'))
    pp_model = torch.load(args.model_reap, map_location=device)
    # initialize model
    model_config = {}
    model_config['hidden_size'] = vars(pp_model['config_args'])['model_config']['hidden_size']
    model_config['num_layers'] = vars(pp_model['config_args'])['model_config']['num_layers']
    model_config.setdefault('encoder', {})
    model_config.setdefault('decoder', {})
    model_config['encoder']['vocab_size'] = len(pp_vocab)
    model_config['decoder']['vocab_size'] = len(pp_vocab)
    model_config['vocab_size'] = model_config['decoder']['vocab_size']
    model_para = transformer.Transformer(**model_config)
    model_para.to(device)
    # load weights
    model_para.load_state_dict(pp_model['state_dict'])
    model_para.eval()

    bos = Variable(torch.from_numpy(np.asarray([pp_vocab["BOS"]]).astype('int32')).long().cuda())
    criterion = torch.nn.NLLLoss(ignore_index=0)

    f_out = open(args.results_file, 'w')
    f_in = open(args.input_file)

    sentences = []
    while True:
        reorder1 = f_in.readline().strip()
        f_in.readline()
        line1 = f_in.readline().strip()
        line2 = f_in.readline().strip()
        f_in.readline()

        if reorder1 == '':
            break
        reorder1 = reorder1.split(' ')
        sentences.append((line1, line2, reorder1))

    print(len(sentences))

    for idx in range(0, len(sentences)):

        input = sentences[idx][0]
        output = sentences[idx][1]
        order = sentences[idx][2]

        print(idx)
        print(input)

        seg_sent = bpe.segment(input).split()
        tokens = input.split(' ')
        seg_sent_idx = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent if w in pp_vocab] + [pp_vocab['EOS']]

        seg_sent_output = bpe.segment(output).split()
        seg_sent_output_idx = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent_output if w in pp_vocab]
        seg_sent_output_idx_x = [pp_vocab[w] for w in seg_sent_output if w in pp_vocab] + [pp_vocab["EOS"]]

        r = get_new_reordering(order, tokens, seg_sent)
        if r is None:
            continue

        r = [int(x) + 1 for x in r]
        r = [0] + r + [max(r) + 1]  # for BOS and EOS

        if len(seg_sent_idx) != len(r):
            continue

        seg_sent_idx = np.asarray(seg_sent_idx)
        r = np.asarray(r)
        seg_sent_output_idx = np.asarray(seg_sent_output_idx)
        seg_sent_output_idx_x = np.asarray(seg_sent_output_idx_x)

        curr_inp = Variable(torch.from_numpy(seg_sent_idx.astype('int32')).long().cuda())
        curr_in_order = Variable(torch.from_numpy(r.astype('int32')).long().cuda())
        curr_out = Variable(torch.from_numpy(seg_sent_output_idx.astype('int32')).long().cuda())
        curr_out_x = Variable(torch.from_numpy(seg_sent_output_idx_x.astype('int32')).long().cuda())

        preds, _ = model_para(curr_inp.unsqueeze(0), curr_out.unsqueeze(0), curr_in_order.unsqueeze(0),
                              get_attention=False)
        preds = preds.view(-1, len(pp_vocab))
        preds = torch.nn.functional.log_softmax(preds, -1)
        nll = criterion(preds, curr_out_x.view(-1))

        gen = model_para.generate(curr_inp.unsqueeze(0), [list(bos)], curr_in_order.unsqueeze(0), beam_size=10,
                                  max_sequence_length=30, top_k=20)[0]
        preds = [s.output for s in gen]
        scores = [float(s.score.detach().cpu().numpy()) for s in gen]
        preds_final = []
        for p in preds:
            p = reverse_bpe([rev_pp_vocab[int(w.data.cpu())] for w in p])[4:-4].strip()
            preds_final.append(p)
        f_out.write("Input Sentence: %s \n" % input)
        f_out.write("Ground Truth Sentence: %s \n" % output)
        f_out.write("Reordered Sentence: %s \n" % order)
        for p, s in zip(preds_final, scores):
            f_out.write("Generated Sentence: %s \n" % p)
            f_out.write("Generation Score: %f \n" % s)
        f_out.write("Ground Truth Ppl: %f \n " % nll)
        f_out.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
