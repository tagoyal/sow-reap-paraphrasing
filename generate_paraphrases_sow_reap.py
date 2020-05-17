from processing.get_phrase_list import get_next_sentence
from processing.subwordnmt.apply_bpe import BPE, read_vocabulary
from sow.end_to_end_generate import get_common_parent, remove_duplicates
from sow.end_to_end_generate import sowModel, get_bpe_ordering
import torch, time, sys, argparse, os, codecs, h5py, csv, random
import pickle as pk
from reap.models import transformer, seq2seq_base
from torch.autograd import Variable
import numpy as np
import itertools
import string

TAGS_TO_IGNORE = ["DT", "IN", "CD", "MD", "TO", "PRP", "PRP$", "RB", "FW", "POS"]
MAX_NUM_REORDER = 3
global IDX_GLOBAL

parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--results_file', default='./outputs/test_sow_reap.out',
                    help='results file')
parser.add_argument('--input_file', default='./sample_test_sow_reap.txt',
                    help='input file')
parser.add_argument('--glove_file', default='/scratch/cluster/tanya/glove/glove.6B.50d.txt',
                    help='glove input file')
parser.add_argument('--model_reap', default='./models/reap.pt',
                    help='model path reap')
parser.add_argument('--model_sow', default='./models/sow.pt',
                    help='model path sow')
parser.add_argument('--model-config', default="{'hidden_size':256,'num_layers':2}",
                    help='architecture configuration')
parser.add_argument('--device_ids', default=1,
                    help='device ids assignment')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--reap_vocab', type=str, default='./resources/parse_vocab.pkl',
                    help='reap word vocabulary')
parser.add_argument('--sow_vocab', type=str, default='./resources/parse_vocab_rules.pkl',
                    help='word vocabulary')
parser.add_argument('--pos_vocab', type=str, default='./resources/pos_vocab.pkl',
                    help='pos vocabulary')
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


def get_all_nodes(tree):
    global IDX_GLOBAL
    if IDX_GLOBAL == 0:
        tree.parent_idx = -1
        tree.height = 0
    nodes = [tree]
    idx_global_init = IDX_GLOBAL
    for child in tree.children:
        child.parent_idx = idx_global_init
        child.height = tree.height + 1
        IDX_GLOBAL += 1
        child_list = get_all_nodes(child)
        nodes += child_list

    return nodes


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

    sow_model_obj = sowModel(args)

    f_out = open(args.results_file, 'w')
    f_in = open(args.input_file)

    sentences = []
    f_in.readline().strip()
    f_in.readline().strip()
    while True:
        line = f_in.readline().strip()
        if line == "":  ### REACHED END OF FILE
            break
        else:
            sentence = get_next_sentence(f_in)
            sentences.append(sentence)

    start_time = time.time()
    for idx in range(0, len(sentences), 2):

        sent = sentences[idx]
        output_sent = sentences[idx + 1]
        global IDX_GLOBAL
        IDX_GLOBAL = 0
        nodes = get_all_nodes(sent.tree)

        print(idx)
        print(sent.sent)
        print("\n")

        tree = {}
        node_pairs = list(itertools.combinations(range(0, len(nodes)), 2))

        for n1_idx, n2_idx in node_pairs:
            n1 = nodes[n1_idx]
            n2 = nodes[n2_idx]

            ### reject if one is subset of another
            if n1.start_idx in range(n2.start_idx, n2.end_idx):
                continue
            elif n2.start_idx in range(n1.start_idx, n1.end_idx):
                continue

            parent = get_common_parent(nodes, n1_idx, n2_idx)

            pa = nodes[parent[0]].phrase
            pa_size = len(pa.split(' '))
            n1_size = len(n1.phrase.split(' '))
            n2_size = len(n2.phrase.split(' '))

            ### reject if too sparse

            num_nr = pa_size - (n1_size + n2_size)
            if num_nr > pa_size / 1.7:
                continue

            ### reject if nothing remains
            if num_nr == 0:
                continue

            if n1_size == 1 and sent.tokens[n1.start_idx].pos in TAGS_TO_IGNORE + list(string.punctuation):
                continue

            if n2_size == 1 and sent.tokens[n2.start_idx].pos in TAGS_TO_IGNORE + list(string.punctuation):
                continue

            if parent[0] in tree.keys():
                tree[parent[0]].append((n1_idx, n2_idx))
            else:
                tree[parent[0]] = [(n1_idx, n2_idx)]

        input = sent.sent

        try:
            minkey = min(list(tree.keys()))
            node_outputs = {}
            reorderings, _ = sow_model_obj.get_reordering(sent, nodes, tree, minkey, node_outputs)
            reorderings_stage2 = []
            for r in reorderings:
                if len(r[1]) > MAX_NUM_REORDER:
                    continue
                order = r[2]
                reordered_sent = r[0]
                if input != nodes[minkey].phrase:
                    preceeding_count = nodes[minkey].start_idx
                    following_count = len(input.split(' ')) - len(order) - preceeding_count
                    order = [x + preceeding_count for x in order]
                    order = list(range(preceeding_count)) + order
                    order = order + list(range(max(order) + 1, max(order) + 1 + following_count))
                    reordered_sent = ' '.join(input.split(' ')[:preceeding_count]) + ' ' + reordered_sent
                    reordered_sent += ' ' + ' '.join(input.split(' ')[-following_count:])
                scores = r[3]
                if len(scores) == 0:
                    scores = [0]
                reorderings_stage2.append((reordered_sent, r[1], order, scores))
                """
                print(x[3])
                print(order)
                print("\n")
                """
            reorderings_stage2.sort(key=lambda x: np.mean(x[3]), reverse=True)
            reorderings_stage2 = remove_duplicates(reorderings_stage2)[:10]
        except:
            temp = input.split(' ')
            reorderings_stage2 = [(input, ["monotone"], list(range(len(temp))), [1.])]
            print("NO REORDERING GENERATED")

        seg_sent = bpe.segment(input).split()
        tokens = input.split(' ')
        seg_sent_idx = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent if w in pp_vocab] + [pp_vocab['EOS']]

        seg_sent_output = bpe.segment(output_sent.sent).split()
        seg_sent_output_idx = [pp_vocab["BOS"]] + [pp_vocab[w] for w in seg_sent_output if w in pp_vocab]
        seg_sent_output_idx_x = [pp_vocab[w] for w in seg_sent_output if w in pp_vocab] + [pp_vocab["EOS"]]

        for x in reorderings_stage2:
            r = get_bpe_ordering(x[2], tokens, seg_sent)
            if r is None:
                continue
            r = [int(x) + 1 for x in r]
            r = [0] + r + [max(r) + 1]

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

            preds, attention = model_para(curr_inp.unsqueeze(0), curr_out.unsqueeze(0), curr_in_order.unsqueeze(0),
                                          get_attention=True)
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
            f_out.write("Input Sentence: %s \n" % sent.sent)
            f_out.write("Ground Truth Sentence: %s \n" % output_sent.sent)
            f_out.write("Reordered Sentence: %s \n" % x[0])
            f_out.write("len: " + str(len(x[1])) + "\n")
            f_out.write("Rule Sequence: \n")
            for y in x[1]:
                f_out.write(str(y) + "\n")
            temp = [str(t) for t in x[2]]
            f_out.write(' '.join(temp) + "\n")
            f_out.write("Rule Scores: %f \n" % np.sum(x[3]))
            for p, s in zip(preds_final, scores):
                f_out.write("Generated Sentence: %s \n" % p)
                f_out.write("Generation Score: %f \n" % s)
            f_out.write("Ground Truth nll: %f \n " % nll)
            f_out.write("\n")

    print(time.time() - start_time)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
