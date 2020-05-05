import itertools
import string
from sklearn.metrics.pairwise import cosine_similarity
import copy, codecs, pickle
from processing.subwordnmt.apply_bpe import BPE, read_vocabulary
from .models.transformer import Transformer
import numpy as np
from torch.autograd import Variable
import torch


class Token(object):
    def __init__(self, word, pos, idx):
        self.word = word
        self.pos = pos
        self.idx = idx

    def __repr__(self):
        return repr(self.word)


class Node(object):
    def __init__(self):
        self.root = False
        self.children = []
        self.label = None
        self.parent = None
        self.phrase = ""
        self.terminal = False
        self.start_idx = 0
        self.end_idx = 0
        self.parent_idx = None


class Sentence(object):
    def __init__(self, sent):
        self.tokens = {}
        self.num_tokens = 0
        self.sent = sent
        self.tree = None


def get_common_parent(nodes, n1, n2):
    parents1 = get_parent_trajectory(nodes, n1)
    parents2 = get_parent_trajectory(nodes, n2)
    for i, p in enumerate(parents1):
        if p in parents2:
            closest_common_parent = p
            break
    common_parents = parents1[i:]
    return common_parents


def get_parent_trajectory(nodes, idx):
    if idx == -1:
        return []
    else:
        parent_idx = nodes[idx].parent_idx
        parents = get_parent_trajectory(nodes, parent_idx)
        parents = [parent_idx] + parents
        return parents


def remove_duplicates(ordered_list):
    if len(ordered_list) == 1:
        return ordered_list

    new_list = []
    all_reorderings = []
    for i, (w_, x_, y_, z_) in enumerate(ordered_list):
        if y_ in all_reorderings:
            continue
        else:
            new_list.append((w_, x_, y_, z_))
            all_reorderings.append(y_)

    return new_list


def get_bpe_ordering(order, toks, bpe_toks):
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


class wordEmbedding(object):
    def __init__(self, filename):
        f = open(filename)
        self.vocab2id = {}
        self.id2vocab = {}
        self.vectors = []

        id = 0
        for line in f.readlines():
            word = line.strip().split()[0]
            vector = np.array([float(x) for x in line.split()[1:]])
            self.id2vocab[id] = word
            self.vocab2id[word] = id
            self.vectors.append(vector)
            id += 1

        self.id2vocab[len(self.id2vocab)] = "UNK"
        self.vocab2id["UNK"] = len(self.vocab2id)

        self.vectors.append(np.zeros(50))
        self.vectors = np.array(self.vectors)

    def get_index(self, word):
        return self.vocab2id[word] if word in self.vocab2id else self.vocab2id["UNK"]

    def get_index_list(self, tokens):
        return [self.get_index(t) for t in tokens]


class sowModel(object):
    def __init__(self, args):
        wordEmbed_file = args.glove_file
        wordEmbed = wordEmbedding(wordEmbed_file)

        device_id = args.device_ids
        device = args.device
        device = torch.device(device, device_id)

        pp_model = torch.load(args.model_sow, map_location=device)

        bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
        bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
        bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
        bpe_vocab.add("X")
        bpe_vocab.add("Y")
        bpe = BPE(bpe_codes, separator='@@', vocab=bpe_vocab)

        pp_vocab, pp_rev_vocab = pickle.load(open(args.sow_vocab, 'rb'))
        len_voc = len(pp_vocab)
        pos_vocab, rev_pos_vocab = pickle.load(open(args.pos_vocab, "rb"))

        model_config = {}
        model_config['hidden_size'] = vars(pp_model['config_args'])['model_config']['hidden_size']
        model_config['num_layers'] = vars(pp_model['config_args'])['model_config']['num_layers']
        model_config.setdefault('encoder', {})
        model_config.setdefault('decoder', {})
        model_config['encoder']['vocab_size'] = len_voc
        model_config['decoder']['vocab_size'] = len_voc
        model_config['vocab_size'] = model_config['decoder']['vocab_size']
        model_config['postag_size'] = len(pos_vocab)
        args.model_config = model_config
        model = Transformer(**model_config)
        model.to(device)

        model.load_state_dict(pp_model['state_dict'])
        model.eval()

        bos = Variable(torch.from_numpy(np.asarray([pp_vocab["BOS"]]).astype('int32')).long().cuda())

        self.wordEmbed = wordEmbed
        self.model = model
        self.bpe = bpe
        self.pp_vocab = pp_vocab
        self.pp_rev_vocab = pp_rev_vocab
        self.pos_vocab = pos_vocab
        self.bos = bos

    def encode_single_sentence(self, phrase_tokens, pos):
        phrase = ' '.join(phrase_tokens)
        phrase_segment = self.bpe.segment(phrase).split()
        x1 = phrase_segment.index("X") + 1
        x2 = phrase_segment.index("Y") + 1

        pos = [self.pos_vocab[p] for p in pos]
        pos = get_bpe_ordering(pos, phrase_tokens, phrase_segment)

        phrase_segment = [self.pp_vocab["BOS"]] + [self.pp_vocab[w] for w in phrase_segment if w in self.pp_vocab] + \
                         [self.pp_vocab['EOS']]
        pos = [0] + pos + [0]

        if len(phrase_segment) != len(pos):
            return None, None

        phrase_segment = np.array(phrase_segment, dtype="int32")
        pos = np.array(pos, dtype="int32")
        length = np.array(len(phrase_segment), dtype="int32")

        curr_inp = Variable(torch.from_numpy(phrase_segment.astype('int32')).long().cuda())
        curr_inp_pos = Variable(torch.from_numpy(pos.astype('int32')).long().cuda())

        monotone_order = np.zeros(curr_inp.shape[0])
        monotone_order[min(x1, x2)] = 1
        monotone_order[max(x1, x2)] = 2

        reverse_order = np.zeros(curr_inp.shape[0])
        reverse_order[max(x1, x2)] = 1
        reverse_order[min(x1, x2)] = 2

        monotone_order_curr = Variable(torch.from_numpy(monotone_order.astype('int32')).long().cuda())
        reverse_order_curr = Variable(torch.from_numpy(reverse_order.astype('int32')).long().cuda())

        x = self.model.generate(curr_inp.unsqueeze(0), curr_inp_pos.unsqueeze(0), [list(self.bos)],
                                reverse_order_curr.unsqueeze(0),
                                beam_size=5, max_sequence_length=20)[0]
        preds = [s.output for s in x]
        scores = [float(s.score.detach().cpu()) for s in x]

        reordered_output = []
        for i, p in enumerate(preds):
            y = ' '.join([self.pp_rev_vocab[int(w.data.cpu())] for w in p][1:-1])
            reordered_output.append((y, scores[i]))

        x = self.model.generate(curr_inp.unsqueeze(0), curr_inp_pos.unsqueeze(0), [list(self.bos)],
                                monotone_order_curr.unsqueeze(0),
                                beam_size=5, max_sequence_length=20)[0]
        preds = [s.output for s in x]
        scores = [float(s.score.detach().cpu()) for s in x]

        monotone_output = []
        for i, p in enumerate(preds):
            y = ' '.join([self.pp_rev_vocab[int(w.data.cpu())] for w in p][1:-1])
            monotone_output.append((y, scores[i]))

        return reordered_output, monotone_output

    def get_rule_from_children(self, p, pos, childen):
        phrase = p.phrase.split(" ")
        for i, c in enumerate(childen):
            start_idx = c.start_idx - p.start_idx
            end_idx = start_idx + c.end_idx - c.start_idx

            for j in range(start_idx, end_idx):
                if i == 0:
                    phrase[j] = "X"
                else:
                    phrase[j] = "Y"
                pos[j] = c.label

        for i in range(len(childen)):
            if i == 0:
                idxs = [j for j, x in enumerate(phrase) if x == "X"]
            else:
                idxs = [j for j, x in enumerate(phrase) if x == "Y"]
            phrase = phrase[:idxs[0]] + phrase[idxs[-1]:]
            pos = pos[:idxs[0]] + pos[idxs[-1]:]

        return phrase, pos

    def get_alignment(self, input, output):
        output = output.split(' ')

        tokens_idx1 = self.wordEmbed.get_index_list(input)
        tokens_idx2 = self.wordEmbed.get_index_list(output)
        embed1 = np.take(self.wordEmbed.vectors, tokens_idx1, axis=0)
        embed2 = np.take(self.wordEmbed.vectors, tokens_idx2, axis=0)

        try:
            mat = cosine_similarity(embed1, embed2)
            mat[input.index('X'), output.index('X')] = 1
            mat[input.index('Y'), output.index('Y')] = 1

            alignments = [0] * len(input)
            for i in range(len(input)):
                alignments[i] = np.argmax(mat[i])

            i = 0
            while i < max(alignments):
                if i not in alignments:
                    next_index = min([x for x in alignments if x > i])
                    diff = next_index - i
                    for j, x in enumerate(alignments):
                        if x > i:
                            alignments[j] = alignments[j] - diff
                i += 1

            return alignments, input.index('X'), input.index('Y')
        except:
            return None, None, None

    def get_reordering_phrase(self, sent, parent, n1, n2):
        pos = [sent.tokens[x].pos for x in sent.tokens]
        pos = pos[parent.start_idx: parent.end_idx]
        phrase, pos = self.get_rule_from_children(parent, pos, [n1, n2])
        phrase_text = sent.sent.split(' ')[parent.start_idx: parent.end_idx]

        reordered_output, monotone_output = self.encode_single_sentence(phrase, pos)
        if reordered_output is None:
            return None, None, None, None, None, None

        monotone_best_score = monotone_output[0][1]

        idx_ = 0
        while True:
            reordered_output_best = reordered_output[idx_][0]
            reordered_output_best_score = reordered_output[idx_][1]
            alignment = self.get_alignment(phrase, reordered_output_best)

            if alignment[0] is not None:
                break

            idx_ += 1
            if idx_ > len(reordered_output):
                break

        if alignment[0] is None:
            return None, None, None, phrase, phrase_text, None

        phrase = ' '.join(phrase)
        phrase_text = ' '.join(phrase_text)

        # reordered_output_best_score -= monotone_best_score
        return reordered_output_best, reordered_output_best_score, monotone_best_score, phrase, phrase_text, alignment

    def get_new_alignment(self, alignment_new, alignment_x_idx, alignment_y_idx, o1_align, o2_align):
        if alignment_x_idx < alignment_y_idx:
            alignment_y_idx += len(o1_align) - 1

        o12 = [x + alignment_new[alignment_x_idx] for x in o1_align]
        for ia, idx_align in enumerate(alignment_new):
            if idx_align > alignment_new[alignment_x_idx]:
                alignment_new[ia] += max(o1_align)
        alignment_new = alignment_new[:alignment_x_idx] + o12 + alignment_new[alignment_x_idx + 1:]

        o22 = [x + alignment_new[alignment_y_idx] for x in o2_align]
        for ia, idx_align in enumerate(alignment_new):
            if idx_align > alignment_new[alignment_y_idx]:
                alignment_new[ia] += max(o2_align)
        alignment_new = alignment_new[:alignment_y_idx] + o22 + alignment_new[alignment_y_idx + 1:]

        return alignment_new

    def get_reordering(self, sent, nodes, tree, key, node_outputs):
        outputs_final = []

        list_to_prune = []
        for (n1_idx, n2_idx) in tree[key]:
            output, score, mono_score, input_rule, input_phrase, alignment = self.get_reordering_phrase(sent,
                                                                                                        nodes[key],
                                                                                                        nodes[n1_idx],
                                                                                                        nodes[n2_idx])
            if output is None:
                continue
            output_phrase = output.replace("X", nodes[n1_idx].phrase.strip()).replace("Y", nodes[n2_idx].phrase.strip())
            list_to_prune.append(
                [output, score, mono_score, input_rule, input_phrase, output_phrase, alignment, n1_idx, n2_idx])

        output_phrases = {}
        best_mono_score = -100
        best_mono_input = ""
        n1_mono_best = None
        n2_mono_best = None
        for i, (_, score, mono_score, input_rule, _, output_phrase, _, n1_idx, n2_idx) in enumerate(list_to_prune):
            if output_phrase in output_phrases.keys():
                if score > output_phrases[output_phrase][0]:
                    output_phrases[output_phrase] = (score, i)
            else:
                output_phrases[output_phrase] = (score, i)

            if mono_score > best_mono_score:
                best_mono_score = mono_score
                best_mono_input = input_rule
                n1_mono_best = n1_idx
                n2_mono_best = n2_idx

        pruned_list = []
        for p in output_phrases:
            pruned_list.append(list_to_prune[output_phrases[p][1]])

        mono_alignment = list(range(len(best_mono_input.split(' '))))
        mono_alignment_x_idx = best_mono_input.split(' ').index('X')
        mono_alignment_y_idx = best_mono_input.split(' ').index('Y')
        alignmnent = (mono_alignment, mono_alignment_x_idx, mono_alignment_y_idx)
        pruned_list.append(
            (
                best_mono_input, 0, 0, best_mono_input, input_phrase, input_phrase, alignmnent, n1_mono_best,
                n2_mono_best))

        for output, score, mono_score, input_rule, input_phrase, _, alignment, n1_idx, n2_idx in pruned_list:

            output1 = []
            output2 = []
            if n1_idx in tree.keys():
                if n1_idx in node_outputs:
                    output1 = node_outputs[n1_idx]
                else:
                    output1, node_outputs = self.get_reordering(sent, nodes, tree, n1_idx, node_outputs)
            if n2_idx in tree.keys():
                if n2_idx in node_outputs:
                    output2 = node_outputs[n2_idx]
                else:
                    output2, node_outputs = self.get_reordering(sent, nodes, tree, n2_idx, node_outputs)

            if len(output1) == 0:
                l1 = len(nodes[n1_idx].phrase.split(' '))
                output1 = [(nodes[n1_idx].phrase, [], list(range(l1)), [])]

            if len(output2) == 0:
                l2 = len(nodes[n2_idx].phrase.split(' '))
                output2 = [(nodes[n2_idx].phrase, [], list(range(l2)), [])]

            pairs_ = itertools.product(output1, output2)

            for o1, o2 in pairs_:
                out_temp = output.replace("X", o1[0]).replace("Y", o2[0])
                rule_temp = input_phrase + "=>" + input_rule + "======>" + output
                alignment_new = copy.deepcopy(alignment[0])
                alignment_x_idx = copy.deepcopy(alignment[1])
                alignment_y_idx = copy.deepcopy(alignment[2])
                alignment_new = self.get_new_alignment(alignment_new, alignment_x_idx, alignment_y_idx, o1[2], o2[2])
                assert len(input_phrase.split(' ')) == len(alignment_new), "fuck it"
                if score == 0:  ### means it's monotone, kinda a hack
                    outputs_final.append((out_temp, o1[1] + o2[1], alignment_new, o1[3] + o2[3]))
                else:
                    score_temp = score  # - mono_score
                    outputs_final.append(
                        (out_temp, [(rule_temp, score_temp)] + o1[1] + o2[1], alignment_new,
                         o1[3] + o2[3] + [score_temp]))

        node_outputs[key] = outputs_final[:10]

        return outputs_final, node_outputs
