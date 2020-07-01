import os
import pickle as pk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import math
import argparse
import string


class Token(object):
    def __init__(self, word, pos, idx):
        self.word = word
        self.pos = pos
        self.idx = idx

    def __repr__(self):
        return repr(self.word)


NON_TERMINALS = ["S", "SBAR", "SQ", "SBARQ", "SINV",
                 "ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST",
                 "NAC", "NP", "NX", "PP", "PRN", "QP",
                 "RRC", "UCP", "VP", "WHADJP", "WHAVP", "WHNP", "WHPP",
                 "X", "ROOT"]


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


def print_tree(tree):
    print(tree.label)
    print(tree.phrase)
    for child in tree.children:
        print_tree(child)


def get_leaf_nodes(tree):
    if tree.terminal == True:
        return [tree]
    else:
        leaf_nodes = []
        for child in tree.children:
            leaf_nodes += get_leaf_nodes(child)

        return leaf_nodes


def similarity(vector1, vector2, phrase1, phrase2, idf_dictionary):
    tokens1 = phrase1.split(" ")
    tokens2 = phrase2.split(" ")
    tokens1 = [t if t in idf_dictionary.keys() else "DEFAULT" for t in tokens1]
    tokens2 = [t if t in idf_dictionary.keys() else "DEFAULT" for t in tokens2]

    weights1 = np.asarray([idf_dictionary[t] for t in tokens1])
    weights2 = np.asarray([idf_dictionary[t] for t in tokens2])
    vector1 = np.reshape(np.average(vector1, axis=0, weights=weights1), (1, -1))
    vector2 = np.reshape(np.average(vector2, axis=0, weights=weights2), (1, -1))
    return cosine_similarity(vector1, vector2)


def similarity_bert_score_type(vector1, vector2, phrase1, phrase2, idf_dictionary):
    tokens1 = phrase1.split(" ")
    tokens2 = phrase2.split(" ")
    tokens1 = [t if t in idf_dictionary.keys() else "DEFAULT" for t in tokens1]
    tokens2 = [t if t in idf_dictionary.keys() else "DEFAULT" for t in tokens2]

    sim_matrix = cosine_similarity(vector1, vector2)
    weights1 = np.asarray([idf_dictionary[t] for t in tokens1])
    weights2 = np.asarray([idf_dictionary[t] for t in tokens2])

    max_axis0 = np.max(sim_matrix, 1)
    max_axis1 = np.max(sim_matrix, 0)

    max_indices_axis0 = np.argmax(sim_matrix, 1)
    max_indices_axis1 = np.argmax(sim_matrix, 0)

    indices_0 = set([(i, j) for i, j in zip(np.arange(len(max_indices_axis0)), max_indices_axis0)])
    indices_1 = set([(i, j) for i, j in zip(max_indices_axis1, np.arange(len(max_indices_axis1)))])

    indices = indices_0.intersection(indices_1)
    indices_diff_0 = indices_0.difference(indices_1)

    indices_diff_1 = indices_1.difference(indices_0)

    R = 0
    P = 0
    for i, j in indices:
        R += weights1[i] * max_axis0[i]
        P += weights2[j] * max_axis1[j]
    # R += max_axis0[i]
    # P += max_axis1[j]
    for i, j in indices_diff_0:
        R += weights1[i] * max_axis0[i] * 0.5

    for i, j in indices_diff_1:
        P += weights2[j] * max_axis1[j] * 0.5

    R = R / float(np.sum(weights1))
    P = P / float(np.sum(weights2))

    F = 2 * (P * R) / (P + R)

    return F


global IDX_GLOBAL


def get_all_nodes(tree):
    global IDX_GLOBAL
    if IDX_GLOBAL == 0:
        tree.parent_idx = -1
    nodes = [tree]
    idx_global_init = IDX_GLOBAL
    for child in tree.children:
        child.parent_idx = idx_global_init
        IDX_GLOBAL += 1
        child_list = get_all_nodes(child)
        nodes += child_list

    return nodes


def get_similarity_matrix(nodes1, elmo1, nodes2, elmo2, idf_dictionary):
    sim_matrix = np.zeros((len(nodes1), len(nodes2)))
    for i, node1 in enumerate(nodes1):
        for j, node2 in enumerate(nodes2):
            sim_matrix[i][j] = similarity_bert_score_type(elmo1[node1.start_idx: node1.end_idx, :],
                                                          elmo2[node2.start_idx: node2.end_idx, :],
                                                          node1.phrase, node2.phrase, idf_dictionary)
    return sim_matrix


def get_label_similarity(labels1, labels2):
    sim_matrix = np.zeros((len(labels1), len(labels2)))
    for i, node1 in enumerate(labels1):
        for j, node2 in enumerate(labels2):
            if node1 == node2:
                sim_matrix[i][j] = 1

    return sim_matrix


def get_parent_trajectory(nodes, idx):
    if idx == -1:
        return []
    else:
        parent_idx = nodes[idx].parent_idx
        parents = get_parent_trajectory(nodes, parent_idx)
        parents = [parent_idx] + parents
        return parents


def get_idf(input_sentences):
    idf_dictionary = {}

    for sent in input_sentences:
        tokens = sent.sent.split(" ")
        for t in set(tokens):
            if t in idf_dictionary.keys():
                idf_dictionary[t] += 1
            else:
                idf_dictionary[t] = 1
    M = len(input_sentences)
    idf_dictionary_new = {}
    for t in idf_dictionary.keys():
        idf_dictionary_new[t] = - math.log(idf_dictionary[t] / M)
        if idf_dictionary_new[t] == 0:
            idf_dictionary_new[t] = - math.log(idf_dictionary[t]/(M + 1))  #hack to fix bug, this condition will usually not be encountered

    idf_dictionary_new["DEFAULT"] = - math.log(1 / M)
    return idf_dictionary_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sow training')
    parser.add_argument('--output_folder', default='sow_intermediate',
                        help='input/output folder')
    args = parser.parse_args()

    input_sentences = []

    input_sentences = pk.load(open(os.path.join(args.output_folder, 'phrases_sample_test_sow_reap.pkl'), 'rb'))
    input_elmo = pk.load(open(os.path.join(args.output_folder, 'elmo.pk'), 'rb'))
    f_out = open(os.path.join(args.output_folder, 'phrase_alignments.out'), 'w')

    idf_dictionary = get_idf(input_sentences)

    for idx in range(int(len(input_elmo) / 2)):

        try:
            sent1 = input_sentences[2 * idx]
            elmo1 = input_elmo[2 * idx]
            elmo1 = np.sum(elmo1, axis=0)
            sent2 = input_sentences[2 * idx + 1]
            elmo2 = input_elmo[2 * idx + 1]
            elmo2 = np.sum(elmo2, axis=0)

            pos_labels1 = [t.pos for _, t in sent1.tokens.items()]
            pos_labels2 = [t.pos for _, t in sent2.tokens.items()]

            if len(pos_labels1) != len(sent1.tree.phrase.split(" ")):
                "parsing eror"
                continue
            if len(pos_labels2) != len(sent2.tree.phrase.split(" ")):
                "parsing eror"
                continue

            assert sent1.tree.end_idx == elmo1.shape[0], "tokenization error, skipping"
            assert sent2.tree.end_idx == elmo2.shape[0], "tokenization error, skipping"

            global IDX_GLOBAL
            IDX_GLOBAL = 0
            nodes1 = get_all_nodes(sent1.tree)
            IDX_GLOBAL = 0
            nodes2 = get_all_nodes(sent2.tree)

            labels1 = [n.label for n in nodes1]
            labels2 = [n.label for n in nodes2]

            #sim_matrix_labels = get_label_similarity(labels1, labels2)

            sim_matrix = get_similarity_matrix(nodes1, elmo1, nodes2, elmo2, idf_dictionary)

            # sim_matrix = sim_matrix + 0.1 * sim_matrix_labels

            max_indices_axis1 = np.argmax(sim_matrix, 0)
            max_indices_axis0 = np.argmax(sim_matrix, 1)

            indices_0 = set([(i, j) for i, j in zip(np.arange(len(max_indices_axis0)), max_indices_axis0)])
            indices_1 = set([(i, j) for i, j in zip(max_indices_axis1, np.arange(len(max_indices_axis1)))])

            indices_intersect = indices_0.intersection(indices_1)
            if len(indices_intersect) == 0:
                continue

            f_out.write(sent1.sent + "\n")
            f_out.write(sent2.sent + "\n")
            f_out.write(" ".join(pos_labels1) + "\n")
            f_out.write(" ".join(pos_labels2) + "\n")

            for i, j in indices_intersect:
                parents1 = get_parent_trajectory(nodes1, i)
                parents2 = get_parent_trajectory(nodes2, j)
                parents1 = parents1[:-1]
                parents2 = parents2[:-1]
                least_sum = 100
                least_sum_list = []

                for p in itertools.product(parents1, parents2):
                    x1 = parents1.index(p[0])
                    x2 = parents2.index(p[1])
                    if (p[0], p[1]) in indices_intersect:

                        if x1 + x2 > least_sum:
                            continue
                        elif x1 + x2 == least_sum:
                            least_sum_list.append((p[0], p[1]))
                        elif x1 + x2 < least_sum:
                            least_sum = x1 + x2
                            least_sum_list.append((p[0], p[1]))

                if len(least_sum_list) > 0:
                    p = least_sum_list[0]
                else:
                    p = [-1, -1]

                if nodes1[i].phrase in string.punctuation and nodes2[j].phrase in string.punctuation:
                    continue

                if nodes1[i].phrase == nodes2[j].phrase:
                    if nodes1[p[0]].phrase == nodes2[p[1]].phrase:
                        continue

                f_out.write("%d\t%d\t%d\t%d\t%d\t%s\t%s\t%f\n" % (i, j, nodes1[i].start_idx, nodes1[i].end_idx, p[0],
                                                                  nodes1[i].phrase, nodes1[i].label, sim_matrix[i, j]))
                f_out.write("%d\t%d\t%d\t%d\t%d\t%s\t%s\t%f\n" % (j, i, nodes2[j].start_idx, nodes2[j].end_idx, p[1],
                                                                  nodes2[j].phrase, nodes2[j].label, sim_matrix[i, j]))
            f_out.write("\n")

        except:
            continue


