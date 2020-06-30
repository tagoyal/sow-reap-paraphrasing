import os
import numpy as np
from itertools import chain, combinations
import argparse


class Phrase(object):
    """docstring for Phrase"""

    def __init__(self, phrase_idx, start_idx, end_idx, size, label, text, parent_idx, align_idx):
        super(Phrase, self).__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.label = label
        self.size = size
        self.text = text
        self.parent_idx = parent_idx
        self.childen = []
        self.phrase_idx = phrase_idx
        self.align_idx = align_idx


class Sentence(object):
    def __init__(self, sentence):
        super(Sentence, self).__init__()
        sent_text = sentence[0]
        sent_ptags = sentence[1]
        self.sent = sent_text
        self.tokens = sent_text.split(" ")
        self.pos = sent_ptags.split(" ")
        self.phrases = {}

        self.phrases[-1] = Phrase(-1, 0, len(self.tokens), len(self.tokens), "S", sent_text, None, -1)

        parents_temp = {}

        for phrase in sentence[2:]:
            phrase_chunks = phrase.split("\t")
            phrase_idx = int(phrase_chunks[0])
            align_idx = int(phrase_chunks[1])
            start_idx = int(phrase_chunks[2])
            end_idx = int(phrase_chunks[3])
            parent_idx = int(phrase_chunks[4])
            text = phrase_chunks[5]
            label = phrase_chunks[6]
            size = end_idx - start_idx
            self.phrases[phrase_idx] = Phrase(phrase_idx, start_idx, end_idx, size, label, text, parent_idx, align_idx)
            if parent_idx not in parents_temp:
                parents_temp[parent_idx] = [phrase_idx]
            else:
                parents_temp[parent_idx].append(phrase_idx)

        for idx in parents_temp.keys():
            if idx == -1:
                continue
            else:
                self.phrases[idx].childen = parents_temp[idx]


def read_from_file(file):
    next_line = file.readline().strip()
    if next_line == "":
        return None
    else:
        lines = [next_line]
        next_line = file.readline().strip()
        while next_line != "":
            lines.append(next_line)
            next_line = file.readline().strip()
        return lines


def get_parent_trajectory(nodes, idx):
    if idx == -1:
        return []
    else:
        parent_idx = nodes.phrases[idx].parent_idx
        parents = get_parent_trajectory(nodes, parent_idx)
        parents = [parent_idx] + parents
        return parents


def get_parents(phrases, p1_idx, p2_idx):
    parents1 = get_parent_trajectory(phrases, p1_idx)
    parents2 = get_parent_trajectory(phrases, p2_idx)

    for i, p in enumerate(parents1):
        if p in parents2:
            closest_common_parent = p
            break

    common_parents = parents1[i:]
    return common_parents


def fix_order(p1, p2):
    if p1.start_idx < p2.start_idx:
        return (p1, p2)
    else:
        return (p2, p1)


def get_rule_from_children(p, pos, childen):
    phrase = p.text.split(" ")
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

    phrase = " ".join(phrase)
    pos = " ".join(pos)
    return (phrase, pos)


def get_rules(s1, s2, f1, f2):
    phrase_alignments = [(p, s1.phrases[p].align_idx) for p in s1.phrases.keys()]
    phrase_pairs = combinations(phrase_alignments, 2)
    for (p11_idx, p21_idx), (p12_idx, p22_idx) in phrase_pairs:

        p11 = s1.phrases[p11_idx]
        p21 = s2.phrases[p21_idx]
        p12 = s1.phrases[p12_idx]
        p22 = s2.phrases[p22_idx]

        #### rejection cases

        ### case1: one is subset of another
        if p11.start_idx in range(p12.start_idx, p12.end_idx):
            continue
        if p12.start_idx in range(p11.start_idx, p11.end_idx):
            continue
        if p21.start_idx in range(p22.start_idx, p22.end_idx):
            continue
        if p22.start_idx in range(p21.start_idx, p21.end_idx):
            continue

        ### case 2: both are adjacent
        if p11.end_idx == p12.start_idx or p12.end_idx == p11.start_idx:
            if p22.start_idx == p21.end_idx or p22.end_idx == p21.start_idx:
                continue

        ### case 3: X's too sparse
        parents1 = get_parents(s1, p11_idx, p12_idx)
        parents2 = get_parents(s2, p21_idx, p22_idx)

        pa1 = s1.phrases[parents1[0]]
        pa2 = s2.phrases[parents2[0]]

        num_nr_1 = pa1.size - (p11.size + p12.size)
        if num_nr_1 > pa1.size / 1.7:
            continue

        num_nr_2 = pa2.size - (p21.size + p22.size)
        if num_nr_2 > pa2.size / 1.7:
            continue

        pos1 = s1.pos[pa1.start_idx: pa1.end_idx]
        pos2 = s2.pos[pa2.start_idx: pa2.end_idx]

        sent1_rule, pos1_rule = get_rule_from_children(pa1, pos1, [p11, p12])
        sent2_rule, pos2_rule = get_rule_from_children(pa2, pos2, [p21, p22])

        sent1_rule = sent1_rule.split(" ")
        idx1 = sent1_rule.index("X")
        idx2 = sent1_rule.index("Y")
        # pos1_rule = pos1_rule.split(" ")
        # sent1_rule[idx1] = pos1_rule[idx1] + "-1"
        # sent1_rule[idx2] = pos1_rule[idx2] + "-2"
        sent1_rule = " ".join(sent1_rule)

        sent2_rule = sent2_rule.split(" ")
        idx1_ = sent2_rule.index("X")
        idx2_ = sent2_rule.index("Y")
        # pos2_rule = pos2_rule.split(" ")
        # sent2_rule[idx1_] = pos2_rule[idx1_] + "-1"
        # sent2_rule[idx2_] = pos2_rule[idx2_] + "-2"
        sent2_rule = " ".join(sent2_rule)

        if sent1_rule == sent2_rule:
            continue

        x = idx1 - idx2 < 0
        y = idx1_ - idx2_ < 0

        if x != y:
            ###uncomment to generate data like the one sent to greg
            """
            f.write("sentence1:\t" + pa1.text + "\n")
            f.write("sentence2:\t" + pa2.text + "\n")
            f.write(sent1_rule + " <==> " + sent2_rule + "\n")
            f.write("\n")
            """

            f1.write(sent1_rule + "\n")
            f1.write(sent2_rule + "\n")
            f1.write(pos1_rule + "\n")
            f1.write(pos2_rule + "\n")
            f1.write("\n")
        else:
            f2.write(sent1_rule + "\n")
            f2.write(sent2_rule + "\n")
            f2.write(pos1_rule + "\n")
            f2.write(pos2_rule + "\n")
            f2.write("\n")


def read_next_sentence(file):
    lines = read_from_file(file)

    if lines is None:
        return lines

    sentence1 = [lines[idx] for idx in range(0, len(lines), 2)]
    sentence2 = [lines[idx] for idx in range(1, len(lines), 2)]

    try:
        sentence1 = Sentence(sentence1)
        sentence2 = Sentence(sentence2)
        return (sentence1, sentence2)
    except:
        "error"
        return -1


parser = argparse.ArgumentParser(description='sow training')
parser.add_argument('--input_folder', help='input file')
args = parser.parse_args()

input_file = open(os.path.join(args.input_folder, 'phrase_alignments.out'))
output_file1 = open(os.path.join(args.input_folder, 'rules_with_reordering.out'), "w")
output_file2 = open(os.path.join(args.input_folder, 'rules_no_reordering.out'), "w")
sentences = []
while True:
    next_sent = read_next_sentence(input_file)

    if next_sent is None:
        break
    elif next_sent == -1:
        continue
    else:
        sentences.append(next_sent)

for s1, s2 in sentences:
    get_rules(s1, s2, output_file1, output_file2)

output_file1.close()
output_file2.close()
