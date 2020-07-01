import numpy as np
from bert_score import plot_example


def batchify(sent1, sent2, batch_size=64):
    batches = []
    for idx in range(0, len(sent1), batch_size):
        batches.append((sent1[idx: idx + batch_size], sent2[idx: idx + batch_size]))
    return batches


class Family(object):
    def __init__(self, sentence1, sentence2, head, children, order):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.head = head
        self.children = children
        self.order = order


def get_family_alignments(current_idx, source, target, alignment):
    target_alignments = []
    for token in current_idx:
        align_idx = np.argmax(alignment[token])
        target_alignments.append(align_idx)
    order = np.argsort(target_alignments)
    if len(target_alignments) == len(set(target_alignments)):
        return True, order, target_alignments
    else:
        return False, order, target_alignments


def compress_one_token(matrix, start_idx, end_idx, propogation, axis):
    if end_idx - start_idx == 1 and propogation == 0:
        return matrix

    if end_idx - start_idx == 1 and propogation > 0:
        if axis == 0:
            stack = matrix[start_idx, :]
            stack = np.reshape(stack, [1, -1])
            stack_ini = matrix[start_idx, :]
            for _ in range(1, propogation):
                stack = np.vstack((stack, stack_ini))
            matrix = np.vstack((matrix[:start_idx + 1, :], stack, matrix[start_idx + 1:, :]))
        elif axis == 1:
            stack = matrix[:, start_idx]
            stack = np.reshape(stack, [-1, 1])
            stack_ini = matrix[:, start_idx]
            for _ in range(1, propogation):
                stack = np.hstack((stack, stack_ini))
            matrix = np.hstack((matrix[:, :start_idx + 1], stack, matrix[:, start_idx + 1:]))

        return matrix

    if axis == 0:
        mat_del = matrix[range(start_idx, end_idx), :]
    elif axis == 1:
        mat_del = matrix[:, range(start_idx, end_idx)]

    mat_replacement = np.max(mat_del, axis=axis)
    matrix = np.delete(matrix, range(start_idx + 1, end_idx), axis=axis)

    if axis == 0:
        matrix[start_idx, :] = mat_replacement
    elif axis == 1:
        matrix[:, start_idx] = mat_replacement

    if propogation > 0:
        stack = mat_replacement
        if axis == 0:
            for _ in range(1, propogation):
                stack = np.vstack((stack, mat_replacement))
            matrix = np.vstack((matrix[:start_idx + 1, :], stack, matrix[start_idx + 1:, :]))
        elif axis == 1:
            for _ in range(1, propogation):
                stack = np.hstack((stack, mat_replacement))
            matrix = np.hstack((matrix[:, :start_idx + 1], stack, matrix[:, start_idx + 1:]))
    return matrix


def fix_one_axis(original, bert, matrix, axis):
    current_idx = 0
    token_idx = 0
    reductions = 0
    propogation_overall = 0

    while True:
        token = original[token_idx]
        token_idx += 1

        if token == "''": token = '"'
        if token == "-LRB-": token = "("
        if token == "-RRB-": token = ")"
        if token == "`": token = "'"
        if token == "-LCB-": token = "{"
        if token == "-RCB-": token = "}"

        if bert[current_idx].startswith("##"):
            bert[current_idx] = bert[current_idx][2:]

        propogation = 0
        while len(token) < len(bert[current_idx]):
            token += original[token_idx]
            token_idx += 1
            propogation += 1
        propogation_overall += propogation
        assert token.startswith(bert[current_idx]), "just give up already"
        start_idx = current_idx
        concat = bert[current_idx]
        current_idx += 1
        done = False
        while not done:
            if concat == token:
                done = True
            else:
                if bert[current_idx].startswith("##"):
                    bert[current_idx] = bert[current_idx][2:]
                concat += bert[current_idx]
                current_idx += 1
        end_idx = current_idx

        matrix = compress_one_token(matrix, start_idx - reductions + propogation_overall,
                                    end_idx - reductions + propogation_overall, propogation, axis)
        reductions += end_idx - start_idx - 1

        if token_idx == len(original): break

    return matrix


def compress_similarity_matrix(otokens1, otokens2, btokens1, btokens2, sim):
    sim_return = []
    for o1, o2, b1, b2, s in zip(otokens1, otokens2, btokens1, btokens2, sim):
        s = s[1:len(b1) + 1, 1: len(b2) + 1]
        try:
            s = fix_one_axis(o1, b1, s, 0)
            s = fix_one_axis(o2, b2, s, 1)
            sim_return.append(s)
        except:
            sim_return.append(None)
    return sim_return


def get_subtree_ordering(source, token, target, alignment, remaining):
    if token not in remaining:
        return None, None
    else:
        remaining.remove(token)

    if len(source.tokens[token].children) == 0:
        return source.tokens[token].word, str(source.tokens[token].idx)
    else:
        if token == 0:
            family_current = [source.tokens[x].idx - 1 for x in source.tokens[token].children]
        else:
            family_current = [token - 1] + [source.tokens[x].idx - 1 for x in
                                            source.tokens[token].children]  ## -1 because sim doesn't have root

        child_orderings = []
        child_orderings_idx = []
        for child in source.tokens[token].children:
            x, x_idx = get_subtree_ordering(source, child, target, alignment, remaining)
            if x_idx == None:
                return None, None
            child_orderings.append(x)
            child_orderings_idx.append(x_idx)

        if token != 0:
            child_orderings = [source.tokens[token].word] + child_orderings
            child_orderings_idx = [str(source.tokens[token].idx)] + child_orderings_idx

        _, order, target_alignments = get_family_alignments(family_current, source, target, alignment)

        output = [child_orderings[x] for x in order]
        output_order = [child_orderings_idx[x] for x in order]
        output = " ".join(output)

        if len(order) > 1:
            output_order = " ".join(output_order)

        return output, output_order


def get_sentence_ordering_dep(source, target, alignment):
    remaining1 = [source.tokens[x].idx for x in source.tokens]
    reordered1, reordered_idx1 = get_subtree_ordering(source, 0, target, alignment, remaining1)
    if reordered_idx1 != None:
        reordered_idx1 = reordered_idx1[0].split(" ")
        reordered_idx1 = [str(reordered_idx1.index(str(x)) + 1) for x in range(1, len(reordered_idx1) + 1)]

    remaining2 = [target.tokens[x].idx for x in target.tokens]
    reordered2, reordered_idx2 = get_subtree_ordering(target, 0, source, np.transpose(alignment), remaining2)
    if reordered_idx2 != None:
        reordered_idx2 = reordered_idx2[0].split(" ")
        reordered_idx2 = [str(reordered_idx2.index(str(x)) + 1) for x in range(1, len(reordered_idx2) + 1)]

    return reordered_idx1, reordered_idx2


def word_alignment(sentence1, sentence2, outfile):
    batches = batchify(sentence1, sentence2, batch_size=500)
    # output = []
    for batch in batches:
        b0 = [x.sent for x in batch[0]]
        b1 = [x.sent for x in batch[1]]
        otokens1 = [[x.tokens[y].word for y in x.tokens if y != 0] for x in batch[0]]
        otokens2 = [[x.tokens[y].word for y in x.tokens if y != 0] for x in batch[1]]
        # otokens1 = [b.split(' ') for b in b0]
        # otokens2 = [b.split(' ') for b in b1]

        btokens1, btokens2, sim = plot_example(b0, b1)

        sim = compress_similarity_matrix(otokens1, otokens2, btokens1, btokens2, sim)

        for i in range(len(batch[0])):
            try:
                sim[i].shape
                cont = True
            except:
                cont = False

            if cont:
                try:
                    reordered1, reordered2 = get_sentence_ordering_dep(batch[0][i], batch[1][i], sim[i])
                    if reordered1 is None or reordered2 is None:
                        continue

                    r1 = " ".join(reordered1)
                    r2 = " ".join(reordered2)
                    t1 = " ".join(otokens1[i])
                    t2 = " ".join(otokens2[i])
                    outfile.write(t1 + '\n')
                    outfile.write(t2 + '\n')
                    outfile.write(r1 + '\n')
                    outfile.write(r2 + '\n')
                    outfile.write('\n')

                except:
                    continue

