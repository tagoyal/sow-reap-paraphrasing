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
                 "RRC", "UCP", "VP", "WHADJP", "WHAVP", "WHNP", "WHPP", "WHADVP",
                 "X", "ROOT", "NP-TMP", "PRT"]


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


class Sentence(object):
    def __init__(self, sent):
        self.tokens = {}
        self.num_tokens = 0
        self.sent = sent
        self.tree = None

    def set_tokens(self, tokens):
        for i, (t, p) in enumerate(tokens):
            if t == "": t = "="
            self.tokens[i] = Token(t, p, i)
        self.num_tokens = len(self.tokens)

    def get_divisions(self, parse_txt_partial):

        parse_txt_partial = parse_txt_partial[1:-1]
        try:
            idx_first_lb = parse_txt_partial.index("(")
            name_const = parse_txt_partial[:idx_first_lb].strip()
            parse_txt_partial = parse_txt_partial[idx_first_lb:]
            count = 0
            partition_indices = []
            for idx in range(len(parse_txt_partial)):
                if parse_txt_partial[idx] == "(":
                    count += 1
                elif parse_txt_partial[idx] == ")":
                    count -= 1
                if count == 0:
                    partition_indices.append(idx + 1)

            partitions = []
            part_idx_prev = 0
            for i, part_idx in enumerate(partition_indices):
                partitions.append(parse_txt_partial[part_idx_prev: part_idx])
                part_idx_prev = part_idx

        except:
            temp = parse_txt_partial.split(" ")
            name_const = temp[0]
            partitions = [temp[1]]

        return name_const, partitions

    def parse_the_parse(self, parse_txt, node):

        if parse_txt.startswith("("):
            phrase_name, divisions = self.get_divisions(parse_txt)

            if node == None:
                node = Node()
                node.root = True

            node.label = phrase_name

            if phrase_name in NON_TERMINALS:
                for phrase in divisions:
                    if phrase.strip() == "":
                        continue
                    node_temp = Node()
                    node_temp.parent = node
                    node.children.append(self.parse_the_parse(phrase, node_temp))
            else:
                node.terminal = True
                node.phrase = divisions[0]

        return node


def parse_token_line(line):
    line = line.replace("[", "").replace("]", "")
    line = line.split(" ")
    token = line[0].split("=")[1]
    pos = line[3].split("=")[1]
    return token, pos


def read_next_constituency_parse(file):
    line = file.readline().strip()
    while not line.startswith("Constituency parse:"):
        line = file.readline().strip()
    parse = []
    line = file.readline().strip()
    while not line == "":
        parse.append(line)
        line = file.readline().strip()
    return parse


def print_tree(tree):
    print(tree.label)
    print(tree.phrase)
    print(tree.start_idx)
    print(tree.end_idx)
    for child in tree.children:
        print_tree(child)


def reduce_tree(tree):
    while len(tree.children) == 1:
        """
        if tree.children[0].label not in NON_TERMINALS:
            label = tree.label
        else:
            label = tree.children[0].label
        """
        tree = tree.children[0]
    # tree.label = label
    children = []
    for child in tree.children:
        child = reduce_tree(child)
        children.append(child)
    tree.children = children
    return tree


def reduce_tree_phrase(tree):
    if tree.terminal == True:
        return tree

    child_labels = [child.label for child in tree.children]
    convert_to_one_phrase = True
    for c in child_labels:
        if c in NON_TERMINALS:
            convert_to_one_phrase = False

    if convert_to_one_phrase:
        phrase = [child.phrase for child in tree.children]
        tree.phrase = " ".join(phrase)
        tree.children = []
        tree.terminal = True
    else:
        children = []
        for child in tree.children:
            c = reduce_tree_phrase(child)
            children.append(c)
        tree.children = children

    return tree


def assign_phrases(tree, phrase_start_idx):
    if tree.terminal:
        tree.start_idx = phrase_start_idx
        tree.end_idx = phrase_start_idx + len(tree.phrase.strip().split(" "))
        return (tree.phrase)
    else:
        phrase = ""
        phrase_idx_add = 0
        for child in tree.children:
            child_phrase = assign_phrases(child, phrase_start_idx + phrase_idx_add).strip()
            child.start_idx = phrase_start_idx + phrase_idx_add
            phrase_idx_add += len(child_phrase.strip().split(" "))
            child.end_idx = phrase_start_idx + phrase_idx_add
            child.phrase = child_phrase
            phrase += " " + child_phrase
            phrase = phrase.strip()

        return phrase


def get_all_phrases_in_post_order(tree):
    phrases = []
    for child in tree.children:
        child_phrase_list = get_all_phrases_in_post_order(child)
        phrases += child_phrase_list

    phrases += [tree.phrase]
    return phrases


def get_next_sentence(file):
    ### READ SENTENCE
    sent = ""
    while True:
        line = file.readline().strip()
        if line == "":
            break
        sent = " " + line
        sent = sent.strip()
    sentence_return = Sentence(sent)

    ### READ TOKENS
    assert file.readline().strip().startswith("Tokens"), "parsing error tokens"
    tokens = []
    while True:
        line = file.readline().strip()
        if line == "":
            break
        token, pos = parse_token_line(line)
        tokens.append((token, pos))
    sentence_return.set_tokens(tokens)

    ### READ CONSTITUENCY PARSE
    assert file.readline().strip().startswith("Constituency parse"), "parsing error constituency"
    parse = ""
    while True:
        line = file.readline().strip()
        if line == "":
            break
        parse += " " + line

    tree = sentence_return.parse_the_parse(parse.strip(), None)
    tree = reduce_tree(tree)
    # tree = reduce_tree_phrase(tree)
    phrase_whole = assign_phrases(tree, 0)
    tree.phrase = phrase_whole
    tree.start_idx = 0
    tree.end_idx = len(phrase_whole.split(" "))
    sentence_return.tree = tree

    #### IGNORE DEPENDENCY PARSE
    file.readline()
    assert file.readline().strip().startswith("Dependency Parse"), "parsing error dependency"
    deps = []
    while True:
        line = file.readline().strip()
        if line == "":
            break

    return sentence_return


def read_next_sentence(file):
    line = file.readline().strip()
    count = 0
    while not line.startswith("Sentence"):
        count += 1
        if count > 1000:  ### HACK assuming no dep parse is > 1000 lines, fix later
            return None
        line = file.readline().strip()
    sentence = file.readline().strip()
    return sentence


