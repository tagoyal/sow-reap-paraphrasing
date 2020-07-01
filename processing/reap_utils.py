class Token(object):
    def __init__(self, word, pos, idx):
        self.word = word
        self.pos = pos
        self.idx = idx
        self.parent = None
        self.children = []
        self.dep = None


class Sentence(object):
    def __init__(self, sent):
        self.tokens = {}
        self.num_tokens = 0
        self.tokens[0] = Token("ROOT", None, 0)
        self.sent = sent
        self.reordering = None

    def set_tokens(self, tokens):
        for i, (t, p) in enumerate(tokens):
            if t == "":
                t = "="
            self.tokens[i + 1] = Token(t, p, i + 1)

        self.num_tokens = len(self.tokens)

    def set_dependencies(self, deps):

        for dep, head_, child_ in deps:
            head, head_idx = head_
            child, child_idx = child_

            assert self.tokens[head_idx].word == head, "dependency head token mismatch"
            assert self.tokens[child_idx].word == child, "dependency child token mismatch"

            self.tokens[head_idx].children.append(child_idx)
            self.tokens[child_idx].parent = head_idx
            self.tokens[child_idx].dep = dep


def parse_token_line(line):
    line = line.replace("[", "").replace("]", "")
    line = line.split(" ")
    token = line[0].split("=")[1]
    pos = line[3].split("=")[1]
    return token, pos


def parse_dep_line(line):
    line = line.replace("[", "").replace("]", "")
    dep = line.split("(")[0]
    remaining = "(".join(line.split("(")[1:])
    remaining = remaining[:-1]

    toks = remaining.split(" ")
    toks[0] = toks[0][:-1]

    if list(remaining).count("-") == 2:
        head, head_idx = toks[0].split("-")
        child, child_idx = toks[1].split("-")
    else:
        try:
            head, head_idx = toks[0].split("-")
        except:
            head_idx = toks[0].split("-")[-1]
            head = toks[0][:- (len(str(head_idx)) + 1)]

        try:
            child, child_idx = toks[1].split("-")
        except:
            child_idx = toks[1].split("-")[-1]
            child = toks[1][:- (len(str(child_idx)) + 1)]

    head_idx = head_idx.replace("'", "")
    child_idx = child_idx.replace("'", "")
    return dep, (head, int(head_idx)), (child, int(child_idx))


def get_next_sentence(file):
    sent = ""
    while True:
        line = file.readline().strip()
        if line == "":
            break
        sent = " " + line
        sent = sent.strip()

    sentence_return = Sentence(sent)

    assert file.readline().strip().startswith("Tokens"), "parsing error tokens"

    tokens = []
    while True:
        line = file.readline().strip()
        if line == "":
            break
        token, pos = parse_token_line(line)
        tokens.append((token, pos))

    sentence_return.set_tokens(tokens)

    assert file.readline().strip().startswith("Constituency parse"), "parsing error constituency"
    parse = ""
    while True:
        line = file.readline().strip()
        if line == "":
            break
        parse += " " + line

    file.readline()
    assert file.readline().strip().startswith("Dependency Parse"), "parsing error dependency"

    deps = []
    while True:
        line = file.readline().strip()
        if line == "":
            break
        dep, head, child = parse_dep_line(line)
        deps.append((dep, head, child))

    sentence_return.set_dependencies(deps)

    return sentence_return
