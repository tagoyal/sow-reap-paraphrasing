import os
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import argparse

parser = argparse.ArgumentParser(description='sow training')
parser.add_argument('--input_file', default='../sample_test_baseline.txt',
                    help='input file')
parser.add_argument('--output_file', default='../outputs/filtered_data.txt',
                    help='output_file file')
parser.add_argument('--wordEmbed_file', default='/scratch/cluster/tanya/glove/glove.6B.50d.txt',
                    help='glove embeddings file')


class wordEmbedding(object):
    def __init__(self, filename):
        f = open(filename)
        self.vocab2id = {}
        self.id2vocab = {}
        self.vectors = []

        id = 0
        for line in f.readlines():
            word = line.strip().split()[0]
            vector = np.asarray(map(float, line.split()[1:]))
            self.id2vocab[id] = word
            self.vocab2id[word] = id
            self.vectors.append(vector)
            id += 1

        self.id2vocab[len(self.id2vocab)] = "UNK"
        self.vocab2id["UNK"] = len(self.vocab2id)

        self.vectors.append(np.zeros(50))
        self.vectors = np.asarray(self.vectors)

    def get_index(self, word):
        return self.vocab2id[word] if word in self.vocab2id else self.vocab2id["UNK"]

    def get_index_list(self, tokens):
        return [self.get_index(t) for t in tokens]


def similarity_matrix(tokens1, tokens2, wordEmbed):
    tokens_idx1 = wordEmbed.get_index_list(tokens1)
    tokens_idx2 = wordEmbed.get_index_list(tokens2)

    embed1 = np.take(wordEmbed.vectors, tokens_idx1, axis=0)
    embed2 = np.take(wordEmbed.vectors, tokens_idx2, axis=0)
    mat = cosine_similarity(embed1, embed2)

    return mat


if __name__ == '__main__':
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    wordEmbed_file = args.wordEmbed_file

    stopWords = set(stopwords.words('english'))
    punctuation = list(string.punctuation)

    wordEmbed = wordEmbedding(wordEmbed_file)
    print("finished reading word embeddings")

    total = 0.
    file = open(input_file, 'r')
    line_idx = 0

    while True:
        sent1 = file.readline().strip().lower()
        sent2 = file.readline().strip().lower()

        if sent1 == '':
            break

        tokens1 = word_tokenize(sent1)
        tokens2 = word_tokenize(sent2)

        tokens1 = [w for w in tokens1 if w not in stopWords and w not in punctuation]
        tokens2 = [w for w in tokens2 if w not in stopWords and w not in punctuation]

        if len(tokens1) == 0 or len(tokens2) == 0:
            continue

        mat = similarity_matrix(tokens1, tokens2, wordEmbed)
        max_indices = np.argmax(mat, axis=1)

        diff = 0
        count = 0
        for row_idx in range(mat.shape[0]):
            diff += np.abs(row_idx - max_indices[row_idx])
            count += 1

        diff = float(diff) / (float(count) * float(count))
        ratio = float(mat.shape[0]) / float(mat.shape[1])

        if 0.75 < ratio < 1.5:
            if diff > 0.35:
                output_file.write(sent1.encode('utf-8') + '\n')
                output_file.write(sent2.encode('utf-8') + '\n')
                total += 1

        line_idx += 1
        if line_idx % 100000 == 0:
            print(line_idx)

    print(total)
