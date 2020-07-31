import os
import get_word_alignment
from reap_utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='reap data creation')
    parser.add_argument('--input_file', default='../sample_test_sow_reap.txt',
                        help='input file')
    parser.add_argument('--output_folder', default='reap_intermediate',
                        help='output folder')
    args = parser.parse_args()

    file = open(args.input_file, encoding="utf-8")
    file.readline()
    file.readline()  # read two extraneous lines at the top

    outfile = open(os.path.join(args.output_folder,  "reap.order"), "w")

    sentence1_list = []
    sentence2_list = []

    i = 0
    while True:
        line = file.readline().strip()
        if line == "":
            break

        sentence1 = get_next_sentence(file)
        sentence2 = get_next_sentence(file)

        sentence1_list.append(sentence1)
        sentence2_list.append(sentence2)

    get_word_alignment.word_alignment(sentence1_list, sentence2_list, outfile)

