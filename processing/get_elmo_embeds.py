from allennlp.commands.elmo import ElmoEmbedder
import os, argparse
import pickle as pk
import get_phrase_list

parser = argparse.ArgumentParser(description='sow training')
parser.add_argument('--input_file', default='../sample_test_sow_reap.txt', help='input file')
parser.add_argument('--elmo_data_dir', help='path to elmo weights and options file')
parser.add_argument('--output_folder', default='sow_intermediate', help='input file')
args = parser.parse_args()

options_file = os.path.join(args.elmo_data_dir, 'options.json')
weight_file = os.path.join(args.elmo_data_dir, 'weights.hdf5')

elmo = ElmoEmbedder(options_file, weight_file)
batch_size = 200
input_file = open(args.input_file)
input_file.readline()  ### READ TWO EXTRANEUOUS LINE OF FILE
input_file.readline()
sentences = []
while True:
	line = input_file.readline()
	if line == "":  ### REACHED END OF FILE
		break
	else:
		sentence = get_phrase_list.get_next_sentence(input_file)
		sentences.append(sentence.sent.split(' '))

embeddings = []
for i in range(int(len(sentences)/batch_size) + 1):
	sentences_curr = sentences[i*batch_size: i*batch_size + batch_size]
	embedding = elmo.embed_batch(sentences_curr)
	embeddings += embedding

output_file = open(os.path.join(args.output_folder, 'elmo.pk'), 'wb')
pk.dump(embeddings, output_file)
