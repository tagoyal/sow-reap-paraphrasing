#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import logging
from ast import literal_eval
from datetime import datetime
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from para_transformer.models import transformer, seq2seq_base
from para_transformer.utils.misc import set_global_seeds, torch_dtypes
from para_transformer.utils.config import PAD
from torch import optim
import torch, time, argparse, os, codecs, h5py, pickle, random
import numpy as np
from torch.autograd import Variable
import scipy.stats
from itertools import permutations 

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

def pad_one(vector, size, padding_idx = 0):
	vec_out = np.zeros(size)
	vec_out[:len(vector)] = vector[:size]
	return vec_out

parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--dataset-dir', metavar='DATASET_DIR',
					help='dataset dir')
parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./para_transformer/results',
					help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
					help='saved folder')
parser.add_argument('--model-config', default="{'hidden_size':256,'num_layers':4}",
					help='architecture configuration')
parser.add_argument('--device-ids', default='3',
					help='device ids assignment (e.g "0,1", {"encoder":0, "decoder":1})')
parser.add_argument('--device', default='cuda',
					help='device assignment ("cpu" or "cuda")')
parser.add_argument('--dtype', default='float',
					help='type of tensor: ' +
					' | '.join(torch_dtypes.keys()) +
					' (default: float)')
parser.add_argument('-j', '--workers', default=8, type=int,
					help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int,
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
					help='mini-batch size (default: 32)')
parser.add_argument('--keep-checkpoints', default=5, type=int,
					help='checkpoints to save')
parser.add_argument('--eval-batch-size', default=None, type=int,
					help='mini-batch size used for evaluation (default: batch-size)')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int,
					help='rank of distributed processes')
parser.add_argument('--dist-init', default='env://', type=str,
					help='init used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--optimization-config',
					default="{'epoch':0, 'optimizer':'SGD', 'lr':0.0001, 'momentum':0.9}",
					type=str, metavar='OPT',
					help='optimization regime used')
parser.add_argument('--print-freq', default=50, type=int,
					help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=300, type=int,
					help='save frequency (default: 1000)')
parser.add_argument('--eval-freq', default=2500, type=int,
					help='evaluation frequency (default: 2500)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
					help='evaluate model FILE on validation set')
parser.add_argument('--grad-clip', default='-1.', type=str,
					help='maximum grad norm value. negative for off')
parser.add_argument('--embedding-grad-clip', default=None, type=float,
					help='maximum embedding grad norm value')
parser.add_argument('--loss-scale', default=1, type=float,
					help='loss scale for mixed precision training.')
parser.add_argument('--label-smoothing', default=0, type=float,
					help='label smoothing coefficient - default 0')
parser.add_argument('--uniform-init', default=None, type=float,
					help='if value not None - init weights to U(-value,value)')
parser.add_argument('--max-length', default=None, type=int,
					help='maximum sequence length')
parser.add_argument('--max-tokens', default=None, type=int,
					help='maximum sequence tokens -- batch is trimmed if exceeded')
parser.add_argument('--fixed-length', default=None, type=int,
					help='fixed sequence length')
parser.add_argument('--chunk-batch', default=1, type=int,
					help='chunk batch size for multiple passes (training) -- used to fit large batches in memory')
parser.add_argument('--duplicates', default=1, type=int,
					help='number of duplicates over singel example')
parser.add_argument('--seed', default=123, type=int,
					help='random seed (default: 123)')
parser.add_argument('--test_data', type=str, default='./para_gen/test_mingda.hdf5',
	        		help='train data location')
parser.add_argument('--vocab', type=str, default='./para_gen/data/parse_vocab.pkl',
	        		help='word vocabulary')
parser.add_argument('--min_sent_length', type=int, default=5,
	        		help='min number of tokens per batch')
parser.add_argument('--eval_mode', type=bool, default=False,
	        		help='run beam search for some examples using a trained model')
parser.add_argument('--model', type=str, default="great8_individual3.pt",
	        		help='model location')


def encode_data():

	# load data
	h5f_train = h5py.File(args.test_data, 'r')
	inp = h5f_train['inputs']
	out = h5f_train['outputs']
	in_len = h5f_train['input_lens']
	out_len = h5f_train['output_lens']
	in_order = h5f_train['reordering_input']

	criterion = nn.NLLLoss(ignore_index=PAD)


	out_x = np.concatenate([out[:, 1:], np.zeros((out.shape[0],1))], axis=1)
	bos = Variable(torch.from_numpy(np.asarray([vocab["BOS"]]).astype('int32')).long().cuda())
	loss1_total = 0.
	loss2_total = 0.

	
	for d_idx in range(len(inp)):
		if d_idx > 39780:
			exit()

		source_sent = [rev_vocab[w] for j,w in enumerate(inp[d_idx]) if j < in_len[d_idx]]
		target_sent = [rev_vocab[w] for j,w in enumerate(out[d_idx]) if j < out_len[d_idx]]

		source_sent = reverse_bpe(source_sent)
		target_sent = reverse_bpe(target_sent)

		# torchify input
		curr_inp = Variable(torch.from_numpy(inp[d_idx].astype('int32')).long().cuda())
		curr_out = Variable(torch.from_numpy(out[d_idx].astype('int32')).long().cuda())
		curr_out_x = Variable(torch.from_numpy(out_x[d_idx].astype('int32')).long().cuda())
		in_sent_lens = torch.from_numpy(in_len[d_idx: d_idx + 1]).long().cuda()
		out_sent_lens = torch.from_numpy(out_len[d_idx: d_idx + 1]).long().cuda()
		curr_in_order = Variable(torch.from_numpy(in_order[d_idx].astype('int32')).long().cuda())

		order = [0] + [x for x in in_order[d_idx] if x >0]
		monotone_order = pad_one(np.arange(len(order)), curr_inp.shape[0])
		torch_ordering_init = Variable(torch.from_numpy(monotone_order.astype('int32')).long().cuda())

		eos = np.where(out[d_idx]==vocab['EOS'])[0][0]
		print('input: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(inp[d_idx])\
		    if j < in_len[d_idx]]))
		
		print('gt output: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(out[d_idx, :eos])\
		    if j < out_len[d_idx]]))
		
		x = model.generate(curr_inp.unsqueeze(0), [list(bos)], curr_in_order.unsqueeze(0), beam_size=10, max_sequence_length=70, top_k=20)[0]
		preds = [s.output for s in x]
		scores = [s.score for s in x]
		print("\ngt ordered output:")
		for p in preds:
			print(' '.join([rev_vocab[int(w.data.cpu())] for w in p]))
			break
		
		"""
		preds,_ = model(curr_inp.unsqueeze(0), curr_out.unsqueeze(0), curr_in_order.unsqueeze(0))
		preds = preds.view(-1, len(vocab))
		preds = nn.functional.log_softmax(preds, -1)
		loss = criterion(preds, curr_out_x.view(-1))
		loss1_total += loss.data.detach().cpu()
		print(loss.data.detach().cpu().numpy())
		"""
		
		"""
		x = model.generate(curr_inp.unsqueeze(0), [list(bos)], torch_ordering_init.unsqueeze(0), beam_size=10, max_sequence_length=70, top_k=20)[0]
		preds = [s.output for s in x]
		print("\nmonotone order output:") 
		for p in preds:
			print(' '.join([rev_vocab[int(w.data.cpu())] for w in p]))

		preds,_ = model(curr_inp.unsqueeze(0), curr_out.unsqueeze(0), torch_ordering_init.unsqueeze(0))
		preds = preds.view(-1, len(vocab))
		preds = nn.functional.log_softmax(preds, -1)
		loss = criterion(preds, curr_out_x.view(-1))
		loss2_total += loss.data.detach().cpu()
		print(loss.data.detach().cpu().numpy())
		
		sp = scipy.stats.stats.spearmanr(order[1:-1], np.arange(len(order))[1:-1])[0]
		print(sp)
		"""
		print("\n\n")
		
		
if __name__ == '__main__':
	args = parser.parse_args()
	device = args.device
	dtype = torch_dtypes.get(args.dtype)
	
	
	if 'cuda' in args.device:
		main_gpu = 3
		if isinstance(args.device_ids, tuple):
			main_gpu = args.device_ids[0]
		elif isinstance(args.device_ids, int):
			main_gpu = args.device_ids
		elif isinstance(args.device_ids, dict):
			main_gpu = args.device_ids.get('input', 0)
		torch.cuda.set_device(main_gpu)
		cudnn.benchmark = True
		device = torch.device(device, main_gpu)
	

	model_path = os.path.join(args.results_dir, args.model)
	pp_model = torch.load(model_path, map_location = device)

	vocab, rev_vocab = pickle.load(open(args.vocab, 'rb'))
	len_voc = len(vocab)

	model_config = {}
	model_config['hidden_size'] = vars(pp_model['config_args'])['model_config']['hidden_size']
	model_config['num_layers'] = vars(pp_model['config_args'])['model_config']['num_layers']
	model_config.setdefault('encoder', {})
	model_config.setdefault('decoder', {})
	model_config['encoder']['vocab_size'] = len(vocab)
	model_config['decoder']['vocab_size'] = len(vocab)
	model_config['vocab_size'] = model_config['decoder']['vocab_size']
	args.model_config = model_config
	model = transformer.Transformer(**model_config)
	model.to(device)

	model.load_state_dict(pp_model['state_dict'])
	model.eval()

	encode_data()