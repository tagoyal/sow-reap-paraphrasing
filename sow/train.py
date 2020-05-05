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
from models import transformer, seq2seq_base
from utils.misc import set_global_seeds, torch_dtypes
from utils.config import PAD
from torch import optim
import torch, time, argparse, os, codecs, h5py, pickle, random
import numpy as np
from torch.autograd import Variable
from utils.cross_entropy import CrossEntropyLoss

def pad_one(vector, size, padding_idx = 0):
	vec_out = np.zeros(size)
	vec_out[:len(vector)] = vector[:size]
	return vec_out


parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./models',
					help='results dir')
parser.add_argument('--model-config', default="{'hidden_size':256,'num_layers':2}",
					help='architecture configuration')
parser.add_argument('--device-ids', default=1,
					help='device ids assignment')
parser.add_argument('--device', default='cuda',
					help='device assignment ("cpu" or "cuda")')
parser.add_argument('--dtype', default='float',
					help='type of tensor: (default: float)')
parser.add_argument('--epochs', default=30, type=int,
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
					help='mini-batch size (default: 32)')
parser.add_argument('--optimization-config',
					default="{'epoch':0, 'optimizer':'SGD', 'lr':0.0001, 'momentum':0.9}",
					type=str, metavar='OPT',
					help='optimization regime used')
parser.add_argument('--save-freq', default=3000, type=int,
					help='save frequency (default: 300)')
parser.add_argument('--grad-clip', default='-1.', type=str,
					help='maximum grad norm value. negative for off')
parser.add_argument('--uniform-init', default=None, type=float,
					help='if value not None - init weights to U(-value,value)')
parser.add_argument('--seed', default=123, type=int,
					help='random seed (default: 123)')
parser.add_argument('--train_data', type=str, default='./data/train_sow.hdf5',
	        		help='train data location')
parser.add_argument('--dev_data', type=str, default='./data/dev_sow.hdf5',
	        		help='dev data location')
parser.add_argument('--vocab', type=str, default='./resources/parse_vocab_rules.pkl',
	        		help='word vocabulary')
parser.add_argument('--pos_vocab', type=str, default='./resources/pos_vocab.pkl',
	        		help='pos vocabulary')
parser.add_argument('--min_sent_length', type=int, default=5,
	        		help='min number of tokens per batch')
parser.add_argument('--model', type=str, default="test.pt",
	        		help='model location')

def main(args):

	set_global_seeds(args.seed)
	device = args.device
	dtype = torch_dtypes.get(args.dtype)
	
	if 'cuda' in args.device:
		device_id = args.device_ids
		device = torch.device(device, device_id)
		torch.cuda.set_device(device_id)

	save_path = os.path.join(args.results_dir, args.model)
	if not os.path.exists(args.results_dir):
		os.mkdir(args.results_dir)
	log_file = open(os.path.join(args.results_dir, args.model + ".log"), "w")

	regime = literal_eval(args.optimization_config)
	model_config = literal_eval(args.model_config)

	vocab, rev_vocab = pickle.load(open(args.vocab, 'rb'))
	pos, rev_pos = pickle.load(open(args.pos_vocab, "rb"))

	model_config.setdefault('encoder', {})
	model_config.setdefault('decoder', {})
	model_config['encoder']['vocab_size'] = len(vocab)
	model_config['decoder']['vocab_size'] = len(vocab)
	model_config['vocab_size'] = model_config['decoder']['vocab_size']
	model_config['postag_size'] = len(pos)
	args.model_config = model_config
	model = transformer.Transformer(**model_config)
	model.to(device)

	criterion = nn.NLLLoss(ignore_index=PAD)
	params = model.parameters()

	optimizer = optim.Adam(params, lr=regime['lr'])

	# load data, word vocab, and parse vocab
	h5f_train = h5py.File(args.train_data, 'r')
	inp_train = h5f_train['inputs']
	out_train = h5f_train['outputs']
	input_lens_train = h5f_train['input_lens']
	output_lens_train = h5f_train['output_lens']
	inp_pos_train = h5f_train['input_pos']
	out_pos_train = h5f_train['output_pos']
	inp_order_train = h5f_train['reordering_input']
	out_order_train = h5f_train['reordering_output']
	num_pos = len(inp_train[0])
	print("training samples: %d" % len(inp_train))
	#log_file.write("training samples: %d \n" % len(inp_train) )

	batch_size = 16
	h5f_dev = h5py.File(args.dev_data, 'r')
	inp_dev = h5f_dev['inputs']
	out_dev = h5f_dev['outputs']
	input_lens_dev = h5f_dev['input_lens']
	output_lens_dev = h5f_dev['output_lens']
	input_pos_dev = h5f_dev['input_pos']
	output_pos_dev = h5f_dev['output_pos']
	inp_order_dev = h5f_dev['reordering_input']
	out_order_dev = h5f_dev['reordering_output']

	train_minibatches = [(start, start + batch_size) for start in range(0, inp_train.shape[0], batch_size)][:-1]
	dev_minibatches = [(start, start + batch_size) for start in range(0, inp_dev.shape[0], batch_size)][:-1]
	random.shuffle(train_minibatches)

	log_file.write("num training batches: %d \n \n" % len(train_minibatches) )

	for ep in range(args.epochs):
		random.shuffle(train_minibatches)
		ep_loss = 0.
		start_time = time.time()
		num_batches = 0

		for b_idx, (start, end) in enumerate(train_minibatches):
			inp = inp_train[start:end]
			out = out_train[start:end]
			in_len = input_lens_train[start:end]
			out_len = output_lens_train[start:end]
			in_order = inp_order_train[start:end]
			out_order = out_order_train[start:end]
			inp_pos = inp_pos_train[start:end]
			out_pos = out_pos_train[start:end]

			# chop input based on length of last instance (for encoder efficiency)
			max_in_len = int(np.amax(in_len))
			inp = inp[:, :max_in_len]
			inp_pos = inp_pos[:, :max_in_len]
			in_order = in_order[:, :max_in_len]

			# compute max output length and chop output (for decoder efficiency)
			max_out_len = int(np.amax(out_len))
			out = out[:, :max_out_len]
			out_pos = out_pos[:, :max_out_len]
			out_order = out_order[:, :max_out_len]

			# sentences are too short
			if max_in_len < args.min_sent_length:
				continue

			swap = random.random() > 0.5
			if swap:
				inp, out = out, inp
				in_order, out_order = out_order, in_order
				inp_pos, out_pos = out_pos, inp_pos
			
			out_x = np.concatenate([out[:, 1:], np.zeros((out.shape[0],1))], axis=1)

			# torchify input
			curr_inp = Variable(torch.from_numpy(inp.astype('int32')).long().cuda())
			curr_inp_pos = Variable(torch.from_numpy(inp_pos.astype('int32')).long().cuda())
			curr_out = Variable(torch.from_numpy(out.astype('int32')).long().cuda())
			curr_out_x = Variable(torch.from_numpy(out_x.astype('int32')).long().cuda())
			curr_in_order = Variable(torch.from_numpy(in_order.astype('int32')).long().cuda())

			# forward prop
			preds = model(curr_inp, curr_out, curr_inp_pos, curr_in_order)
			preds = preds.view(-1, len(vocab))
			preds = nn.functional.log_softmax(preds, -1)
			num_batches += 1
			# compute masked loss
			loss = criterion(preds, curr_out_x.view(-1))
			optimizer.zero_grad()
			loss.backward(retain_graph=False)
			torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
			optimizer.step()
			ep_loss += loss.data.detach().cpu()

			if b_idx % (args.save_freq) == 0:
				
				to_print = random.randint(0, len(dev_minibatches)-1)
				dev_nll = 0.
				for b_dev_idx, (start, end) in enumerate(dev_minibatches):

					inp = inp_dev[start:end]
					out = out_dev[start:end]
					pos = input_pos_dev[start:end]
					in_len = input_lens_dev[start:end]
					out_len = output_lens_dev[start:end]
					in_order = inp_order_dev[start:end]
					curr_bsz = inp.shape[0]

					max_in_len = int(np.amax(in_len))
					inp = inp[:, :max_in_len]
					inp_pos = pos[:, :max_in_len]
					in_order = in_order[:, :max_in_len]

					max_out_len = int(np.amax(out_len))
					out = out[:, :max_out_len]
					out_x = np.concatenate([out[:, 1:], np.zeros((out.shape[0], 1))], axis=1)

					curr_inp = Variable(torch.from_numpy(inp.astype('int32')).long().cuda())
					curr_inp_pos = Variable(torch.from_numpy(inp_pos.astype('int32')).long().cuda())
					curr_out = Variable(torch.from_numpy(out.astype('int32')).long().cuda())
					curr_out_x = Variable(torch.from_numpy(out_x.astype('int32')).long().cuda())
					curr_in_order = Variable(torch.from_numpy(in_order.astype('int32')).long().cuda())
					
					preds = model(curr_inp, curr_out, curr_inp_pos, curr_in_order)
					preds = preds.view((-1, len(vocab)))
					preds = nn.functional.log_softmax(preds, -1)

					bos = Variable(torch.from_numpy(np.asarray([vocab["BOS"]]).astype('int32')).long().cuda())
					loss_dev = criterion(preds, curr_out_x.view(-1))
					dev_nll += loss_dev.data.detach().cpu()					
					
					preds = preds.view(curr_bsz, max_out_len, -1).cpu().data.numpy()

					if b_dev_idx == to_print:
						for i in range(min(3, curr_bsz)):
							eos = np.where(out[i]==vocab['EOS'])[0][0]
							print('input: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(inp[i])\
							    if j < in_len[i]]))
							print('gt output: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(out[i])\
							    if j < out_len[i]]))
							
							x = model.generate(curr_inp[i].unsqueeze(0), curr_inp_pos[i].unsqueeze(0), [list(bos)], curr_in_order[i].unsqueeze(0), beam_size=5, max_sequence_length=20)[0]
							preds = [s.output for s in x]
							print([' '.join([rev_vocab[int(w.data.cpu())] for w in p]) for p in preds][0])
							print("\n")

						
				print('dev nll per token: %f' % (dev_nll/float(len(dev_minibatches))))
				
				print('done with batch %d / %d in epoch %d, loss: %f, time:%d' \
				    % (b_idx, len(train_minibatches), ep, 
				       ep_loss / num_batches, time.time()-start_time))
				print('train nll per token : %f \n' % (float(ep_loss)/float(num_batches)))

				torch.save({'state_dict':model.state_dict(), 
				    'ep_loss':ep_loss / num_batches, 
				    'train_minibatches': train_minibatches,
				    'config_args': args}, save_path)

				log_file.write("epoch : %d , batch : %d" % (ep, num_batches))
				log_file.write("dev nll: %f \n" % (dev_nll/float(len(dev_minibatches))))
				log_file.write("train nll: %f \n \n" % (float(ep_loss)/float(num_batches)))

				ep_loss = 0.
				num_batches = 0.
				start_time = time.time()

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)

