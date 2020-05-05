#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import logging
from ast import literal_eval
from datetime import datetime
import torch.nn as nn
import torch.nn.parallel
import torch, time, argparse, os, codecs, h5py, pickle, random
import numpy as np
from torch.autograd import Variable
from processing.subwordnmt.apply_bpe import BPE, read_vocabulary
from .models.transformer import Transformer
import codecs


def pad_one(vector, size, padding_idx = 0):
	vec_out = np.zeros(size)
	vec_out[:len(vector)] = vector[:size]
	return vec_out


parser = argparse.ArgumentParser(description='PyTorch Seq2Seq Training')
parser.add_argument('--results_file', default='./outputs/test_sow_reap.out',
                    help='results file')
parser.add_argument('--input_file', default='./sample_test_sow_reap.txt',
                    help='input file')
parser.add_argument('--model_reap', default='./models/reap.pt',
                    help='model path reap')
parser.add_argument('--model_sow', default='./models/sow.pt',
                    help='model path sow')
parser.add_argument('--model-config', default="{'hidden_size':256,'num_layers':2}",
                    help='architecture configuration')
parser.add_argument('--device_ids', default=1,
                    help='device ids assignment')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--reap_vocab', type=str, default='./resources/parse_vocab.pkl',
                    help='reap word vocabulary')
parser.add_argument('--sow_vocab', type=str, default='./resources/parse_vocab_rules.pkl',
                    help='word vocabulary')
parser.add_argument('--pos_vocab', type=str, default='./resources/pos_vocab.pkl',
                    help='pos vocabulary')
parser.add_argument('--bpe_codes', type=str, default='./resources/bpe.codes')
parser.add_argument('--bpe_vocab', type=str, default='./resources/vocab.txt')
parser.add_argument('--bpe_vocab_thresh', type=int, default=50)



