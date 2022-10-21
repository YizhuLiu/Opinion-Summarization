#test
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys

import numpy as np
import torch
from rouge import Rouge, FilesRouge

from tqdm import tqdm

from fairseq import (
    checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter
from fairseq.models.bart.hub_interface import BARTHubInterface
from fairseq.models.bart import BARTModel

label=sys.argv[1]
pt_path=sys.argv[2]
data_name_path=sys.argv[3]
beam=int(sys.argv[4])
minlen=int(sys.argv[5])
maxlen=int(sys.argv[6])
lenp = float(sys.argv[7])
multi_views = sys.argv[8]
bsz = int(sys.argv[9])
ckpt = sys.argv[10]
savepath=sys.argv[11]
typ = sys.argv[12]
bart = BARTModel.from_pretrained(
    pt_path,
    checkpoint_file=ckpt,
    data_name_or_path=data_name_path
)

bart.cuda()
bart.eval()
bart.half()


count = 1
if typ == 'yelp':
   with open('../data/test100_8_15_all_all_300_pairs.source') as source, open('../data/test100_8_15_all_all_300_others.source') as source2, open('./'+savepath, 'wt', encoding='utf-8') as fout:
     s1 = source.readlines()
     s2 = source2.readlines()
            
     slines = [s1[0].strip()]
     slines2 = [s2[0].strip()]
            
     for i in tqdm(range(1, len(s1))):
                if count % bsz == 0:
                    with torch.no_grad():
                        if multi_views:
                            hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                        else:
                            hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                    slines2 = []
                
                slines.append(s1[i].strip())
                slines2.append(s2[i].strip())
            
                count += 1
                
     if slines != []:
                if multi_views:
                    hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                else:
                    hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
if typ == 'amazon':
   with open('../data/test.gold.pairs.source') as source, open('../data/test.gold.others.source') as source2, open('./'+savepath, 'wt', encoding='utf-8') as fout:
     s1 = source.readlines()
     s2 = source2.readlines()
            
     slines = [s1[0].strip()]
     slines2 = [s2[0].strip()]
            
     for i in tqdm(range(1, len(s1))):
                if count % bsz == 0:
                    with torch.no_grad():
                        if multi_views:
                            hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                        else:
                            hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                    slines2 = []
                
                slines.append(s1[i].strip())
                slines2.append(s2[i].strip())
            
                count += 1
                
     if slines != []:
                if multi_views:
                    hypotheses_batch = bart.sample(slines, sentences2 = slines2, balance = True, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                else:
                    hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenp, max_len_b=maxlen, min_len=minlen, no_repeat_ngram_size=3)
                
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
