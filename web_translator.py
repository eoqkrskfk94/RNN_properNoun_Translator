import argparse
import os

import torch
import torch.nn as nn
from nmt_main import train_model, translate_file
from deeplib.text_data import TextPairIterator, TextIterator, read_dict
from nmt_model import translate_beam_k, translate_beam_1
from deeplib.utils import timeSince, ids2words, unbpe
from subprocess import Popen, PIPE, check_output, call
from io import open

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--max_length", type=int, default=50) #Max length of input src
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--src_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--trg_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--model_file", type=str, default='')
parser.add_argument("--beam_width", type=int, default=1)

EOS_token = 1

args = parser.parse_args()

torch.no_grad()
args.src_dict='/home/nmt19/data_05/bleu05/test/vocab.kr.pkl'
args.trg_dict='/home/nmt19/data_05/bleu05/test/vocab.en.pkl'
args.save_dir = './results'
args.model_file = 'kr2en.mylstm.300.500.500.500.bleu05'
args.beam_width = 3
src_file = '/home/nmt19/RNN_model/input.kr.tok.sub'


trg_dict = read_dict(args.trg_dict)

trg_inv_dict = dict()
for kk, vv in trg_dict.items():
    trg_inv_dict[vv] = kk


file_name = args.save_dir + '/' + args.model_file + '.pth' + '.best.pth'
print("Using best model")
model = torch.load(file_name)

for i in range(3):
    input_sen = input("source: ")
    print(input_sen)

    text_file = open("input.kr", "w",encoding="utf8")
    text_file.write(input_sen)
    text_file.close()

    tokenizer=check_output('./tokenizer.perl en < input.kr> input.kr.tok',shell=True)
    apply_bpe=check_output('../subword_nmt/apply_bpe.py -c ../data_05/bleu05/test/kr.10000.code < ./input.kr.tok > ./input.kr.tok.sub',shell=True)

    valid_iter = TextIterator(src_file, args.src_dict,
                             batch_size=1, maxlen=1000,
                             ahead=1, resume_num=0)


    for x_data, x_mask, cur_line, iloop in valid_iter:

        if args.beam_width == 1:
            samples = translate_beam_1(model, x_data, args)
        else:
            samples = translate_beam_k(model, x_data, args)

        sentence = ids2words(trg_inv_dict, samples, eos_id=EOS_token)
        sentence = unbpe(sentence)
        print("trans: ",sentence)
        #print(sentence.replace("&apos;", "/''"))
        #print(sentence)
