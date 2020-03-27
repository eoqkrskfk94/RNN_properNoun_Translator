import argparse
import os

import torch
import torch.nn as nn
from nmt_main import train_model, translate_file
from nmt_model import AttNMT
from deeplib.text_data import TextPairIterator, TextIterator, read_dict



parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
# arguments that has string type are the directory of that file.
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--model_file", type=str, default='')
parser.add_argument("--train_src_file", type=str, default='')
parser.add_argument("--train_trg_file", type=str, default='')
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--valid_trg_file", type=str, default='')
parser.add_argument("--trans_file", type=str, default='')
parser.add_argument("--src_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--trg_dict", type=str, default='') #path of trg word to token pairs
parser.add_argument("--max_length", type=int, default=100) #Max length of input src
parser.add_argument("--emb_act", type=int, default=0) #?
parser.add_argument("--bleu_script", type=str, default='multi-bleu.perl') #?
parser.add_argument("--rnn_name", type=str, default='gru')
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--dropout_p", type=float, default=0.1)
parser.add_argument("--emb_noise", type=float, default=0.0) #?
parser.add_argument("--reload", type=int, default=0) #?
parser.add_argument("--h0_init", type=int, default=0)
parser.add_argument("--dim_enc", type=int, default=0)
parser.add_argument("--dim_wemb", type=int, default=0)
parser.add_argument("--dim_att", type=int, default=0)
parser.add_argument("--dim_dec", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--print_every", type=int, default=100) #?
parser.add_argument("--valid_every", type=int, default=10000) #?
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5) #?
parser.add_argument("--train", type=int, default=1) #?
parser.add_argument("--trans", type=int, default=0) #?
parser.add_argument("--use_best", type=int, default=0) #?
parser.add_argument("--beam_width", type=int, default=1)
parser.add_argument("--load_model", type=int, default=0)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

print(args)

# training
if args.train:
    print ('Training...')
    train_iter = TextPairIterator(args.train_src_file, args.train_trg_file,
                        args.src_dict, args.trg_dict, batch_size=args.batch_size,
                        maxlen=args.max_length,ahead=1000, resume_num=0)
    valid_iter = TextIterator(args.valid_src_file, args.src_dict,
                        batch_size=1, maxlen=100,
                        ahead=1, resume_num=0)

    args.src_words_n = len(train_iter.src_dict2)
    args.trg_words_n = len(train_iter.trg_dict2)


    #model 사용 부분
    model = AttNMT(args=args)

    # 모델 계속 사용
    if args.load_model == 1:
        model_path = args.save_dir + '/' + args.model_file + '.pth'
        print("Loading the exist model : {}".format(model_path))
        tmp_model = torch.load(model_path)
        state_dict = tmp_model.state_dict()
        model.load_state_dict(state_dict)

    # 베스트 모델 사용
    if args.load_model == 2:
        model_path = args.save_dir + '/' + args.model_file + '.pth.best.pth'
        print("Loading the exist model : {}".format(model_path))
        tmp_model = torch.load(model_path)
        state_dict = tmp_model.state_dict()
        model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(args.device)

    train_model(args, model, train_iter, valid_iter)

if args.trans:
    print(args.model_file)
    print ('Translating...')
    bleu_score = translate_file(args)
    if bleu_score >=0:
        print ('bleu_score', bleu_score)

print ('Done')
