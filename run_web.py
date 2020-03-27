#-*-coding: utf-8-*-
# 위에 코드 지우면 안 됨, 샵도 그대로 둬야함
from flask import Flask, render_template, request, url_for
import subprocess
import sys
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
from bible_people.convert_PN import convert_pn_for_web
import pickle
import operator

#참조  https://www.fun-coding.org/flask_basic-2.html
#Usage : "python3 run_web.py"

# 하이퍼파라미터
parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--max_length", type=int, default=100) #Max length of input src
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--src_file", type=str, default='')
parser.add_argument("--kr_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--en_dict", type=str, default='') #path of src word to token pairs
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--k2e_model_file", type=str, default='')
parser.add_argument("--e2k_model_file", type=str, default='')
parser.add_argument("--beam_width", type=int, default=1)

EOS_token = 1

args = parser.parse_args()
k2e_model = ""
e2k_model = ""
k2e_trg_inv_dict = ""
e2k_trg_inv_dict = ""
PN_dict = ""
PN_list = ""
PN_dict_name = 'bible_people/bible_peo+loc.pkl'
lang = "k2e"


# 2. Flask 객체를 app에 할당
app = Flask(__name__)

# 3. app 객체를 이용해 라우팅 경로를 설정

# 함수 설정
def setting():
    print("Setting models")
    torch.no_grad()

    args.kr_dict='bible_people/bible_data_GYGJ+NIV/PN/subword/vocab.kr.pkl'
    args.en_dict='bible_people/bible_data_GYGJ+NIV/PN/subword/vocab.en.pkl'
    args.save_dir = './results'
    args.k2e_model_file = 'kr2en.mylstm.150.250.250.250.bible_data_GJNIV_PN'
    args.e2k_model_file = 'en2kr.mylstm.300.500.500.500.bleu05'
    args.src_file = '/home/nmt19/RNN_model/input.txt.tok.sym.pn.sub'

    kr_dict = read_dict(args.kr_dict)
    en_dict = read_dict(args.en_dict)
    with open(PN_dict_name, 'rb') as f:
        PN_dict = pickle.load(f, encoding="utf-8")

    ###한국어(키) 한글자짜리 뻄
    print(len(PN_dict))
    str_temp =""
    count = 0
    for kk, vv in PN_dict.items():
        if(len(kk) == 1):
            count += 1
            str_temp += kk
            str_temp += "\n"
    # print("count : ", count)
    one_char_name = "bible_people/name_one_char.txt"
    with open(one_char_name, 'w') as f:
        f.write(str_temp)

    key_file = open(one_char_name, "r", encoding="utf-8")

    key_line = key_file.readline()
    while key_line:
        key_line = key_line.replace('\n', '')
        PN_dict.pop(key_line)
        key_line = key_file.readline()

    key_file.close()

    ###영어(밸류)기준으로 긴단어부터해서 정렬
    # print("ch1 : ", type(PN_dict))
    ###딕트를 정렬하니까 리스트로 바뀜
    PN_list= sorted(PN_dict.items(), key=operator.itemgetter(1), reverse=True)
    # print("ch2 : ", type(PN_dict))
    print("ch3 : ", type(PN_list[0][0]))
    print(PN_list[0][0])
    print(PN_list[0][1])
    # print(PN_dict)

    print(len(PN_list))

    k2e_trg_inv_dict = dict()
    for kk, vv in en_dict.items():
        k2e_trg_inv_dict[vv] = kk

    e2k_trg_inv_dict = dict()
    for kk, vv in kr_dict.items():
        e2k_trg_inv_dict[vv] = kk

    k2e_model_name = args.save_dir + '/' + args.k2e_model_file + '.pth' + '.best.pth'
    e2k_model_name = args.save_dir + '/' + args.e2k_model_file + '.pth' + '.best.pth'

    k2e_model = torch.load(k2e_model_name)
    print("k2e best model loaded")
    e2k_model = torch.load(e2k_model_name)
    print("e2k best model loaded")
    return k2e_model, e2k_model, k2e_trg_inv_dict, e2k_trg_inv_dict, PN_list

# 4. 해당 라우팅 경로로 요청이 올 때 실행할 함수를 바로 밑에 작성해야 함
#(해당 웹페이지에 대한 경로를 URI로 만들어준다고 이해하자)
@app.route("/")
def hello():
    return render_template("k2e.html")

@app.route('/k2e_trans', methods=['POST', 'GET'])
def k2e_trans(num=None):
    #야매임
    if request.method == 'GET':
        return render_template("k2e.html")
    if request.method == 'POST':
        if request.form['src'] == "":
            return render_template("k2e.html")
        input_sen = request.form['src']
        replaced_sen = ""
        print("src_kr : " + input_sen)

        #토큰화
        text_file = open("input.txt", "w", encoding="utf8")
        text_file.write(input_sen)
        text_file.close()
        tokenizer=check_output('./tokenizer.perl en < input.txt> input.txt.tok',shell=True)

        #숫자기호화
        number_sym=call('./web_symbolize.py',shell=True) #일단 kr -> en 만 했음.
        text_file = open("input.txt.tok.sym", "r", encoding="utf8")
        replaced_sen = text_file.read()
        print("number_sym : ", replaced_sen)

        #성경인물 => P0
        lang = "k2e"
        replaced_sen, info_dict = convert_pn_for_web(replaced_sen, PN_list, lang)
        print("replaced_sen : " + replaced_sen)
        print("info_dict : ", info_dict)
        text_file.close()


        text_file = open("input.txt.tok.sym.pn", "w", encoding="utf8")
        text_file.write(replaced_sen)
        text_file.close()


        #참고하는 코드 파일로 바꿔줘야함
        apply_bpe=check_output("../subword_nmt/apply_bpe.py -c " +\
                                "./bible_people/bible_data_GYGJ+NIV/PN/subword/kr.5000.code " +\
                                "< ./input.txt.tok.sym.pn > ./input.txt.tok.sym.pn.sub", shell=True)

        #k2e 모델에 넣기
        valid_iter = TextIterator(args.src_file, args.kr_dict,
                                 batch_size=1, maxlen=1000,
                                 ahead=1, resume_num=0)
        for x_data, x_mask, cur_line, iloop in valid_iter:
            samples = translate_beam_1(k2e_model, x_data, args)
            # print("samples : ", samples)
            output = ids2words(k2e_trg_inv_dict, samples, eos_id=EOS_token)
            output = unbpe(output)

        output = output.replace(" &apos; ", "\'")
        output = output.replace(" &apos;", "\'")
        output = output.replace("&apos; ", "\'")
        output = output.replace("&apos;", "\'")
        output = output.replace(" &quot; ", "\"")
        output = output.replace(" &quot;", "\"")
        output = output.replace("&quot; ", "\"")
        output = output.replace("&quot;", "\"")

        #숫자 기호화 되돌리기
        mapping=open("mapping.sym","rb")
        num_dict=pickle.load(mapping)

        print("num_dict : ", num_dict)
        print("output1 : " + output)
        for key, value in num_dict.items(): #key : __NO / value : 25
            if key in output:
                output=output.replace(key,value)

        #__P0같은거 원래대로 변환
        for key, val in info_dict.items(): #key : __P0, val : 예수(한국어)
            # print("key : " + key)
            # print("val : " + val)
            temp = key.strip()
            if temp in output:
                # print("key2 : " + key)
                # print("val2 : " + val)
                for (PN_key, PN_val) in PN_list :
                # for PN_key, PN_val in PN_dict.items():
                    if val == PN_key:
                        # print("key : " + key)
                        # print("val : " + val)
                        # print("PN_key : " + PN_key)
                        # print("PN_val : " + PN_val)

                        # print("temp : " + temp + "\n")
                        # output = output.replace(key, PN_val)
                        output = output.replace(temp, PN_val)


        print("output2 : ", output)

        return render_template('k2e.html', src_contents = input_sen, trans_contents = output)
    else:
        return render_template("k2e.html")

@app.route('/e2k_trans', methods=['POST', 'GET'])
def e2k_trans(num=None):
    #야매임
    if request.method == 'GET':
        return render_template("e2k.html")
    if request.method == 'POST':
        print("test2")
        if request.form['src'] == "":
            return render_template("e2k.html")
        input_sen = request.form['src']
        print("src_en : ", input_sen)


        text_file = open("input.txt", "w", encoding="utf8")
        text_file.write(input_sen)
        text_file.close()

        tokenizer=check_output('./tokenizer.perl en < input.txt> input.txt.tok',shell=True)
        apply_bpe=check_output('../subword_nmt/apply_bpe.py -c ../data_05/bleu05/test/kr.10000.code < ./input.txt.tok > ./input.txt.tok.sym.sub',shell=True)

        valid_iter = TextIterator(args.src_file, args.en_dict,
                                 batch_size=1, maxlen=1000,
                                 ahead=1, resume_num=0)
        for x_data, x_mask, cur_line, iloop in valid_iter:
            samples = translate_beam_1(e2k_model, x_data, args)
            output = ids2words(e2k_trg_inv_dict, samples, eos_id=EOS_token)
            output = unbpe(output)

        output = output.replace(" &apos; ", "\'")
        print("trans_kr : ",output)
        return render_template('e2k.html', src_contents = input_sen, trans_contents = output)
    else:
        print("test3")
        return render_template("e2k.html")


#5. 메인 모듈로 실행될 때 플라스크 서버 구동 (서버로 구동한 IP 와 포트를 옵션으로 넣어줄 수 있음)
host_addr = "203.252.112.19"
port_num = "8888"

if __name__ == "__main__":
    k2e_model, e2k_model, k2e_trg_inv_dict, e2k_trg_inv_dict, PN_list = setting()
    app.run(host=host_addr, port=port_num, threaded=True)
