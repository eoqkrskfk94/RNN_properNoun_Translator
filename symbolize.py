#!/usr/bin/python
#-*-encoding:utf-8-*-
import argparse
import subprocess
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

SRC_FILE = 'bible_people/bible_data_GYGJ+NIV/bible_data_test.en.shuf.tok'
TRG_FILE = 'bible_people/bible_data_GYGJ+NIV/bible_data_test.kr.shuf.tok'
MAX_NUM = 10

en_word = ['hundred', 'hundreds',  'thousand', 'thousands', 'million', 'millions', 'billion', 'billions', 'trillion', 'trillions']
en_num = ['100', '100', '1000', '1000', '1000000', '1000000', '1000000000', '1000000000', '1000000000000', '1000000000000']

kr_word = [u'십', u'백', u'천', u'만', u'억']
kr_num = ['10', '100', '1000', '10000', '100000000']

symbol = ["__N0", "__N1", "__N2", "__N3", "__N4", "__N5", "__N6", "__N7", "__N8", "__N9", "__MM"]

#parser = argparse.ArgumentParse(description=''' hello ''', formatter_class=argparse.RawTextHelpFormatter)

def symbolize_kr(line):
	kr_line = line.decode('UTF-8')

	kr_line = number_unit_kr(kr_line, 1)
	kr_line = number_unit_kr(kr_line, 2)

	kr_words, kr_num = find_number(kr_line)

	count_num = 0
	sym_list = []
	for i in kr_num:
		if isNumber(kr_words[i]):
			sym_list.append(kr_words[i])
			replace2(kr_words, kr_num, kr_words[i], symbol[count_num])
			if count_num < MAX_NUM:
				count_num += 1

	kr_line = " ".join(kr_words)
	# print("kr_line : ",kr_line)
	# print("sym_list : ",sym_list)
	return kr_line, sym_list

def symbolize_kr2en(train_kr=SRC_FILE, train_en=TRG_FILE):
	global COUNT_NUM

	kr_in = open(train_kr, "r")
	kr_out = open(train_kr + ".sym", "w")
	en_in = open(train_en, "r")
	en_out = open(train_en + ".sym", "w")

	f_wordnum = open("wordsum.txt", "r")
	f_realnum = open("realsum.txt", "r")
	wordnum = f_wordnum.read().split(", ")
	realnum = f_realnum.read().split(", ")
	wordnum[-1] = wordnum[-1].replace("\n", "")
	realnum[-1] = realnum[-1].replace("\n", "")
	f_wordnum.close()
	f_realnum.close()

	loop_count = 0
	while(True):
		kr_line = kr_in.readline().decode('UTF-8')
		en_line = en_in.readline()
		if not kr_line or not en_line:
			break
		#kr_out.write(kr_line)
		#en_out.write(en_line)

		en_line = en_line.replace("\n", " ")
		for i in range(len(wordnum)):
			en_line=en_line.replace(wordnum[i]+' ', realnum[i]+' ')
			en_line=en_line.replace(wordnum[i]+'-', realnum[i]+'-')

		COUNT_NUM = 0

		en_line = number_unit_en(en_line)
		kr_line = number_unit_kr(kr_line, 1)
		kr_line = number_unit_kr(kr_line, 2)

		kr_words, kr_num = find_number(kr_line)
		en_words, en_num = find_number(en_line)

		kr_words, en_words = NumtoSym_same(kr_words, kr_num, en_words, en_num)
		kr_words, en_words = check1to20(kr_words, kr_num, en_words)

		kr_words = NumtoSym_rest(kr_words, kr_num)
		en_words = NumtoSym_rest(en_words, en_num)

		kr_line = " ".join(kr_words)
		en_line = " ".join(en_words)
		#print(kr_line)
		kr_out.write(kr_line + "\n")
		en_out.write(en_line + "\n")

		loop_count += 1
		if loop_count % 100000 == 0:
			print (str(loop_count) + ' lines have been done')

	kr_in.close()
	kr_out.close()
	en_in.close()
	en_out.close()

def number_unit_kr(line, phase):
        dict_kr = dict(zip(kr_word, kr_num))
        pattern = ''
        if phase == 1:
                for w in kr_word[0:3]:
                        pattern += '\d+[.]*\d*\s?' + w + '\s*|\d+[,]*\d*\s?' + w + '\s*|'
        else:
                for w in kr_word[3:5]:
                        pattern += '\d+[.]*\d*\s?' + w + '\s*|\d+[,]*\d*\s?' + w + '\s*|'
	pattern = pattern[:-1]
	result = re.findall(pattern, line)

	for i in range(len(result)):
		old = result[i]
		new  = result[i].replace(",", "")
		result[i] = new
		line = line.replace(old, new)

		n1 = n2 = 0
                result_words = result[i].split()
                if len(result_words) == 1:
                        result_word = result_words[0]
                        for j in range(1, len(result_word)):
                                if result_word[j] in kr_word:
                                	n1 = result_word[:j]
                                        n2 = dict_kr[result_word[j]]
                                        if j < len(result_word) - 1:
                                                n2 *= int(dict_kr[resultword[j+1]])
					break
                elif len(result_words) == 2:
                        n1 = result_words[0]
			n2 = dict_kr[result_words[1]]
                mul = float(n1) * int(n2)
                if mul - int(mul) == 0:
                        mul = int(mul)
                line = line.replace(result[i], str(mul) + '&&')

	pattern = '\d+[&&]+\s?\d*[&&]*|\d+[&&]+\s?\d+[&&]+\s?\d*[&&]*|\d+[&&]+\s?\d+[&&]+\s?\d+[&&]+\s?\d*[&&]*'
	result = re.findall(pattern, line)
	for i in range(len(result)):
		sum_ = 0
		for num in result[i].split('&&'):
			if isNumber(num):
				sum_ += int(num)
		line = line.replace(result[i], str(sum_))

        return line

def number_unit_en(line):
	dict_en = dict(zip(en_word, en_num))
	pattern = ''
	for w in en_word:
       		pattern += '\s+\d+[.]*\d*[-]*\s?' + w + '[s]*|'
	pattern = pattern[:-1]
	result = re.findall(pattern, line)

        for i in range(len(result)):
		old = result[i]
		new  = result[i].replace("-", " ")
		result[i] = new
		line = line.replace(old, new)

		n1 = n2 = 0
                result_words = result[i].split()

                if len(result_words) == 1:
                        result_word = result_words[0]
                        for j in range(1, len(result_word)):
                                if not isNumber(result_word[:j]):
                                        n1 = result_word[:j-1]
                                        n2 = dict_en[result_word[j-1:]]
					break
		elif len(result_words) == 2:
                        n1 = result_words[0]
			n2 = dict_en[result_words[1]]
		mul = float(n1) * int(n2)
                if mul - int(mul) == 0:
                        mul = int(mul)
                if result[i][0] == ' ':
                        mul = ' ' + str(mul)
                line = line.replace(result[i], str(mul))
        return line

def separate_number(line):
	newline = ''
	preflag = False #if pre-index is number, True
	for i in range(len(line)):
		if line[i] == ".":
			newline += line[i]
			continue
		if line[i].isspace(): #space
			if preflag is True and i + 1 < len(line) and line[i + 1] == '0':
				continue
			newline += line[i]
			preflag = False
		elif isNumber(line[i]): #number
			if preflag is True:
				newline += line[i]
			elif preflag is False and i != 0 and not newline[-1].isspace():
				newline = newline + "&& " + line[i]
			else:
				newline += line[i]
			preflag = True
		else: # char
			if preflag is True:
				if line[i] == '월':
					newline += line[i]
					preflag = False
					continue
				elif line[i] == ',':
					if i + 3 < len(line) and isNumber(line[i + 1:i + 4]):
						continue
					else:
						newline += " " + line[i]
						if i + 1 < len(line) and not line[i + 1].isspace():
							newline += " "
						continue
                                newline = newline + " ^^" + line[i]
			elif preflag is False:
				newline += line[i]
			preflag = False

	return newline

def find_number(line):
	line = separate_number(line)
	words = line.split(" ")
	words[-1] = words[-1].replace("\n", "")

	index = []

	for i in range(len(words)):
		if isNumber(words[i]):
			index.append(i)

	return words, index

def NumtoSym_same(src_split, src_num_index, trg_split, trg_num_index):
	global COUNT_NUM
	for i in src_num_index:
		for j in trg_num_index:
			if isNumber(src_split[i]) and src_split[i] == trg_split[j]:
				replace2(src_split, src_num_index, src_split[i], symbol[COUNT_NUM])
				replace2(trg_split, trg_num_index, trg_split[j], symbol[COUNT_NUM])
				if COUNT_NUM < MAX_NUM:
					COUNT_NUM += 1
	return src_split, trg_split

def check1to20(src_split, src_num_index, trg_split):
	global COUNT_NUM
	checking_list = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
	for i in src_num_index:
		if isInt(src_split[i]) and 1 <= int(src_split[i]) and int(src_split[i]) <= 20 :
			if checking_list[int(src_split[i])] in trg_split:
                                replace2(trg_split, [n for n in range(len(trg_split))], checking_list[int(src_split[i])], symbol[COUNT_NUM])
				replace2(src_split, src_num_index, src_split[i], symbol[COUNT_NUM])
				if COUNT_NUM < MAX_NUM:
					COUNT_NUM += 1
	return src_split, trg_split

def NumtoSym_rest(split, num_index):
	global COUNT_NUM
	for i in num_index:
		if isNumber(split[i]):
			replace2(split, num_index, split[i], symbol[COUNT_NUM])
			if COUNT_NUM < MAX_NUM:
				COUNT_NUM += 1
	return split

def isNumber(s):
        try:
                float(s)
                return True
        except ValueError:
                pass

def isInt(s):
	try:
		if isNumber(s):
			if (float(s)-int(s))==0:
				return True
	except ValueError:
		pass

def replace2(split, index, old, new):
	for i in index:
		if split[i] == old:
			split[i] = new

if __name__ == "__main__":
        symbolize_kr2en()
