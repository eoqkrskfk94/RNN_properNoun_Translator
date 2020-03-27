import six; from six.moves import cPickle as pkl
import gzip
import numpy as np
from utils import equizip

BOS_token = 0
EOS_token = 1
UNK_token = 2
# in the dict file, <s>/</s>=0, <unk>=1


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextPairIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target, src_dict, trg_dict,
                 unk_id=2, batch_size=128, maxlen=100,
                 ahead=1, resume_num=0):
        self.unk_id = unk_id
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        with open(src_dict, 'rb') as f:
            self.src_dict = pkl.load(f, encoding="utf-8")
        with open(trg_dict, 'rb') as f:
            self.trg_dict = pkl.load(f, encoding="utf-8")

        self.src_dict2 = dict()
        for kk, vv in self.src_dict.items():
            self.src_dict2[kk] = vv+1
        self.src_dict2['<s>'] = BOS_token


        self.trg_dict2 = dict()
        for kk, vv in self.trg_dict.items():
            self.trg_dict2[kk] = vv+1
        self.trg_dict2['<s>'] = BOS_token

        import operator
        sorted_vocab_src = sorted(self.trg_dict2.items(), key=operator.itemgetter(1))
        print(sorted_vocab_src)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.end_of_data = False

        self.x_buf =[]
        self.y_buf =[]
        self.buf_remain = 0
        self.cur_line_num=0
        self.ahead=ahead
        self.iters = 0

        if resume_num > 0:
            self.cur_line_num=resume_num
            for i in range(resume_num):
                ss = self.source.readline()
                tt = self.target.readline()

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        self.cur_line_num=0
        self.iters = self.iters + 1

    def __next__(self):
        if self.buf_remain == 0:
            self.x_buf = []
            self.y_buf = []
            i = 0
            while True:
                ss = self.source.readline()
                tt = self.target.readline()
                if ss == "" or tt == "":
                    self.reset()
                    ss = self.source.readline()
                    tt = self.target.readline()
                    #raise StopIteration # validation

                ss = ss.strip().split()
                tt = tt.strip().split()

                ss = [self.src_dict2.get(key, self.unk_id) for key in ss] # 0 BOS, 1 EOS, 2 UNK
                tt = [self.trg_dict2.get(key, self.unk_id) for key in tt]

                self.cur_line_num = self.cur_line_num + 1

                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.x_buf.append(ss)
                self.y_buf.append(tt)

                i = i + 1
                if i >= self.batch_size*self.ahead:
                    break

            self.buf_remain = self.ahead
            #self.buf_remain = (i-1)/self.batch_size + 1

            len_xy = [(len(x), len(y), x, y) for x, y in equizip(self.x_buf, self.y_buf)]
            sorted_len_xy = sorted(len_xy, key=lambda xy: (xy[0], xy[1]))
            self.x_buf = [xy[2] for xy in sorted_len_xy]
            self.y_buf = [xy[3] for xy in sorted_len_xy]

        # with self.buf_remain as index
        br = self.ahead-self.buf_remain
        bs = self.batch_size

        source = self.x_buf[br*bs:(br+1)*bs]
        target = self.y_buf[br*bs:(br+1)*bs]

        self.buf_remain = self.buf_remain - 1

        x_data, x_mask, y_data, y_mask = self.prepare_text_pair(source, target)
        self.iters = self.iters + 1
        return x_data, x_mask, y_data, y_mask, self.cur_line_num, self.iters

    # batch preparation
    def prepare_text_pair(self, seqs_x, seqs_y):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        lengths_y = [len(s) for s in seqs_y]

        n_samples = len(seqs_x)
        maxlen_x = np.max(lengths_x) + 2 # for BOS and EOS
        maxlen_y = np.max(lengths_y) + 1 # for EOS

        x_data = np.ones((maxlen_x, n_samples)).astype('int64') # BOS_token = 0
        y_data = np.ones((maxlen_y, n_samples)).astype('int64') # BOS_token = 0
        x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
        y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
        for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
            x_data[1:lengths_x[idx]+1, idx] = s_x
            x_data[0, idx] = BOS_token
            x_mask[:lengths_x[idx]+2, idx] = 1. # extra +2 for BOS/EOS)
            y_data[:lengths_y[idx], idx] = s_y
            y_mask[:lengths_y[idx]+1, idx] = 1. # extra +1 for EOS (zero)

        return x_data, x_mask, y_data, y_mask


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, src_dict,
                 unk_id=2, batch_size=128, maxlen=100,
                 ahead=1, resume_num=0):
        self.source_name = source
        self.unk_id = unk_id

        self.source = fopen(source, 'r')
        with open(src_dict, 'rb') as f:
            self.src_dict = pkl.load(f, encoding="utf-8")

        self.src_dict2 = dict()
        for kk, vv in self.src_dict.items():
            self.src_dict2[kk] = vv+1
        self.src_dict2['<s>'] = BOS_token

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.end_of_data = False

        self.x_buf =[]
        self.buf_remain = 0
        self.cur_line_num=0
        self.ahead=ahead
        self.iters = 0

        if resume_num > 0:
            self.cur_line_num=resume_num
            for i in range(resume_num):
                ss = self.source.readline()

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.cur_line_num=0
        self.iters = self.iters + 1

    def __next__(self):

        if self.buf_remain == 0:
            self.x_buf = []
            i = 0
            while True:
                ss = self.source.readline()
                if ss == "":
                    self.reset()
                    if self.ahead == 1:
                        raise StopIteration # validation
                    ss = self.source.readline()

                ss = ss.strip().split()
                ss = [self.src_dict2.get(key, self.unk_id) for key in ss] # 0 BOS, 1 EOS, 2 UNK
                self.cur_line_num = self.cur_line_num + 1

                if len(ss) > self.maxlen:
                    continue

                self.x_buf.append(ss)

                if len(self.x_buf) >= self.batch_size*self.ahead:
                    break

            self.buf_remain = self.ahead

            len_xs = [(len(x), x) for x in self.x_buf]
            sorted_len_xs = sorted(len_xs, key=lambda xs: xs[0])
            self.x_buf = [xs[1] for xs in sorted_len_xs]

        # with self.buf_remain as index
        br = self.ahead-self.buf_remain
        bs = self.batch_size
        #print br, bs, len(self.x_buf)

        source = self.x_buf[br*bs:(br+1)*bs]

        self.buf_remain = self.buf_remain - 1

        x_data, x_mask = self.prepare_text(source)
        self.iters = self.iters + 1
        return x_data, x_mask, self.cur_line_num, self.iters

    # batch preparation, returns padded batch and mask
    def prepare_text(self, seqs_x):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x]
        n_samples = len(seqs_x)

        maxlen_x = np.max(lengths_x) + 2 # +2 for BOS and EOS

        x_data = np.ones((maxlen_x, n_samples)).astype('int64')
        x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
        for idx, s_x in enumerate(seqs_x):
            x_data[1:lengths_x[idx]+1, idx] = s_x
            x_data[0, idx] = BOS_token
            x_mask[:lengths_x[idx]+2, idx] = 1. # extra +2 for BOS and EOS

        return x_data, x_mask


if __name__ == "__main__":
    base_dir = '/home/nmt19/data_05/bleu05/test/'
    src_file = base_dir + 'train.kr'
    trg_file = base_dir + 'train.kr'
    src_dict = base_dir + 'vocab.en.pkl'
    trg_dict = base_dir + 'vocab.kr.pkl'

    train_iter = TextPairIterator(src_file, trg_file, src_dict, trg_dict,
                         batch_size=3, maxlen=50,
                         ahead=5, resume_num=0)
    valid_iter = TextIterator(src_file,src_dict)

    idx = 0
    for x, xm, y, ym, tmp1, tmp2 in train_iter:

        print (x.shape)
        idx = idx + 1
        #if idx >= 10:
        break


    for x, xm, y, ym in valid_iter:

        print (x.shape)
        idx = idx + 1
        #if idx >= 10:
        break
