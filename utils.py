# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np

def time_format(s):
    h = math.floor(s / 3600)
    m = math.floor((s-3600*h) / 60)
    s = s - h*3600 - m*60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (time_format(s))

def ids2words(dict_map_inv, raw_data, sep=' ', eos_id=0, unk_sym='<unk>'):
    str_text = ''
    for vv in raw_data:
        if vv == eos_id:
            break
        if vv in dict_map_inv:
            str_text = str_text + sep + dict_map_inv[vv]
        else:
            str_text = str_text + sep + unk_sym
    return str_text.strip()

def unbpe(sentence):
    sentence = sentence.replace('<s>', '').strip()
    sentence = sentence.replace('</s>', '').strip()
    sentence = sentence.replace('@@ ', '')
    sentence = sentence.replace('@@', '')
    return sentence

def equizip(*iterables):
    iterators = [iter(x) for x in iterables]
    while True:
        try:
            first_value = iterators[0].__next__()
            try:
                other_values = [x.__next__() for x in iterators[1:]]
            except StopIteration:
                raise IterableLengthMismatch
            else:
                values = [first_value] + other_values
                yield tuple(values)
        except StopIteration:
            for iterator in iterators[1:]:
                try:
                    extra_value = iterator.__next__()
                except StopIteration:
                    pass # this is what we expect
                else:
                    raise IterableLengthMismatch
            raise StopIteration
