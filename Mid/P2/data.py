import unicodedata
import re
import random

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 50


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.num_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII
    # refer to https://stackoverflow.com/a/518232/2809427
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize(sen):
    sen = unicodeToAscii(sen.lower().strip())
    sen = re.sub(r"([.!?])", r" \1", sen)
    sen = re.sub(r"[^a-zA-z.!?]+", r" ", sen)

    return sen


def read_from_text(filepath, q, a):
    print('Reading....')
    lines = open(filepath, encoding='utf-8').read().strip().split('\n')

    pairs = [[normalize(s) for s in l.split('\t')] for l in lines]

    questions=Lang(q)
    answers = Lang(a)

    return  questions,answers,pairs


def filter_pair(p):
    return len(p[0])<MAX_LENGTH and len(p[1]) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def perpare_dataset(filepath, q,a):
    input,output,pairs=read_from_text(filepath, q,a)
    print('Read {} sentence pairs'.format(len(pairs)))

    pairs=filter_pairs(pairs)

    for pair in pairs:
        input.add_sentence(pair[0])
        output.add_sentence(pair[1])
    print('Counted words: {}- {}\t{} - {}'.format(input.name, input.num_words,
                                                  output.name, output.num_words))

    return input, output, pairs