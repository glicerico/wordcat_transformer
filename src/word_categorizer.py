import numpy as np
from src import BertLM


class WordCategorizer:
    def __init__(self, pretrained_model, vocab_filename, device_number='cuda:2', use_cuda=False):

        self.device_number = device_number
        self.use_cuda = use_cuda

        self.Bert_Model = BertLM(pretrained_model, device_number, use_cuda)

        self.matrix = None
        self.vocab = None
        self.sents = None
        self.load_vocabulary(vocab_filename)

    def load_vocabulary(self, vocab_filename):
        """
        Reads vocabulary file. File format must be one word per line, no comments accepted.
        :param vocab_filename:  Path to vocab file
        :return:                None
        """
        with open(vocab_filename, 'r') as fv:
            self.vocab = fv.readlines()

    def load_sentences(self, sents_filename):
        """
        Reads sentences file. File format must be one sentence per line, no comments accepted.
        :param sents_filename:  Path to sentence file
        :return:                None
        """
        with open(sents_filename, 'r') as fs:
            self.sents = fs.readlines()

