import numpy as np
import random as rand

# My modules
from src import BertLM


class WordCategorizer:
    def __init__(self, vocab_filename, pretrained_model='bert-base-uncased', device_number='cuda:2', use_cuda=False):

        self.device_number = device_number
        self.use_cuda = use_cuda

        self.Bert_Model = BertLM(pretrained_model=pretrained_model, device_number=device_number, use_cuda=use_cuda)

        self.vocab = []
        self.load_vocabulary(vocab_filename)
        self.matrix = [[] for _ in range(len(self.vocab))]  # Initialize matrix as list of lists for each word

    def load_vocabulary(self, vocab_filename):
        """
        Reads vocabulary file. File format must be one word per line, no comments accepted.
        :param vocab_filename:  Path to vocab file
        :return:                None
        """
        with open(vocab_filename, 'r') as fv:
            self.vocab = fv.readlines()

    # def load_sentences(self, sents_filename):
    #     """
    #     Reads sentences file. File format must be one sentence per line, no comments accepted.
    #     :param sents_filename:  Path to sentence file
    #     :return:                None
    #     """
    #     with open(sents_filename, 'r') as fs:
    #         self.sents = fs.readlines()

    def populate_matrix(self, sents_filename):
        with open(sents_filename, 'r') as fs:
            for sent in fs:
                process_sentence(sent)

    def process_sentence(self, sent, num_masks=2):
        # rand.sample()  # Sample which positions will be used for masking
        mask_locations = [rand.randint(1, len(sent) - 2)]  # Avoid masking special boundary tokens
        for pos in mask_locations:
            for word_id, word in enumerate(self.vocab):
                sent[pos] = word
                curr_prob = self.Bert_Model.get_sentence_prob(sent)
                self.matrix[word_id].append(curr_prob)

