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
        self.matrix = []

    def load_vocabulary(self, vocab_filename):
        """
        Reads vocabulary file. File format must be one word per line, no comments accepted.
        :param vocab_filename:  Path to vocab file
        :return:                None
        """
        with open(vocab_filename, 'r') as fv:
            self.vocab = fv.read().splitlines()

    def populate_matrix(self, sents_filename, num_masks=1, verbose=False):
        """
        Calculates probability matrix for the sentence-word pairs
        Currently can only handle one mask per sentence. We can repeat sentences in the sents_file as
        a workaround to this.
        :param sents_filename:  File with input sentences
        :param num_masks:       Repetitions for each sentence, with different masks
        :return: None
        """
        num_sents = 0
        with open(sents_filename, 'r') as fs:
            for sent in fs:
                tokenized_sent = self.Bert_Model.tokenize_sent(sent)
                masks_pos = rand.sample(range(1, len(tokenized_sent) + 1), num_masks)  # Don't mask boundary tokens
                for mask_pos in masks_pos:
                    sent_row = [self.process_sentence(tokenized_sent, word, mask_pos, verbose=verbose) for word in self.vocab]
                    self.matrix.extend(sent_row)
                    num_sents += 1

            self.matrix = np.reshape(self.matrix, (round(len(self.matrix)/len(self.vocab)), len(self.vocab)))

    def process_sentence(self, tokenized_sent, word, mask_pos, verbose=False):
        """
        Replaces word in mask_pos for input word, and evaluates the sentence probability
        :param tokenized_sent: Input sentence
        :param word: Input word
        :param mask_pos: Position to replace input word
        :return:
        """
        tokenized_sent[mask_pos] = word
        curr_prob = self.Bert_Model.get_sentence_prob(tokenized_sent, verbose=verbose)

        return curr_prob


if __name__ == '__main__':
    wc = WordCategorizer('../vocabularies/test.vocab')
    wc.populate_matrix('../sentences/9sentences.txt', verbose=True)
    print(wc.matrix)

