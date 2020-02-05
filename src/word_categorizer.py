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
        """
        Calculates probability matrix for the sentence-word pairs
        Currently can only handle one mask per sentence. We can repeat sentences in the sents_file as
        a workaround to this.
        :param sents_filename:  File with input sentences
        :return: None
        """
        with open(sents_filename, 'r') as fs:
            self.matrix = [self.process_sentence(sent, word) for sent in fs for word in self.vocab]
            self.matrix = np.reshape(self.matrix, (round(len(self.matrix)/len(self.vocab)), len(self.vocab)))

    def process_sentence(self, sent, word):
        """
        Replaces word in mask_pos for input word, and evaluates the sentence probability
        :param sent: Input sentence
        :param word: Input word
        :param mask_pos: Position to replace input word
        :return:
        """
        tokenized_sent = self.Bert_Model.tokenizer.tokenize(sent)
        mask_pos = rand.randint(0, len(tokenized_sent) - 1)
        tokenized_sent[mask_pos] = word
        curr_prob = self.Bert_Model.get_sentence_prob(tokenized_sent)

        return curr_prob


if __name__ == '__main__':
    wc = WordCategorizer('../vocabularies/test.vocab')
    wc.populate_matrix('../sentences/9sentences.txt')
    print(wc.matrix)

