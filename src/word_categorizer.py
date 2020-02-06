import os
import numpy as np
import random as rand
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm

# My modules
from src.BertLM import BertLM


class WordCategorizer:
    def __init__(self, vocab_filename, pretrained_model='bert-base-uncased', device_number='cuda:2', use_cuda=False):

        self.device_number = device_number
        self.use_cuda = use_cuda

        print("Loading BERT model...")
        self.Bert_Model = BertLM(pretrained_model=pretrained_model, device_number=device_number, use_cuda=use_cuda)

        self.vocab = []
        print("Loading vocabulary...")
        self.load_vocabulary(vocab_filename)
        self.matrix = []  # Stores sent probability for each word-sentence pair (rows are words)

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
        print("Evaluating word-sentence probabilities")
        num_sents = 0
        with open(sents_filename, 'r') as fs:
            for sent in fs:
                tokenized_sent = self.Bert_Model.tokenize_sent(sent)
                masks_pos = rand.sample(range(1, len(tokenized_sent) - 1), num_masks)  # Don't mask boundary tokens
                for mask_pos in masks_pos:
                    # Calculate sentence probability for each word in current masked position
                    print(f"Evaluating sentence {tokenized_sent} with mask in pos {mask_pos}")
                    sent_row = [self.process_sentence(tokenized_sent[:], word, mask_pos, verbose=verbose) for word in
                                self.vocab]
                    self.matrix.append(sent_row)
                    num_sents += 1

        self.matrix = np.array(self.matrix).T  # Make rows be word-senses

    def process_sentence(self, tokenized_sent, word, mask_pos, verbose=False):
        """
        Replaces word in mask_pos for input word, and evaluates the sentence probability
        :param tokenized_sent: Input sentence
        :param word: Input word
        :param mask_pos: Position to replace input word
        :param verbose:
        :return:
        """
        tokenized_sent[mask_pos] = word
        curr_prob = self.Bert_Model.get_sentence_prob(tokenized_sent, verbose=verbose)

        return curr_prob

    def cluster_words(self, method='KMeans', **kwargs):
        if method != 'KMeans':
            print("Method not implemented... using KMeans instead")

        print("Clustering word-sense vectors")
        k = kwargs.get('k', 2)  # 2 is default value, if no kwargs were passed
        estimator = KMeans(n_clusters=k, n_jobs=4)
        estimator.fit(self.matrix)  # Transpose matrix to cluster words, not sentences
        return estimator.labels_

    def write_clusters(self, save_to, labels):
        """
        Write clustering results to file
        :param save_to:        Directory to save disambiguated senses
        :param labels:         Cluster labels
        """
        num_clusters = max(labels) + 1
        print(f"Writing {num_clusters} clusters to file")

        # Write word categories to file
        save_to = save_to + "_KMeans_k" + str(num_clusters)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        with open(save_to + '/categories.txt', "w") as fo:
            for i in range(-1, num_clusters):  # Also write unclustered words
                # sense_members = [self.vocab_map[word][j] for j, k in enumerate(labels) if k == i]
                cluster_members = [self.vocab[j] for j, k in enumerate(labels) if k == i]
                fo.write(f"Cluster #{i}")
                if len(cluster_members) > 0:  # Handle empty clusters
                    fo.write(": \n[")
                    np.savetxt(fo, cluster_members, fmt="%s", newline=", ")
                    fo.write(']\n')
                else:
                    fo.write(" is empty\n\n")


if __name__ == '__main__':
    for _ in tqdm(range(1)):
        wc = WordCategorizer('../vocabularies/test.vocab', pretrained_model='bert-base-uncased')
        wc.populate_matrix('../sentences/9sentences.txt', num_masks=3, verbose=False)
        cluster_labels = wc.cluster_words(method='KMeans', k=5)
        wc.write_clusters('test', cluster_labels)
        print(wc.matrix)

