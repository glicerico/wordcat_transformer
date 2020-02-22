import argparse
import itertools
import os
import pickle
import sys

import numpy as np
import random as rand
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from tqdm import tqdm
from scipy import sparse

# My modules
sys.path.insert(0, os.path.abspath('../src'))
from BertModel import BertLM


class WordCategorizer:
    def __init__(self, pretrained_model='bert-base-uncased', use_cuda=False, device_number='cuda:2'):

        self.device_number = device_number
        self.use_cuda = use_cuda

        print("Loading BERT model...")
        self.Bert_Model = BertLM(pretrained_model=pretrained_model, device_number=device_number, use_cuda=use_cuda)

        self.vocab = []
        self.matrix = []  # Stores sent probability for each word-sentence pair (rows are words)
        self.gold = []

    def load_vocabulary(self, vocab_filename):
        """
        Reads vocabulary file. File format must be one word per line, no comments accepted.
        :param vocab_filename:  Path to vocab file
        :return:                None
        """
        with open(vocab_filename, 'r') as fv:
            lines = fv.read().splitlines()
            for li in lines:
                split_line = li.split()
                self.vocab.append(split_line[0])  # Ignores POS labels if present
                self.gold.append(split_line[-1])  # Saves gold standard labels if present

    def load_matrix(self, vocab_filename, sentences_filename, pickle_filename, num_masks=1, verbose=False, sparse_thres=-8):
        """
        If pickle file is present, load data; else, calculate it.
        :param vocab_filename:      File with vocabulary to categorize
        :param sentences_filename:  File with sentence to use as features for word categorization
        :param pickle_filename:     File to store data
        :param num_masks:           If sentence prob is under this value, assign 0
        :param verbose:
        :param sparse_thres:        Cutoff to eliminate very low values and make sparse matrix
        :return:
        """
        try:
            with open(pickle_filename, 'rb') as h:
                _data = pickle.load(h)
                self.vocab = _data[0]
                self.gold = _data[1]
                self.matrix = _data[2]
                self.Bert_Model = _data[3]

                print("MATRIX FOUND!")
                # print(self.matrix)

        except:
            print("MATRIX File Not Found!! \n")
            print("Performing matrix calculation...")

            self.load_vocabulary(vocab_filename)
            self.populate_matrix(sentences_filename, num_repl=num_masks, verbose=verbose, sparse_thres=sparse_thres)

            with open(pickle_filename, 'wb') as h:
                _data = (self.vocab, self.gold, self.matrix, self.Bert_Model)
                pickle.dump(_data, h)

            print("Data stored in " + pickle_filename)

    def populate_matrix(self, sents_filename, num_repl=1, sparse_thres=-4, verbose=False):
        """
        Calculates probability matrix for the sentence-word pairs
        :param sents_filename:  File with input sentences
        :param num_repl:       Repetitions for each sentence, with different replacements
        :param sparse_thres:    If sentence prob is under this value, assign 0
        :param verbose:
        :return: None
        """
        print("Evaluating word-sentence probabilities")
        num_sents = 0
        with open(sents_filename, 'r') as fs:
            for sent in fs:
                tokenized_sent = self.Bert_Model.tokenize_sent(sent)
                # word_starts stores indexes where words begin (ignoring sub-words)
                word_starts = [index for index, token in enumerate(tokenized_sent) if not token.startswith("##")]
                num_words = len(word_starts)
                # Don't mask boundary tokens, sample same sentence with various masks (less than num_words)
                replacements_pos = rand.sample(range(1, num_words - 1), min(num_words - 2, num_repl))
                for repl_pos in replacements_pos:
                    # Calculate sentence probability for each word in current replacement position
                    print(f"Evaluating sentence {tokenized_sent} replacing word "
                          f"{tokenized_sent[word_starts[repl_pos]:word_starts[repl_pos + 1]]}")
                    sent_row = np.array(
                        [self.process_sentence(tokenized_sent, word, repl_pos, word_starts, verbose=verbose)
                         for word in self.vocab])
                    sent_row = sent_row * (sent_row > sparse_thres)  # Cut low probability values
                    self.matrix.append(sent_row)
                    num_sents += 1

        self.matrix = np.array(self.matrix).astype(np.float32)  # Reduce matrix precision, make rows be word-senses
        self.matrix = sparse.csr_matrix(self.matrix.T)  # Convert to sparse matrix

    def process_sentence(self, tokenized_sent, word, repl_pos, word_init, verbose=False):
        """
        Replaces word in repl_pos, incl. all subwords it may contain, for input word with subwords;
        then evaluates the sentence probability.
        :param tokenized_sent: Input sentence
        :param word: Input word
        :param repl_pos: Word position to replace input word
        :param word_init: List with initial positions of each word in tokenized_sent
        :param verbose:
        :return: curr_prob: Probability of current sentence
        """
        # Following substitution handles removing sub-words, as well as inserting them
        word_tokens = self.Bert_Model.tokenizer.tokenize(word)
        replaced_sent = tokenized_sent[:word_init[repl_pos]] + word_tokens + tokenized_sent[word_init[repl_pos + 1]:]
        curr_prob = self.Bert_Model.get_sentence_prob_directional(replaced_sent, verbose=verbose)

        return curr_prob

    def cluster_words(self, method='KMeans', **kwargs):
        if method == 'KMeans':
            k = kwargs.get('k', 2)  # 2 is default value, if no kwargs were passed
            estimator = KMeans(n_clusters=int(k), n_jobs=4, n_init=10)
            estimator.fit(self.matrix)  # Cluster matrix
        elif method == 'DBSCAN':
            eps = kwargs.get('k', 0.2)
            min_samples = kwargs.get('min_samples', 3)
            estimator = DBSCAN(min_samples=min_samples, eps=eps, n_jobs=4, metric='cosine')
            estimator.fit(self.matrix)  # Cluster matrix
        elif method == 'OPTICS':
            estimator = OPTICS(min_samples=3, metric='cosine', xi=0.02, n_jobs=4)
            estimator.fit(self.matrix.toarray())  # Cluster matrix
        else:
            print("Clustering method not implemented...")
            exit(1)


        return estimator.labels_

    def write_clusters(self, method, save_to, labels, clust_param):
        """
        Write clustering results to file
        :param save_to:        Directory to save disambiguated senses
        :param labels:         Cluster labels
        :param method:         Clustering method used
        """
        num_clusters = max(labels) + 1
        print(f"Writing {num_clusters} clusters to file")

        # Write word categories to file
        append = "/" + method + "_" + str(clust_param)
        # if not os.path.exists(save_to):
        #     os.makedirs(save_to)
        with open(save_to + append + '.wordcat', "w") as fo:
            for i in range(-1, num_clusters):  # Also write unclustered words
                cluster_members = [self.vocab[j] for j, k in enumerate(labels) if k == i]
                fo.write(f"Cluster #{i}")
                if len(cluster_members) > 0:  # Handle empty clusters
                    fo.write(": \n[")
                    np.savetxt(fo, cluster_members, fmt="%s", newline=", ")
                    fo.write(']\n')
                else:
                    fo.write(" is empty\n\n")

    def eval_clusters(self, logfile, pred):
        if self.gold == self.vocab:
            print("Gold labels not present in vocabulary file. Can't evaluate!\n")
            return
        else:
            true_pairs = self.get_pairs(np.array(self.gold))
            pred_pairs = self.get_pairs(pred)
            int_size = len(set(true_pairs).intersection(pred_pairs))
            p = int_size / float(len(pred_pairs))
            r = int_size / float(len(true_pairs))
            f_score = 2 * p * r / float(p + r)
            print(f"Fscore: {f_score}\n")
            fl.write(f"FScore: {f_score}\n")

    @staticmethod
    def get_pairs(labels):
        result = []
        for label in np.unique(labels):
            ulabels = np.where(labels == label)[0]
            for p in itertools.combinations(ulabels, 2):
                result.append(p)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSD using Transformers')
    parser.add_argument('--pretrained', type=str, default='bert-base-uncased', help='Pretrained model to use')
    parser.add_argument('--use_cuda', action='store_true', help='Use GPU?')
    parser.add_argument('--device', type=str, default='cuda:2', help='GPU Device to Use?')
    parser.add_argument('--sentences', type=str, required=True, help='Sentence Corpus')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary Corpus')
    parser.add_argument('--masks', type=int, default=1, help='Min freq of word to be disambiguated')
    parser.add_argument('--sparse_thres', type=int, default=-8, help='Low log(prob) cut')
    parser.add_argument('--clusterer', type=str, default='KMeans', help='Clustering method to use')
    parser.add_argument('--start_k', type=float, default=10, help='Initial value of clustering param')
    parser.add_argument('--end_k', type=float, default=10, help='Final value of clustering param')
    parser.add_argument('--steps_k', type=int, default=5, help='Step for clustering param exploration')
    parser.add_argument('--save_to', type=str, default='test', help='Directory to save disambiguated words')
    parser.add_argument('--verbose', action='store_true', help='Print processing details')
    parser.add_argument('--pickle_file', type=str, default='test.pickle', help='Pickle file with embeddings/Save '
                                                                               'embeddings to file')
    args = parser.parse_args()

    wc = WordCategorizer(pretrained_model=args.pretrained, use_cuda=args.use_cuda, device_number=args.device)
    # Heavy part of the process: calculate sentence probabilities for each vocab word
    for _ in tqdm(range(1)):  # Time the process
        wc.load_matrix(args.vocab, args.sentences, args.pickle_file, num_masks=args.masks, verbose=args.verbose,
                       sparse_thres=args.sparse_thres)

    print("Start clustering...")
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    with open(args.save_to + '/results.log', 'w') as fl:
        for curr_k in tqdm(np.linspace(args.start_k, args.end_k, args.steps_k)):
            print(f"Clustering with k={curr_k}")
            cluster_labels = wc.cluster_words(method=args.clusterer, k=curr_k)
            wc.write_clusters(args.clusterer, args.save_to, cluster_labels, curr_k)
            print(f"\nEvaluation for k={curr_k}")
            fl.write(f"Evaluation for k={curr_k}\n")
            wc.eval_clusters(fl, cluster_labels)
