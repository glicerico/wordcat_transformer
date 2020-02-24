import argparse
import itertools
import os
import pickle
import sys

import numpy as np
import random as rand
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
from scipy import sparse

# My modules
sys.path.insert(0, os.path.abspath('../src'))
from BertModel import BertLM
from word_senser import WordSenseModel


class WordCategorizer:
    def __init__(self, pretrained_model='bert-base-uncased', use_cuda=False, device_number='cuda:2'):

        self.device_number = device_number
        self.use_cuda = use_cuda

        print("Loading BERT model...")
        self.Bert_Model = BertLM(pretrained_model=pretrained_model, device_number=device_number, use_cuda=use_cuda)
        self.Bert_WSD = WordSenseModel(pretrained_model, device_number, use_cuda)

        self.vocab = []
        self.disamb_vocab = []  # List where word is repeated if ambiguous (for writing clusters)
        self.matrix = []  # Stores sent probability for each word-sentence pair (rows are words)
        self.gold = []
        self.senses = {}  # Stores words that have more than one sense, according to WSD pickle file

    def load_senses(self, pickle_senses):
        """
        Load ambiguous word senses, as stored by word_senser.py
        :param pickle_senses:
        :return:
        """
        with open(pickle_senses, 'rb') as fs:
            self.senses = pickle.load(fs)

    def load_vocabulary(self, vocab_txt=None, vocab_pickle=None):
        """
        Reads vocabulary file from txt or pickle file.
        If text file, format must be one word per line, no comments accepted.
        If pickle file, it must contain only a list of all words in vocabulary.
        :param vocab_txt:       Path to txt vocab file
        :param vocab_pickle:    Path to pickle vocab file
        :return:                None
        """
        if vocab_txt:
            with open(vocab_txt, 'r') as fv:
                lines = fv.read().splitlines()
                for li in lines:
                    split_line = li.split()
                    self.vocab.append(split_line[0])  # Ignores POS labels if present
                    self.gold.append(split_line[-1])  # Saves gold standard labels if present
        elif vocab_pickle:
            with open(vocab_pickle, 'rb') as fv:
                self.vocab = pickle.load(fv)
                self.gold = None
        else:
            print("Need vocabulary file in text or pickle format")
            exit()

    def load_matrix(self, sentences_filename, pickle_emb, num_masks=1, verbose=False, sparse_thres=-8):
        """
        If pickle file is present, load data; else, calculate it.
        :param sentences_filename:  File with sentence to use as features for word categorization
        :param pickle_emb:          File to store/load embeddings
        :param num_masks:           If sentence prob is under this value, assign 0
        :param verbose:
        :param sparse_thres:        Cutoff to eliminate very low values and make sparse matrix
        :return:
        """
        try:
            with open(pickle_emb, 'rb') as h:
                _data = pickle.load(h)
                self.disamb_vocab = _data[0]
                self.gold = _data[1]
                self.matrix = _data[2]
                self.Bert_Model = _data[3]

                print("MATRIX FOUND!")
                # print(self.matrix)

        except:
            print("MATRIX File Not Found!! \n")
            print("Performing matrix calculation...")

            self.populate_matrix(sentences_filename, num_repl=num_masks, verbose=verbose, sparse_thres=sparse_thres)

            with open(pickle_emb, 'wb') as h:
                _data = (self.disamb_vocab, self.gold, self.matrix, self.Bert_Model)
                pickle.dump(_data, h)

            print("Data stored in " + pickle_emb)

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
                word_starts = [index for index, token in enumerate(tokenized_sent) if not token.startswith("##")]  #
                # TODO: Simplify this with convert_tokens_to_string
                num_words = len(word_starts)
                # Don't mask boundary tokens, sample same sentence with various masks (less than num_words)
                replacements_pos = rand.sample(range(1, num_words - 1), min(num_words - 2, num_repl))
                for repl_pos in replacements_pos:
                    # Calculate sentence probability for each word in current replacement position
                    print(f"Evaluating sentence {tokenized_sent} replacing word "
                          f"{tokenized_sent[word_starts[repl_pos]:word_starts[repl_pos + 1]]}")
                    # Build sentence embeddings, considering disambiguation
                    sent_row = []
                    for word in self.vocab:
                        sent_row.extend(self.process_sentence(tokenized_sent, word, repl_pos, word_starts,
                                                              verbose=verbose))
                    sent_row = np.array(sent_row)
                    sent_row = sent_row * (sent_row > sparse_thres)  # Cut low probability values
                    self.matrix.append(sent_row)
                    num_sents += 1

        self.matrix = np.array(self.matrix).astype(np.float32)  # Reduce matrix precision, make rows be word-senses
        self.matrix = sparse.csr_matrix(self.matrix.T)  # Convert to sparse matrix

    def process_sentence(self, tokenized_sent, word, repl_pos, word_init, verbose=False):
        """
        Replaces word in repl_pos, incl. all subwords it may contain, for input word with subwords;
        then evaluates the sentence probability.
        If word is ambiguous according to self.senses, then this instance is disambiguated, and the sentence
        probability assigned to corresponding word-sense vector. Each instance only contributes to the
        embedding vector of the closest sense.
        :param tokenized_sent: Input sentence
        :param word: Input word
        :param repl_pos: Word position to replace input word
        :param word_init: List with initial positions of each word in tokenized_sent
        :param verbose:
        :return: curr_prob: Probability of current sentence, stored in the corresponding sense (if ambiguous)
        """
        # Following substitution handles removing sub-words, as well as inserting them
        word_tokens = self.Bert_Model.tokenizer.tokenize(word)
        replaced_sent = tokenized_sent[:word_init[repl_pos]] + word_tokens + tokenized_sent[word_init[repl_pos + 1]:]
        curr_prob = self.Bert_Model.get_sentence_prob_directional(replaced_sent, verbose=verbose)

        # Determine which word-sense vector gets the calculated prob
        sense_centroids = self.senses.get(word, [0])  # If word not in ambiguous dict, return 1-item list
        num_senses = len(sense_centroids)
        prob_list = [0] * num_senses
        self.disamb_vocab.extend([word] * num_senses)  # append word repetitions
        sense_id = 0
        if num_senses > 1:  # If word is ambiguous
            sense_id = self.get_closest_sense(sense_centroids, replaced_sent, word_init[repl_pos],
                                              word_init[repl_pos+ 1])
        prob_list[sense_id] = curr_prob  # Only instance with closest meaning contributes to vector

        return prob_list

    def get_closest_sense(self, sense_centroids, replaced_sent, word_start, word_end):
        """
        Calculates word embedding for word in replaced_sent, then determines which of the
        sense centroids is closest to it.
        :param sense_centroids: Centroids of current word's disambiguated senses
        :param replaced_sent:   Tokenized sentence
        :param word_start:      Position where word starts in replaced_sent
        :param word_end:        Position where word ends in replaced_sent
        :return:                Index of closest centroid
        """
        final_layer = self.Bert_WSD.get_bert_embeddings(replaced_sent)
        embedding = np.mean(final_layer[word_start:word_end], 0)
        distances = cosine_distances(embedding.reshape(1, -1), sense_centroids)

        return np.argmin(distances)

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
            estimator = OPTICS(min_samples=3, metric='cosine', n_jobs=4)
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
        with open(save_to + append + '.wordcat', "w") as fo:
            for i in range(-1, num_clusters):  # Also write unclustered words
                cluster_members = [self.disamb_vocab[j] for j, k in enumerate(labels) if k == i]
                fo.write(f"Cluster #{i}")
                if len(cluster_members) > 0:  # Handle empty clusters
                    fo.write(": \n[")
                    np.savetxt(fo, cluster_members, fmt="%s", newline=", ")
                    fo.write(']\n')
                else:
                    fo.write(" is empty\n\n")

    def eval_clusters(self, logfile, pred):
        """
        Calculate F1 score for predicted categories, against gold standard.
        Uses evaluation from AdaGram's test-all.py, where number of clusters
        can differ btw predicted and gold answers.
        :param logfile:
        :param pred:
        :return:
        """
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
            logfile.write(f"FScore: {f_score}\n")

    @staticmethod
    def get_pairs(labels):
        """
        Used by AdaGram's evaluation method.
        Finds all possible pairs between members of the same cluster.
        :param labels:
        :return:
        """
        result = []
        for label in np.unique(labels):
            ulabels = np.where(labels == label)[0]
            for p in itertools.combinations(ulabels, 2):
                result.append(p)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSD using Transformers')
    parser.add_argument('--pretrained', type=str, default='bert-large-uncased', help='Pretrained model to use')
    parser.add_argument('--use_cuda', action='store_true', help='Use GPU?')
    parser.add_argument('--device', type=str, default='cuda:2', help='GPU Device to Use?')
    parser.add_argument('--sentences', type=str, required=True, help='Sentence Corpus')
    parser.add_argument('--vocab', type=str, required=False, help='Vocabulary Corpus')
    parser.add_argument('--masks', type=int, default=1, help='Min freq of word to be disambiguated')
    parser.add_argument('--sparse_thres', type=int, default=-8, help='Low log(prob) cut')
    parser.add_argument('--clusterer', type=str, default='KMeans', help='Clustering method to use')
    parser.add_argument('--start_k', type=float, default=10, help='Initial value of clustering param')
    parser.add_argument('--end_k', type=float, default=10, help='Final value of clustering param')
    parser.add_argument('--steps_k', type=int, default=5, help='Step for clustering param exploration')
    parser.add_argument('--save_to', type=str, default='test', help='Directory to save disambiguated words')
    parser.add_argument('--verbose', action='store_true', help='Print processing details')
    parser.add_argument('--pickle_WSD', type=str, required=False, help='Pickle file w/sense centroids')
    parser.add_argument('--pickle_vocab', type=str, required=False, help='Pickle file w/vocabulary')
    parser.add_argument('--pickle_emb', type=str, default='test.pickle', help='Pickle file with embeddings/Save '
                                                                               'embeddings to file')
    args = parser.parse_args()

    wc = WordCategorizer(pretrained_model=args.pretrained, use_cuda=args.use_cuda, device_number=args.device)
    no_eval = False  # Flag to trigger evaluation

    if args.vocab:
        print("Using annotated vocabulary file to categorize")
        wc.load_vocabulary(vocab_txt=args.vocab)
    elif args.pickle_vocab:
        print(f"Using pickle vocabulary file to categorize")
        wc.load_vocabulary(vocab_pickle=args.pickle_vocab)
        no_eval = True
    else:
        print("Vocabulary pickle or text file needed")
        exit()

    if args.pickle_WSD:
        print("Word senses file found")
        wc.load_senses(args.pickle_WSD)

    # Heavy part of the process: calculate sentence probabilities for each vocab word
    for _ in tqdm(range(1)):  # Time the process
        wc.load_matrix(args.sentences, args.pickle_emb, num_masks=args.masks, verbose=args.verbose,
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

