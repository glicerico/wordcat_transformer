# Based on the code by Wiedemann et al. (2019, github.com/uhh-lt/bert-sense), and
# modified for unsupervised word-sense disambiguation purposes
# Tries to disambiguate words from sentences in plain text file.
# Similar code that works with xml file sentences is tried in word_senser_XML.py

import os
import pickle
import torch
import argparse
import numpy as np
import random as rand

from sklearn.cluster import KMeans, OPTICS, DBSCAN
from tqdm import tqdm
import warnings

from BertModel import BertLM

warnings.filterwarnings('ignore')


class WordSenseModel:
    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=True):
        self.sentences = []  # List of corpus textual sentences
        self.vocab_map = {}  # Dictionary with counts and coordinates of every occurrence of each word
        self.cluster_centroids = {}  # Dictionary with cluster centroid embeddings for each word sense
        self.matrix = []  # sentence-word matrix, containing instance vectors to cluster
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.lang_mod = BertLM(pretrained_model, device_number, use_cuda)

    def apply_bert_tokenizer(self, word):
        return self.lang_mod.tokenizer.tokenize(word)

    def load_matrix(self, pickle_filename, corpus_file, func_frac, verbose=False):
        """
        First pass on the corpus sentences. If pickle file is present, load data; else, calculate it.
        This method:
          a) Stores sentences as an array.
          b) Creates dictionary where each vocabulary word is mapped to its occurrences in corpus.
          c) Calculates instance-word matrix, for instances and vocab words in corpus.
        :param pickle_filename
        :param corpus_file
        :param func_frac:       Fraction of words that are considered functional words (ignored for disamb)
        """
        try:
            with open(pickle_filename, 'rb') as h:
                _data = pickle.load(h)
                self.sentences = _data[0]
                self.vocab_map = _data[1]
                self.matrix = _data[2]

                print("MATRIX FOUND!")

        except:
            print("MATRIX File Not Found!! \n")

            print("Loading vocabulary")
            self.get_vocabulary(corpus_file, verbose=verbose)
            with open(pickle_filename[:-6] + 'vocab', 'wb') as v:
                pickle.dump(list(self.vocab_map.keys()), v)

            print("Calculate matrix...")
            self.calculate_matrix(verbose=verbose)
            print(f"Removing the top {func_frac} fraction of words")
            self.remove_function_words(func_frac)

            with open(pickle_filename, 'wb') as h:
                _data = (self.sentences, self.vocab_map, self.matrix)
                pickle.dump(_data, h)

            print("Data stored in " + pickle_filename)

    def get_words(self, tokenized_sent):
        """
        Returns the complete words in a BERT-tokenized sentence (merges sub-words)
        :param tokenized_sent:
        :return:
        """
        sentence = self.lang_mod.tokenizer.convert_tokens_to_string(tokenized_sent[1:-1])  # Ignore boundary tokens
        return sentence.split()

    def remove_function_words(self, functional_threshold):
        """
        Remove top words from vocabulary, assuming that most common words are functional words,
        which we don't want to disambiguate
        :param functional_threshold:    Fraction of words to remove
        """
        sorted_vocab = sorted(self.vocab_map.items(), key=lambda kv: len(kv[1]))  # Sort words by frequency
        nbr_delete = int(len(sorted_vocab) * functional_threshold)  # Nbr of words to delete
        self.vocab_map = dict(sorted_vocab[:-nbr_delete])  # Delete most common words

    def get_vocabulary(self, corpus_file, verbose=False):
        """
        Reads all word instances in file, stores their location
        :param corpus_file:     file to get vocabulary
        """
        with open(corpus_file, 'r') as fi:
            instance_nbr = 0
            # Process each sentence in corpus
            for sent_nbr, sent in tqdm(enumerate(fi)):
                bert_tokens = self.lang_mod.tokenize_sent(sent)
                self.sentences.append(bert_tokens)
                words = self.get_words(bert_tokens)
                # Store word instances in vocab_map
                for word_pos, word in enumerate(words):
                    if word not in self.vocab_map:
                        self.vocab_map[word] = []
                    # TODO: Can avoid storing coordinates if not CAPS target word in export_clusters
                    self.vocab_map[word].append((sent_nbr, word_pos, instance_nbr))  # Register instance location
                    instance_nbr += 1
        if verbose:
            print("Vocabulary:")
            print(self.vocab_map)

    def calculate_matrix(self, verbose=False):
        """
        Calculates embeddings for all word instances in corpus_file
        """
        instances = {}  # Stores matrix indexes for each instance embedding
        embeddings_count = 0  # Counts embeddings created (matrix row nbr)
        # Process each sentence in corpus
        for bert_tokens in tqdm(self.sentences):
            sent_rows = []
            words = self.get_words(bert_tokens)
            word_starts = [index for index, token in enumerate(bert_tokens) if not token.startswith("##")]

            # Replace all words in sentence to get their instance-embeddings
            for word_pos, word in enumerate(words):
                if word not in instances:
                    instances[word] = []
                instances[word].append(embeddings_count)
                embeddings_count += 1
                # self.build_embedding()
                embedding = []  # Store one word instance (sentence with blank) embedding
                # Calculate sentence's probabilities with different filling words: embedding
                for repl_word in self.vocab_map.keys():
                    word_tokens = self.lang_mod.tokenizer.tokenize(repl_word)
                    replaced_sent = bert_tokens[:word_starts[word_pos + 1]] + word_tokens + bert_tokens[
                                                                                     word_starts[word_pos + 2]:]
                    curr_prob = self.lang_mod.get_sentence_prob_normalized(replaced_sent, verbose=verbose)
                    embedding.append(curr_prob)

                # Store this sentence embeddings in the general list
                self.matrix.append(np.float32(embedding))  # Lower precision to save mem, speed

    def disambiguate(self, save_dir, clust_method='OPTICS', freq_threshold=5, pickle_cent='test_cent.pickle', **kwargs):
        """
        Disambiguate word senses through clustering their transformer embeddings.
        Clustering is done using the selected sklearn algorithm.
        If OPTICS method is used, then DBSCAN clusters are also obtained
        :param save_dir:        Directory to save disambiguated senses
        :param clust_method:    Clustering method used
        :param freq_threshold:  Frequency threshold for a word to be disambiguated
        :param pickle_cent:     Pickle file to store cluster centroids
        :param kwargs:          Clustering parameters
        """
        # Use OPTICS estimator also to get DBSCAN clusters
        if clust_method == 'OPTICS':
            min_samples = kwargs.get('min_samples', 1)
            # Init clustering object
            estimator = OPTICS(min_samples=min_samples, metric='cosine', n_jobs=4)
            save_to = save_dir + "_OPTICS_minsamp" + str(min_samples)
        elif clust_method == 'KMeans':
            k = kwargs.get('k', 5)  # 5 is default value, if no kwargs were passed
            freq_threshold = max(freq_threshold, k)
            estimator = KMeans(init="k-means++", n_clusters=k, n_jobs=4)
            save_to = save_dir + "_KMeans_k" + str(k)
        elif clust_method == 'DBSCAN':
            min_samples = kwargs.get('min_samples', 2)
            eps = kwargs.get('eps', 0.3)
            estimator = DBSCAN(metric='cosine', n_jobs=4, min_samples=5, eps=eps)
            save_to = save_dir + "_DBSCAN_minsamp" + str(min_samples) + '_eps' + str(eps)
        else:
            print("Clustering methods implemented are: OPTICS, DBSCAN, KMeans")
            exit(1)

        if not os.path.exists(save_to):
            os.makedirs(save_to)
        fl = open(save_to + "/clustering.log", 'w')  # Logging file
        fl.write(f"# WORD\t\tCLUSTERS\n")

        # Loop for each word in vocabulary
        for word, instances in self.vocab_map.items():
            # Build embeddings list for this word
            curr_embeddings = [self.matrix[row] for _, _, row in instances]

            if len(curr_embeddings) < freq_threshold:  # Don't disambiguate if word is infrequent
                print(f"Word \"{word}\" frequency lower than threshold")
                continue

            print(f'Disambiguating word \"{word}\"...')
            estimator.fit(curr_embeddings)  # Disambiguate
            curr_centroids = self.export_clusters(fl, save_to, word, estimator.labels_)
            if len(curr_centroids) > 1:  # Only store centroids for ambiguous words
                self.cluster_centroids[word] = curr_centroids

        with open(pickle_cent, 'wb') as h:
            pickle.dump(self.cluster_centroids, h)

        print("Cluster centroids stored in " + pickle_cent)

        fl.write("\n")
        fl.close()

    def export_clusters(self, fl, save_dir, word, labels):
        """
        Write clustering results to files
        :param fl:              handle for logging file
        :param save_dir:        Directory to save disambiguated senses
        :param word:            Current word to disambiguate
        :param labels:          Cluster labels for each word instance
        """
        sense_centroids = []  # List with word sense centroids
        num_clusters = max(labels) + 1
        print(f"Num clusters: {num_clusters}")
        fl.write(f"{word}\t\t{num_clusters}\n")

        # Write senses to file, with some sentence examples
        with open(save_dir + '/' + word + ".disamb", "w") as fo:
            for i in range(-1, num_clusters):  # Also write unclustered words
                sense_members = [self.vocab_map[word][j] for j, k in enumerate(labels) if k == i]
                fo.write(f"Cluster #{i}")
                if len(sense_members) > 0:  # Handle empty clusters
                    fo.write(": \n[")
                    np.savetxt(fo, sense_members, fmt="(%s, %s, %s)", newline=", ")
                    fo.write(']\n')
                    # Write at most 3 sentence examples for the word sense
                    sent_samples = rand.sample(sense_members, min(len(sense_members), 3))
                    fo.write('Samples:\n')
                    # Write sample sentences to file, with focus word in CAPS for easier reading
                    for sample, focus_word, _ in sent_samples:
                        bold_sent = self.get_words(self.sentences[sample])
                        bold_sent[focus_word] = bold_sent[focus_word].upper()
                        fo.write(" ".join(bold_sent) + '\n')

                    # Calculate cluster centroid and save
                    if i >= 0:  # Don't calculate centroid for unclustered (noise) instances
                        sense_embeddings = [self.matrix[row] for _, _, row in sense_members]
                        sense_centroids.append(np.mean(sense_embeddings, 0))
                else:
                    fo.write(" is empty\n\n")

        return sense_centroids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WSD using BERT')

    parser.add_argument('--use_cuda', action='store_true', help='Use GPU?')
    parser.add_argument('--device', type=str, default='cuda:2', help='GPU Device to Use?')
    parser.add_argument('--corpus', type=str, required=True, help='Training Corpus')
    parser.add_argument('--threshold', type=int, default=2, help='Min freq of word to be disambiguated')
    parser.add_argument('--func_frac', type=float, default=0.05, help='Top fraction of words considered functional')
    parser.add_argument('--start_k', type=int, default=10, help='First number of clusters to use in KMeans')
    parser.add_argument('--end_k', type=int, default=10, help='Final number of clusters to use in KMeans')
    parser.add_argument('--step_k', type=int, default=1, help='Increase in number of clusters to use')
    parser.add_argument('--save_to', type=str, default='test', help='Directory to save disambiguated words')
    parser.add_argument('--pretrained', type=str, default='bert-large-uncased', help='Pretrained model to use')
    parser.add_argument('--clustering', type=str, default='OPTICS', help='Clustering method to use')
    parser.add_argument('--pickle_cent', type=str, default='test_cent.pickle', help='Pickle file for cluster centroids')
    parser.add_argument('--verbose', action='store_true', help='Print processing details')
    parser.add_argument('--pickle_emb', type=str, default='test.pickle', help='Pickle file for Embeddings/Save '
                                                                               'Embeddings to file')
    parser.add_argument('--norm_file', type=str, default='', help='Sentences file to use for normalization')
    parser.add_argument('--norm_pickle', type=str, default='test.pickle', help='Pickle file to use for normalization')

    args = parser.parse_args()

    print("Corpus is: " + args.corpus)

    if args.use_cuda:
        print("Processing with CUDA!")

    else:
        print("Processing without CUDA!")

    print("Loading WSD Model!")
    WSD = WordSenseModel(pretrained_model=args.pretrained, device_number=args.device, use_cuda=args.use_cuda)

    # Calculate normalization scores if option is present
    WSD.lang_mod.load_norm_scores(args.norm_pickle, args.norm_file)

    print("Obtaining word embeddings...")
    WSD.load_matrix(args.pickle_emb, args.corpus, args.func_frac, verbose=args.verbose)

    print("Start disambiguation...")
    for nn in range(args.start_k, args.end_k + 1, args.step_k):
        WSD.disambiguate(args.save_to, clust_method=args.clustering, freq_threshold=args.threshold, k=nn,
                         pickle_cent=args.pickle_cent)

    print("\n\n*******************************************************")
    print(f"WSD finished. Output files written in {args.save_to}")

