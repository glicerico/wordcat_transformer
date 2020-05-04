# Based on the code by Wiedemann et al. (2019, github.com/uhh-lt/bert-sense), and
# modified for unsupervised word-sense disambiguation purposes
# Tries to disambiguate words from sentences in plain text file.
# Similar code that works with xml file sentences is tried in word_senser_XML.py

import os
import pickle
import argparse
import numpy as np
import random as rand

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from spherecluster import SphericalKMeans, VonMisesFisherMixture
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from BertModel import BertLM, BertTok

warnings.filterwarnings('ignore')

MASK = '[MASK]'


class WordSenseModel:
    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=True, freq_threshold=5):
        self.sentences = []  # List of corpus textual sentences
        self.vocab_map = {}  # Dictionary with counts and coordinates of every occurrence of each word
        self.cluster_centroids = {}  # Dictionary with cluster centroid embeddings for each word sense
        self.matrix = []  # sentence-word matrix, containing instance vectors to cluster
        self.pretrained_model = pretrained_model
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.lang_mod = None
        self.estimator = None  # Clustering object
        self.save_dir = None  # Directory to save disambiguated senses
        self.num_senses = None  # Stores nbr of senses for each vocabulary word
        self.freq_threshold = freq_threshold
        self.labels = None

    def apply_bert_tokenizer(self, word):
        return self.lang_mod.tokenizer.tokenize(word)

    def load_matrix(self, pickle_filename, corpus_file, verbose=False, norm_pickle=None, norm_file=''):
        """
        First pass on the corpus sentences. If pickle file is present, load data; else, calculate it.
        This method:
          a) Stores sentences as an array.
          b) Creates dictionary where each vocabulary word is mapped to its occurrences in corpus.
          c) Calculates instance-word matrix, for instances and vocab words in corpus.
        :param pickle_filename
        :param corpus_file
        """
        try:
            with open(pickle_filename, 'rb') as h:
                _data = pickle.load(h)
                self.sentences = _data[0]
                self.vocab_map = _data[1]
                self.matrix = _data[2]

                print("MATRIX FOUND!")

            # Load tokenizer, needed by export_clusters method
            self.lang_mod = BertTok(self.pretrained_model)

        except:
            print("MATRIX File Not Found!! \n")

            print("Loading Bert MLM...")
            self.lang_mod = BertLM(self.pretrained_model, self.device_number, self.use_cuda)

            # Calculate normalization scores
            self.lang_mod.load_norm_scores(norm_pickle, norm_file)

            print("Loading vocabulary")
            self.get_vocabulary(corpus_file, verbose=verbose)
            with open(pickle_filename[:-6] + 'vocab', 'wb') as v:
                pickle.dump(list(self.vocab_map.keys()), v)

            print("Calculate matrix...")
            self.calculate_matrix(verbose=verbose)

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
        if nbr_delete > 0:  # Prevent deleting all words if nbr_delete is zero
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
            print(f"Processing sentence: {bert_tokens}")
            words = self.get_words(bert_tokens)
            word_starts = [index for index, token in enumerate(bert_tokens) if not token.startswith("##")]

            # Replace all words in sentence to get their instance-embeddings
            for word_pos, word in enumerate(words):
                if word not in instances:
                    instances[word] = []
                instances[word].append(embeddings_count)
                embeddings_count += 1
                embedding = []  # Store one word instance (sentence with blank) embedding

                # Calculate common part of sentence probability steps for all words to fill
                # Will only be used when replacement word is composed of one token, otherwise, we need to do the
                # whole calculation
                left_sent = bert_tokens[:word_starts[word_pos + 1]]
                right_sent = bert_tokens[word_starts[word_pos + 2]:]
                common_probs = self.get_common_probs(left_sent, right_sent, verbose=verbose)

                # Calculate sentence's probabilities with different filling words: embedding
                for repl_word in self.vocab_map.keys():
                    word_tokens = self.lang_mod.tokenizer.tokenize(repl_word)
                    if len(word_tokens) > 1:  # Ignore common probs; do whole calculation
                        replaced_sent = left_sent + word_tokens + right_sent
                        score = self.lang_mod.get_sentence_prob_directional(replaced_sent, verbose=verbose)
                        sent_len = len(replaced_sent)
                    else:
                        score = self.complete_probs(common_probs, left_sent, right_sent, repl_word)
                        sent_len = len(left_sent) + len(right_sent) + 1

                    curr_prob = self.lang_mod.normalize_score(sent_len, score)
                    embedding.append(curr_prob)

                # Store this sentence embeddings in the general list
                self.matrix.append(np.float32(embedding))  # Lower precision to save mem, speed

    def complete_probs(self, common_probs, left_sent, right_sent, word_token, verbose=False):
        """
        Given the common probability calculations for a sentence, complete calculations filling blank with word_tokens
        """
        preds_blank_left, preds_blank_right, log_sent_prob_forw, log_sent_prob_back = common_probs
        temp_left = left_sent[:]
        temp_right = right_sent[:]

        # Get probabilities for word filling the blank: b) and g)
        log_sent_prob_forw += self.get_log_prob(preds_blank_left, word_token, len(left_sent), verbose=verbose)
        log_sent_prob_back += self.get_log_prob(preds_blank_right, word_token, len(left_sent), verbose=verbose)

        # Get remaining probs with blank filled: c), d), and h)
        for i in range(1, len(right_sent)):  # d), c)
            temp_right[-1 - i] = MASK
            repl_sent = left_sent + [word_token] + temp_right
            predictions = self.lang_mod.get_predictions(repl_sent)
            log_sent_prob_forw += self.get_log_prob(predictions, right_sent[-1 - i], -1 - i, verbose=verbose)
        for j in range(len(left_sent) - 1):  # h)
            temp_left[1 + j] = MASK
            repl_sent = temp_left + [word_token] + right_sent
            predictions = self.lang_mod.get_predictions(repl_sent)
            log_sent_prob_back += self.get_log_prob(predictions, left_sent[1 + j], 1 + j, verbose=verbose)

        # Obtain geometric average of forward and backward probs
        log_geom_mean_sent_prob = 0.5 * (log_sent_prob_forw + log_sent_prob_back)
        if verbose:
            print(f"Raw forward sentence probability: {log_sent_prob_forw}")
            print(f"Raw backward sentence probability: {log_sent_prob_back}\n")
            print(f"Average normalized sentence prob: {log_geom_mean_sent_prob}\n")

        return np.power(10, log_geom_mean_sent_prob)

    def get_log_prob(self, predictions, token, position, verbose=False):
        """
        Given BERT's predictions, return probability for required token, in required position
        """
        probs_first = self.lang_mod.sm(predictions[0, position])  # Softmax to get probabilities for first (sub)word
        if verbose:
            self.lang_mod.print_top_predictions(probs_first)
        log_prob_first = probs_first[self.lang_mod.tokenizer.convert_tokens_to_ids(token)]

        return np.log10(log_prob_first.detach().cpu().numpy())

    def get_common_probs(self, left_sent, right_sent, verbose=False):
        """
        Calculate partial forward and backwards probabilities of sentence probability estimation, for
        the sections that are common to all iterations of a fill-in-the-blank process.
        Example sentence: "Not ___ real sentence". We need probabilities:
        FORWARD:
        a) P(M1 = Not           |M1 M2 M3 M4)
        b) P(M2 = ___           |Not M2 M3 M4)
        c) P(M3 = real          |Not ___ M3 M4)
        d) P(M4 = sentence      |Not ___ real M4)
        BACKWARD:
        e) P(M4 = sentence      |M1 M2 M3 M4)
        f) P(M3 = real          |M1 M2 M3 sentence)
        g) P(M2 = ___           |M1 M2 real sentence)
        h) P(M1 = Not           |M1 ___ real sentence)
        :param left_sent:   Tokens before the blank
        :param right_sent:  Tokens after the blank
        :param verbose:
        :return:            log10(a)) as log_common_prob_forw,
                            log10(e) * f)) as log_common_prob_back,
                            The whole vocabulary prediction array for both b) and g), to be used later by all
                            words filling the blank.
        """
        masks_left = ['[CLS]'] + [MASK] * (len(left_sent) - 1)
        masks_right = [MASK] * (len(right_sent) - 1) + ['[SEP]']
        temp_left = left_sent[:]
        temp_right = right_sent[:]
        log_common_prob_forw = 0
        log_common_prob_back = 0

        # Estimate a) and e) if they are not the position of the blank
        repl_sent = masks_left + [MASK] + masks_right  # Fully masked sentence
        predictions = self.lang_mod.get_predictions(repl_sent)
        if len(left_sent) > 1:
            log_common_prob_forw += self.get_log_prob(predictions, left_sent[1], 1, verbose=verbose)
        if len(right_sent) > 1:
            log_common_prob_back += self.get_log_prob(predictions, right_sent[-2], len(repl_sent) - 2, verbose=verbose)

        # Get all predictions for b)
        repl_sent = left_sent + [MASK] + masks_right
        preds_blank_left = self.lang_mod.get_predictions(repl_sent)

        # Get all predictions for g)
        repl_sent = masks_left + [MASK] + right_sent
        preds_blank_right = self.lang_mod.get_predictions(repl_sent)

        # Estimate common probs for forward sentence probability
        for i in range(1, len(left_sent) - 1):  # Skip [CLS] token
            temp_left[-i] = MASK
            repl_sent = temp_left + [MASK] + masks_right
            predictions = self.lang_mod.get_predictions(repl_sent)
            log_common_prob_forw += self.get_log_prob(predictions, left_sent[-i], len(left_sent) - i, verbose=verbose)

        # Estimate common probs for backwards sentence probability (f in the example)
        for j in range(len(right_sent) - 2):
            temp_right[j] = MASK
            repl_sent = masks_left + [MASK] + temp_right
            predictions = self.lang_mod.get_predictions(repl_sent)
            log_common_prob_back += self.get_log_prob(predictions, right_sent[j], len(left_sent) + 1 + j, verbose=verbose)

        return preds_blank_left, preds_blank_right, log_common_prob_forw, log_common_prob_back

    @staticmethod
    def plot_instances(embeddings, labels, word):
        """
        Plot word-instance embeddings
        :param embeddings:
        :param labels:
        :param word:
        :return:
        """
        # PCA processing
        comps_pca= min(3, len(embeddings))
        pca = PCA(n_components=comps_pca)
        pca_result = pca.fit_transform(embeddings)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        # t-SNE processing
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)

        # PLOTTING
        plt.figure()
        plt.subplot(211)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels)
        plt.title(word)

        plt.subplot(212)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels)
        plt.show()
        print("PLOTTED")

    def init_estimator(self, save_to, clust_method='OPTICS', **kwargs):
        if clust_method == 'OPTICS':
            self.min_samples = kwargs.get('min_samples', 1)
            # Init clustering object
            self.estimator = OPTICS(min_samples=self.min_samples, metric='cosine', n_jobs=4)
            self.save_dir = save_to + "_OPTICS_minsamp" + str(self.min_samples)
        elif clust_method == 'KMeans':
            k = kwargs.get('k', 5)  # 5 is default value, if no kwargs were passed
            self.freq_threshold = max(self.freq_threshold, k)
            self.estimator = KMeans(init="k-means++", n_clusters=k, n_jobs=4)
            self.save_dir = save_to + "_KMeans_k" + str(k)
        elif clust_method == 'DBSCAN':
            self.min_samples = kwargs.get('min_samples', 2)
            eps = kwargs.get('eps', 0.3)
            self.estimator = DBSCAN(metric='cosine', n_jobs=4, min_samples=self.min_samples, eps=eps)
            self.save_dir = save_to + "_DBSCAN_minsamp" + str(self.min_samples) + '_eps' + str(eps)
        elif clust_method == 'SphericalKMeans':
            k = kwargs.get('k', 5)  # 5 is default value, if no kwargs were passed
            self.freq_threshold = max(self.freq_threshold, k)
            self.estimator = SphericalKMeans(n_clusters=k, n_jobs=4)
            self.save_dir = save_to + "_SphericalKMeans_k" + str(k)
        elif clust_method == 'movMF-soft':
            k = kwargs.get('k', 5)  # 5 is default value, if no kwargs were passed
            self.freq_threshold = max(self.freq_threshold, k)
            self.estimator = VonMisesFisherMixture(n_clusters=k, posterior_type="soft")
            self.save_dir = save_to + "_movMF-soft_k" + str(k)
        elif clust_method == 'movMF-hard':
            k = kwargs.get('k', 5)  # 5 is default value, if no kwargs were passed
            self.freq_threshold = max(self.freq_threshold, k)
            self.estimator = VonMisesFisherMixture(n_clusters=k, posterior_type="hard")
            self.save_dir = save_to + "_movMF-hard_k" + str(k)
        else:
            print("Clustering methods implemented are: OPTICS, DBSCAN, KMeans, SphericalKMeans, movMF-soft, movMF-hard")
            exit(1)

    def disambiguate(self, plot=False):
        """
        Disambiguate word senses through clustering their transformer embeddings.
        Clustering is done using the sklearn algorithm selected in init_estimator()
        :param plot:            Flag to plot 2D projection of word instance embeddings
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        fl = open(self.save_dir + "/clustering.log", 'w')  # Logging file
        fl.write(f"# WORD\t\tCLUSTERS\n")

        # Stores nbr of senses for each vocab word
        self.num_senses = []
        self.labels = {}

        # Loop for each word in vocabulary
        for word, instances in self.vocab_map.items():
            # Build embeddings list for this word
            curr_embeddings = [self.matrix[row] for _, _, row in instances]
            curr_embeddings = normalize(curr_embeddings)  # Make unit vectors

            if len(curr_embeddings) < self.freq_threshold:  # Don't disambiguate if word is infrequent
                print(f"Won't cluster: word \"{word}\" frequency is lower than threshold")
                self.num_senses.append(1)
                continue

            print(f'Disambiguating word \"{word}\"...')
            self.estimator.fit(curr_embeddings)  # Disambiguate
            self.labels[word] = self.estimator.labels_
            if plot:
                self.plot_instances(curr_embeddings, self.estimator.labels_, word)

            self.export_clusters(fl, word, self.estimator.labels_)

        with open(self.save_dir + '.labels', 'wb') as flabels:
            pickle.dump(self.labels, flabels)

        fl.write("\n")
        fl.close()

    def export_clusters(self, fl, word, labels):
        """
        Write clustering results to files
        :param fl:              handle for logging file
        :param word:            Current word to disambiguate
        :param labels:          Cluster labels for each word instance
        """
        num_clusters = max(labels) + 1
        self.num_senses.append(num_clusters)
        print(f"Num clusters: {num_clusters}")
        fl.write(f"{word}\t\t{num_clusters}\n")

        # Write senses to file, with some sentence examples
        with open(self.save_dir + '/' + word + ".disamb", "w") as fo:
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

                else:
                    fo.write(" is empty\n\n")


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
    parser.add_argument('--clustering', type=str, default='SphericalKmeans', help='Clustering method to use')
    parser.add_argument('--verbose', action='store_true', help='Print processing details')
    parser.add_argument('--plot', action='store_true', help='Plot word embeddings?')
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

    WSD = WordSenseModel(pretrained_model=args.pretrained, device_number=args.device, use_cuda=args.use_cuda,
                         freq_threshold=args.threshold)

    print("Obtaining word embeddings...")
    WSD.load_matrix(args.pickle_emb, args.corpus, verbose=args.verbose, norm_pickle=args.norm_pickle,
                    norm_file=args.norm_file)

    # Remove top words from disambiguation
    print(f"Removing the top {args.func_frac} fraction of words")
    WSD.remove_function_words(args.func_frac)

    print("Start disambiguation...")
    for nn in range(args.start_k, args.end_k + 1, args.step_k):
        WSD.init_estimator(args.save_to, clust_method=args.clustering, k=nn)
        WSD.disambiguate(plot=args.plot)

    print("\n\n*******************************************************")
    print(f"WSD finished. Output files written in {args.save_to}")
