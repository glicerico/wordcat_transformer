import argparse
import os
import pickle

import numpy as np
import random as rand
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


class WordCategorizer:
    def __init__(self):
        self.matrix = None  # Stores sent probability for each word-sentence pair (rows are words)
        self.sentences = None  # List of corpus textual sentences
        self.vocab_map = None  # Dictionary with counts and coordinates of every occurrence of each word
        self.num_senses = None  # Stores nbr of senses for each vocabulary word
        self.wsd_labels = None  # Stores labels for each vocabulary word

    def load_senses(self, pickle_senses):
        """
        Load ambiguous word senses, as stored by word_senser.py
        :param pickle_senses:
        :return:
        """
        try:
            with open(pickle_senses, 'rb') as fs:
                self.num_senses, self.wsd_labels = pickle.load(fs)
            print("WSD data successfully loaded!\n")
        except:
            print("ERROR: Loading WSD data failed!!\n")
            exit(1)

    def load_matrix(self, pickle_emb, verbose=False):
        """
        If pickle file is present, load data; else, calculate it.
        :param pickle_emb:          File to load embeddings
        :param verbose:
        :return:
        """
        try:
            with open(pickle_emb, 'rb') as h:
                _data = pickle.load(h)
                self.sentences = _data[0]
                self.vocab_map = _data[1]
                self.matrix = _data[2]

            print("MATRIX FOUND!")

        except:
            print("MATRIX File Not Found!! \n")
            exit(1)

        # If word is ambiguous according to self.senses, then this instance is disambiguated, and the sentence
        # probability assigned to corresponding word-sense vector. Each instance only contributes to the
        # embedding vector of the closest sense.

    def cluster_words(self, method='KMeans', **kwargs):
        if method == 'KMeans':
            k = kwargs.get('k', 2)  # 2 is default value, if no kwargs were passed
            estimator = KMeans(n_clusters=int(k), n_jobs=4)
            estimator.fit(self.matrix)  # Cluster matrix
        elif method == 'DBSCAN':
            eps = kwargs.get('k', 0.2)
            min_samples = kwargs.get('min_samples', 3)
            estimator = DBSCAN(min_samples=min_samples, eps=eps, n_jobs=4, metric='cosine')
            estimator.fit(self.matrix)  # Cluster matrix
        elif method == 'OPTICS':
            estimator = OPTICS(min_samples=2, metric='cosine', n_jobs=4)
            estimator.fit(self.matrix)  # Cluster matrix
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word categorization using BERT')
    parser.add_argument('--clusterer', type=str, default='KMeans', help='Clustering method to use')
    parser.add_argument('--start_k', type=float, default=10, help='Initial value of clustering param')
    parser.add_argument('--end_k', type=float, default=10, help='Final value of clustering param')
    parser.add_argument('--steps_k', type=int, default=1, help='Step for clustering param exploration')
    parser.add_argument('--save_to', type=str, default='test', help='Directory to save word categories')
    parser.add_argument('--verbose', action='store_true', help='Print processing details')
    parser.add_argument('--pickle_WSD', type=str, required=False, help='Pickle file WSD info')
    parser.add_argument('--pickle_emb', type=str, default='test.pickle', help='Pickle file with embeddings matrix')
    args = parser.parse_args()

    wc = WordCategorizer()

    # Load probability matrix for sentence-word pairs
    wc.load_matrix(args.pickle_emb, verbose=args.verbose)

    # Load WSD data
    if args.pickle_WSD:
        print("Word senses file found")
        wc.load_senses(args.pickle_WSD)
        # Restructure matrix with WSD info
        wc.restructure_matrix()  # TODO: implement

    print("Start clustering...")
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    with open(args.save_to + '/results.log', 'w') as fl:
        for curr_k in tqdm(np.linspace(args.start_k, args.end_k, args.steps_k)):
            print(f"Clustering with k={curr_k}")
            cluster_labels = wc.cluster_words(method=args.clusterer, k=curr_k)
            wc.write_clusters(args.clusterer, args.save_to, cluster_labels, curr_k)
