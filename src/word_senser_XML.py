# Based on the code by Wiedemann et al. (2019, github.com/uhh-lt/bert-sense), and
# modified for unsupervised word-sense disambiguation purposes

import os
import pickle
import xml.etree.ElementTree as ET
import torch
import argparse
import numpy as np
import random as rand

from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans, OPTICS, DBSCAN, cluster_optics_dbscan

from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class BERT:
    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=True, output_hidden_states=True):
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model, output_hidden_states=output_hidden_states)
        with torch.no_grad():
            self.model.eval()

        if use_cuda:
            self.model.to(device_number)


class WordSenseModel:
    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=True):
        self.sentences = []  # List of corpus textual sentences
        self.vocab_map = {}  # Dictionary that stores coordinates of every occurrence of each word
        self.cluster_centroids = {}  # Dictionary with cluster centroid embeddings for each word sense
        self.embeddings = []  # Embeddings for all words in corpus

        self.device_number = device_number
        self.use_cuda = use_cuda

        self.Bert_Model = BERT(pretrained_model, device_number, use_cuda)

    @staticmethod
    def open_xml_file(file_name):
        tree = ET.parse(file_name)
        root = tree.getroot()

        return root, tree

    @staticmethod
    def semeval_sent_sense_collect(xml_struct):
        _sent = []
        _sent1 = ""
        _senses = []
        for idx, j in enumerate(xml_struct.iter('word')):
            _temp_dict = j.attrib
            words = _temp_dict['surface_form'].lower()
            if '*' not in words:
                _sent1 += words + " "
                _sent.extend([words])
                if 'wn30_key' in _temp_dict:
                    _senses.extend([_temp_dict['wn30_key'].split(';')[0]] * len([words]))  # Keep 1st sense only
                else:
                    _senses.extend([0] * len([words]))

        return _sent, _sent1, _senses

    def apply_bert_tokenizer(self, word):
        return self.Bert_Model.tokenizer.tokenize(word)

    def collect_bert_tokens(self, _sent):
        _bert_tokens = ['[CLS]', ]
        for idx, j in enumerate(_sent):
            _tokens = self.apply_bert_tokenizer(_sent[idx])
            _bert_tokens.extend(_tokens)
        _bert_tokens.append('[SEP]')

        return _bert_tokens

    def get_bert_embeddings(self, tokens):
        _ib = self.Bert_Model.tokenizer.convert_tokens_to_ids(tokens)
        _st = [0] * len(_ib)
        if self.use_cuda:
            _t1, _t2 = torch.tensor([_ib]).to(self.device_number), torch.tensor([_st]).to(self.device_number)
        else:
            _t1, _t2 = torch.tensor([_ib]), torch.tensor([_st])

        with torch.no_grad():
            _, _, _encoded_layers = self.Bert_Model.model(_t1, token_type_ids=_t2)
            # Average last 4 hidden layers (second best result from Devlin et al. 2019)
            _e1 = _encoded_layers[-4:]
            _e2 = torch.cat((_e1[0], _e1[1], _e1[2], _e1[3]), 0)
            _e3 = torch.mean(_e2, dim=0)
            if self.use_cuda:
                _final_layer = _e3.cpu().numpy()
            else:
                _final_layer = _e3.numpy()
                _final_layer = np.around(_final_layer, decimals=5)  # LOWER PRECISION, process faster. TODO: Check!

        return _final_layer

    def load_embeddings(self, pickle_file_name, corpus_file, mode):
        """
        First pass on the corpus sentences. If pickle file is present, load data; else, calculate it.
        This method:
          a) Stores sentences as an array.
          b) Creates dictionary where each vocabulary word is mapped to its occurrences in corpus.
          c) Calculates embeddings for each vocabulary word.
        :param pickle_file_name
        :param corpus_file
        :param mode:            Determine if only gold-ambiguous words are stored
        """
        try:
            with open(pickle_file_name, 'rb') as h:
                _data = pickle.load(h)
                self.sentences = _data[0]
                self.vocab_map = _data[1]
                self.embeddings = _data[2]

                print("EMBEDDINGS FOUND!")

        except:
            print("Embedding File Not Found!! \n")
            print("Performing first pass...")

            self.calculate_embeddings(corpus_file, mode)
            with open(pickle_file_name, 'wb') as h:
                _data = (self.sentences, self.vocab_map, self.embeddings)
                pickle.dump(_data, h)

            print("Data stored in " + pickle_file_name)

    def calculate_embeddings(self, corpus_file, mode):
        """
        Calculates embeddings for all words in corpus_file, creates vocabulary dictionary
        :param corpus_file:     file to get vocabulary
        :param mode:            Determine if only gold-ambiguous words are stored
        """
        _test_root, _test_tree = self.open_xml_file(corpus_file)
        stored_embeddings = 0
        fk = open(corpus_file[:-3] + 'key', 'w')  # Key to GOLD word senses
        inst_counter = 0  # Useless instance counter needed for evaluator

        # Process each sentence in corpus
        for sent_nbr, i in tqdm(enumerate(_test_root.iter('sentence'))):
            sent_embeddings = []  # Store one sentence's word embeddings as elements
            sent, sent1, senses = self.semeval_sent_sense_collect(i)
            self.sentences.append(sent1)
            bert_tokens = self.collect_bert_tokens(sent)
            final_layer = self.get_bert_embeddings(bert_tokens)

            token_count = 1
            # Process all words in sentence
            for word_pos, j in enumerate(zip(sent, senses)):
                word = j[0]
                sense = j[1]
                word_len = len(self.apply_bert_tokenizer(word))  # Handle subwords

                if mode == 'eval_only' and sense == 0:  # Don't store embedding if it's not gold-ambiguous
                    sent_embeddings.append(0)
                else:
                    # Save sense in key file
                    fk.write(f"{word} {inst_counter} {sense}\n")
                    inst_counter += 1

                    embedding = np.mean(final_layer[token_count:token_count + word_len], 0)
                    sent_embeddings.append(np.float32(embedding))  # Lower precision to save mem, speed
                    stored_embeddings += 1

                    # Register word location in vocabulary dictionary
                    if word not in self.vocab_map.keys():
                        self.vocab_map[word] = []
                    self.vocab_map[word].append((sent_nbr, word_pos))

                token_count += word_len

            # Store this sentence embeddings in the general list
            self.embeddings.append(sent_embeddings)

        fk.close()
        print(f"{stored_embeddings} EMBEDDINGS STORED")

    def disambiguate(self, save_dir, clust_method='OPTICS', freq_threshold=5, **kwargs):
        """
        Disambiguate word senses through clustering their transformer embeddings
        Clustering is done using the selected sklearn algorithm.
        If OPTICS method is used, then DBSCAN clusters are also obtained

        :param save_dir:        Directory to save disambiguated senses
        :param clust_method:    Clustering method used
        :param freq_threshold:  Frequency threshold for a word to be disambiguated
        :param kwargs:          Clustering parameters
        """
        # Use OPTICS estimator also to get DBSCAN clusters
        if clust_method == 'OPTICS':
            min_samples = kwargs.get('min_samples', 1)
            # Init clustering object
            estimator = OPTICS(min_samples=min_samples, metric='cosine', n_jobs=4, max_eps=0.4)
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
        fk = open(save_to + "/disamb.pred", 'w')  # Predictions for evaluation against GOLD

        # Loop for each word in vocabulary
        for word, instances in self.vocab_map.items():
            # Build embeddings list for this word
            curr_embeddings = []
            for instance in instances:
                x, y = instance  # Get current word instance coordinates
                curr_embeddings.append(self.embeddings[x][y])

            if len(curr_embeddings) < freq_threshold:  # Don't disambiguate if word is uncommon
                print(f"Word \"{word}\" frequency out of threshold")
                continue

            print(f'Disambiguating word \"{word}\"...')
            estimator.fit(curr_embeddings)  # Disambiguate
            self.cluster_centroids[word] = self.export_clusters(fl, save_to, word, estimator.labels_)
            self.write_predictions(fk, word, estimator.labels_, instances)

        fl.write("\n")
        fl.close()
        fk.close()

    @staticmethod
    def write_predictions(fk, word, labels, instances):
        for count, label in enumerate(labels):
            fk.write(f"{word} {count} {label}\n")

    def export_clusters(self, fl, save_dir, word, labels):
        """
        Write clustering results to file
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
                    np.savetxt(fo, sense_members, fmt="(%s, %s)", newline=", ")
                    fo.write(']\n')
                    # Write at most 3 sentence examples for the word sense
                    sent_samples = rand.sample(sense_members, min(len(sense_members), 3))
                    fo.write('Samples:\n')
                    # Write sample sentences to file, with focus word in CAPS for easier reading
                    for sample, focus_word in sent_samples:
                        bold_sent = self.sentences[sample].split()
                        bold_sent[focus_word] = bold_sent[focus_word].upper()
                        fo.write(" ".join(bold_sent) + '\n')

                    # Calculate cluster centroid and save
                    sense_embeddings = [self.embeddings[x][y] for x, y in sense_members]
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
    parser.add_argument('--start_k', type=int, default=10, help='First number of clusters to use in KMeans')
    parser.add_argument('--end_k', type=int, default=10, help='Final number of clusters to use in KMeans')
    parser.add_argument('--step_k', type=int, default=1, help='Increase in number of clusters to use')
    parser.add_argument('--save_to', type=str, default='test', help='Directory to save disambiguated words')
    parser.add_argument('--pretrained', type=str, default='bert-large-uncased', help='Pretrained model to use')
    parser.add_argument('--mode', type=str, default='eval_only', help='Determines if all words need to be clustered')
    parser.add_argument('--clustering', type=str, default='OPTICS', help='Clustering method to use')
    parser.add_argument('--pickle_file', type=str, default='test.pickle', help='Pickle file of Bert Embeddings/Save '
                                                                               'Embeddings to file')

    args = parser.parse_args()

    print("Corpus is: " + args.corpus)

    if args.use_cuda:
        print("Processing with CUDA!")

    else:
        print("Processing without CUDA!")

    if args.mode == "eval_only":
        print("Processing only ambiguous words in training corpus...")
    else:
        print("Processing all words below threshold")

    print("Loading WSD Model!")
    WSD = WordSenseModel(args.pretrained, device_number=args.device, use_cuda=args.use_cuda)

    print("Obtaining word embeddings...")
    WSD.load_embeddings(args.pickle_file, args.corpus, args.mode)

    print("Start disambiguation...")
    for nn in range(args.start_k, args.end_k + 1, args.step_k):
        WSD.disambiguate(args.save_to, clust_method=args.clustering, freq_threshold=args.threshold, k=nn)

    print("\n\n*******************************************************")
    print(f"WSD finished. Output files written in {args.save_to}")

