import os
import pickle
import xml.etree.ElementTree as ET
import torch
import argparse
import numpy as np
import random as rand

from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, DBSCAN

from tqdm import tqdm, trange
import warnings

warnings.filterwarnings('ignore')


class BERT:

    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=True):
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        self.model = BertModel.from_pretrained(pretrained_model, output_hidden_states=True)
        self.model.eval()

        if use_cuda:
            self.model.to(device_number)


class WordSenseModel:

    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=True):

        self.device_number = device_number
        self.use_cuda = use_cuda

        self.Bert_Model = BERT(pretrained_model, device_number, use_cuda)

    def open_xml_file(self, file_name):

        tree = ET.parse(file_name)
        root = tree.getroot()

        return root, tree

    def semeval_sent_sense_collect(self, xml_struct):

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

                    _senses.extend([_temp_dict['wn30_key']] * len([words]))

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

            _e1 = _encoded_layers[-4:]

            # _e2 = torch.cat((_e1[0], _e1[1], _e1[2], _e1[3]), 2)
            _e2 = _e1[0]

            if self.use_cuda:
                _final_layer = _e2[0].cpu().numpy()

            else:
                _final_layer = _e2[0].numpy()
                _final_layer = np.around(_final_layer, decimals=5)  # LOWER PRECISION, process faster. CHECK if good!!

        return _final_layer

    def load_embeddings(self, pickle_file_name, corpus_file):
        """
        First pass on the corpus sentences. If pickle file is present, load data; else, calculate it.
        This method:
          a) Stores sentences as an array.
          b) Creates dictionary where each vocabulary word is mapped to its occurrences in corpus.
          c) Calculates embeddings for each vocabulary word.
        :param pickle_file_name:
        :param corpus_file:
        :return: sentences, vocab_map, embeddings
        """
        try:

            with open(pickle_file_name, 'rb') as h:
                _x, _y, _z = pickle.load(h)

                print("EMBEDDINGS FOUND!")
                return _x, _y, _z

        except:

            print("Embedding File Not Found!! \n")
            print("Performing first pass...")

            _x, _y, _z = self.calculate_embeddings(corpus_file=corpus_file)

            with open(pickle_file_name, 'wb') as h:
                pickle.dump((_x, _y, _z), h)

            print("Data stored in " + pickle_file_name)

            return _x, _y, _z

    def calculate_embeddings(self, corpus_file):
        """
        Calculates embeddings for all words in corpus_file, creates vocabulary dictionary
        :param corpus_file:     file to obtain vocabulary from
        :return:                sentences, vocab_map, embeddings
        """

        _test_root, _test_tree = self.open_xml_file(corpus_file)

        embeddings_count = 0
        all_embeddings = []  # Store each sent_embeddings as elements
        _sentences = []  # Store textual sentences
        _vocab_map = {}  # Store occurrences of each word in corpus

        # Process each sentence in corpus
        for sent_nbr, i in tqdm(enumerate(_test_root.iter('sentence'))):
            sent_embeddings = []  # Store one sentence's word embeddings as elements

            sent, sent1, senses = self.semeval_sent_sense_collect(i)
            _sentences.append(sent1)

            bert_tokens = self.collect_bert_tokens(sent)

            final_layer = self.get_bert_embeddings(bert_tokens)

            token_count = 1

            # Process all words in sentence
            for word_pos, j in enumerate(zip(senses, sent)):
                word = j[1]
                # Register word location in vocabulary dictionary
                if word not in _vocab_map.keys():
                    _vocab_map[word] = []
                _vocab_map[word].append((sent_nbr, word_pos))

                embedding = np.mean(final_layer[token_count:token_count + len(self.apply_bert_tokenizer(word))], 0)
                sent_embeddings.append(np.float32(embedding))  # Lower precision for speed
                token_count += len(self.apply_bert_tokenizer(word))

                embeddings_count += 1

            # Store this sentence embeddings in the general list
            all_embeddings.append(sent_embeddings)

        print(f"{embeddings_count} EMBEDDINGS GENERATED")

        return _sentences, _vocab_map, all_embeddings

    @staticmethod
    def disambiguate(_sentences, _vocab_map, _embeddings, save_dir, freq_threshold=5, **kwargs):
        """
        Disambiguate word senses through clustering their transformer embeddings
        Clustering is done using the selected sklearn algorithm.

        :param _sentences:      List of corpus sentences
        :param _vocab_map:      Dictionary that stores coordinates of every occurrence of each word
        :param _embeddings:     Embeddings for all words in corpus
        :param save_dir:        Directory to save disambiguated senses
        :param freq_threshold:  Frequency threshold for a word to be disambiguated
        :param kwargs:          Clustering parameters
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Init clustering object
        k = kwargs.get('k', 10)  # 10 is default value, if no kwargs were passed
        freq_threshold = max(freq_threshold, k)
        # estimator = KMeans(init="k-means++", n_clusters=k, n_jobs=4)
        # estimator = OPTICS(min_samples=3, cluster_method='dbscan', metric='cosine', max_eps=0.1, eps=0.1)
        estimator = DBSCAN(metric='cosine', n_jobs=4, min_samples=5, eps=0.5)

        fl = open(save_dir + "/clustering.log", 'w')  # Logging file
        fl.write(f"# WORD\t\tCLUSTERS\n")

        # Loop for each word in vocabulary
        for word, instances in _vocab_map.items():
            if len(instances) < freq_threshold:  # Don't disambiguate if word is uncommon
                continue

            print(f'Disambiguating word \"{word}\"...')
            curr_embeddings = []
            # Build embeddings list for this word
            for instance in instances:
                x, y = instance  # Get current word instance coordinates
                curr_embeddings.append(_embeddings[x][y])

            estimator.fit(curr_embeddings)  # Disambiguate
            num_clusters = max(estimator.labels_) + 1
            print(f"Num clusters: {num_clusters}")
            fl.write(f"{word}\t\t{num_clusters}\n")

            # If disambiguated, write senses to file, with some sentence examples
            if num_clusters > 1:
                with open(save_dir + '/' + word + ".disamb", "w") as fo:
                    for i in range(num_clusters):
                        fo.write(f"Cluster #{i}:\n[")
                        sense_members = [instances[j] for j, k in enumerate(estimator.labels_) if k == i]
                        np.savetxt(fo, sense_members, fmt="(%s, %s)", newline=", ")
                        fo.write(']\n')
                        # Write at most 3 sentence examples for the word sense
                        sent_samples = rand.sample(sense_members, min(len(sense_members), 3))
                        fo.write('Samples:\n')
                        # Write sample sentences to file, with focus word in CAPS
                        for sample, focus_word in sent_samples:
                            bold_sent = _sentences[sample].split()
                            bold_sent[focus_word] = bold_sent[focus_word].upper()
                            fo.write(" ".join(bold_sent) + '\n')
        fl.write("\n")
        fl.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WSD using BERT')

    parser.add_argument('--no_cuda', action='store_false', help='Use GPU?')
    parser.add_argument('--device', type=str, default='cuda:2', help='GPU Device to Use?')
    parser.add_argument('--corpus', type=str, required=True, help='Training Corpus')
    parser.add_argument('--start_k', type=int, default=10, help='First number of clusters to use')
    parser.add_argument('--end_k', type=int, default=10, help='Final number of clusters to use')
    parser.add_argument('--step_k', type=int, default=5, help='Increase in number of clusters to use')
    parser.add_argument('--save_to', type=str, help='Directory to save disambiguated words')
    parser.add_argument('--pickle_file', type=str, help='Pickle file of Bert Embeddings/Save Embeddings to file')
    parser.add_argument('--pretrained', type=str, default='bert-large-uncased', help='Pretrained model to use')
    parser.add_argument('--use_euclidean', type=int, default=0, help='Use Euclidean Distance to Find NNs?')

    args = parser.parse_args()

    print("Corpus is: " + args.corpus)
    # print("Number of clusters: " + str(args.end_k))

    if args.no_cuda:
        print("Processing with CUDA!")

    else:
        print("Processing without CUDA!")

    if args.use_euclidean:
        print("Using Euclidean Distance!")

    else:
        print("Using Cosine Similarity!")

    print("Loading WSD Model!")

    WSD = WordSenseModel(args.pretrained, device_number=args.device, use_cuda=args.no_cuda)

    print("Loaded WSD Model!")

    sentences, vocab_map, embeddings = WSD.load_embeddings(args.pickle_file, args.corpus)

    print("Start disambiguation...")
    for nn in range(args.start_k, args.end_k + 1, args.step_k):
        WSD.disambiguate(sentences, vocab_map, embeddings, args.save_to, freq_threshold=5, k=nn)
