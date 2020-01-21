import xml.etree.ElementTree as ET
import torch
import argparse
import numpy as np

from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, DBSCAN

from tqdm import tqdm, trange
import warnings

warnings.filterwarnings('ignore')


class BERT:

    def __init__(self, device_number='cuda:2', use_cuda=True):
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        self.model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
        self.model.eval()

        if use_cuda:
            self.model.to(device_number)


class WordSenseModel:

    def __init__(self, device_number='cuda:2', use_cuda=True):

        self.device_number = device_number
        self.use_cuda = use_cuda

        self.Bert_Model = BERT(device_number, use_cuda)

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

            _e2 = torch.cat((_e1[0], _e1[1], _e1[2], _e1[3]), 2)

            if self.use_cuda:
                _final_layer = _e2[0].cpu().numpy()

            else:
                _final_layer = _e2[0].numpy()
                _final_layer = np.around(_final_layer, decimals=4)  # LOWER PRECISION, process faster. CHECK if good!!

        return _final_layer

    def get_embeddings(self,
                       corpus_file):
        """
        Finds BERT data for all words in train_file, and writes them to file
        :param corpus_file:     file to obtain vocabulary from
        :return: data:    data for the words in corpus_file
        """

        _test_root, _test_tree = self.open_xml_file(corpus_file)

        embeddings_count = 0
        all_embeddings = []
        words = []

        for i in tqdm(_test_root.iter('sentence')):

            sent, sent1, senses = self.semeval_sent_sense_collect(i)

            bert_tokens = self.collect_bert_tokens(sent)

            final_layer = self.get_bert_embeddings(bert_tokens)

            token_count = 1

            for idx, j in enumerate(zip(senses, sent)):
                word = j[1]
                embedding = np.mean(final_layer[token_count:token_count + len(self.apply_bert_tokenizer(word))], 0)
                all_embeddings.append(embedding)
                token_count += len(self.apply_bert_tokenizer(word))

                words.append(word)
                embeddings_count += 1

        print(f"{embeddings_count} EMBEDDINGS GENERATED")

        return all_embeddings, words

    def cluster_embeddings(self, data, words, cluster_file, *kwargs):
        """
        Cluster the data vectors using an sklearn algorithm, and write clusters to file

        :param data:            Embeddings to cluster
        :param words:           Words corresponding to each data vector
        :param cluster_file:    File to write the clusters
        :param kwargs:
        """
        # estimator = KMeans(init="k-means++", n_clusters=20, n_jobs=4)
        estimator = OPTICS(min_samples=3, cluster_method='dbscan', metric='cosine', max_eps=0.3, eps=0.3)
        #estimator = DBSCAN(metric='cosine', n_jobs=4, min_samples=5, eps=0.3)
        estimator.fit(data)
        print(estimator.labels_)
        num_clusters = max(estimator.labels_)

        with open(cluster_file, "w") as fo:
            words = np.array(words)
            for i in range(num_clusters):
                print(f"Cluster #{i}:")
                fo.write(f"Cluster #{i}:\n[")
                # print(estimator.labels_==i)
                category = words[estimator.labels_ == i]
                print(category)
                category.tofile(fo, sep=", ")
                fo.write(']\n')
            print("Finished clustering")

        print(f"Num clusters: {num_clusters}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WSD using BERT')

    parser.add_argument('--no_cuda', action='store_false', help='Use GPU?')
    parser.add_argument('--device', type=str, default='cuda:2', help='GPU Device to Use?')
    parser.add_argument('--corpus', type=str, required=True, help='Training Corpus')
    parser.add_argument('--start_k', type=int, default=1, help='First number of clusters to use')
    parser.add_argument('--end_k', type=int, default=1, help='Final number of clusters to use')
    parser.add_argument('--embeddings_file', type=str, help='Where to save the data')
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

    WSD = WordSenseModel(device_number=args.device, use_cuda=args.no_cuda)

    print("Loaded WSD Model!")

    for nn in range(args.start_k, args.end_k + 1):
        embeddings, labels = WSD.get_embeddings(corpus_file=args.corpus)
        WSD.cluster_embeddings(embeddings, labels, args.embeddings_file)
