import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

BOS_TOKEN = '[CLS]'
EOS_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'


class BertLM:
    def __init__(self, pretrained_model='bert-base-uncased', device_number='cuda:2', use_cuda=False):
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        self.model = BertForMaskedLM.from_pretrained(pretrained_model)
        with torch.no_grad():
            self.model.eval()

        if use_cuda:
            self.model.to(device_number)

    def print_top_predictions(self, probs, k=5):
        """
        Prints the top-k predicted words contained in probs, and their probabilities.
        :param probs:   A tensor containing the probability for each word in the vocabulary.
                        Probs is calculated via softmax from the logit tensor returned by
                        a masked prediction by the transformer.
        :param k:       Number of top predictions to print
        :return:        Nothing, only prints info
        """
        probs = probs.detach().numpy()
        top_indexes = np.argpartition(probs, -k)[-k:]
        sorted_indexes = top_indexes[np.argsort(-probs[top_indexes])]
        top_tokens = self.tokenizer.convert_ids_to_tokens(sorted_indexes)
        print(f"Ordered top predicted tokens: {top_tokens}")
        print(f"Ordered top predicted values: {probs[sorted_indexes]}")

    def get_sentence_prob(self, tokenized_input, verbose=False):

        """
        Estimate the sentence probability P(S), where S is a sentence.
        This probability is composed by using the given transformer model's predictions, as follows:
        P(S) = Prod_i(P(w_i|w_0, w_1,..,w_i-1,w_i+1,..,w_N)),
        where N is the number of words in the sentence, and each P(w_i|...) is given by a transformer masked
        word prediction.
        Hence, one sentence probability requires N masked word prediction evaluations.
        :param sentence: Input sentence
        :param verbose: Print information about the obtained probabilities or not.
        :return: Sentence probability normalized by sentence length
        """
        sm = torch.nn.Softmax(dim=0)  # used to convert last hidden state to probs

        # Pre-process sentence, adding special tokens
        # tokenized_input = self.tokenizer.tokenize(sentence)  ### Input comes pre-tokenized, bad design :(
        sent_len = len(tokenized_input)
        if tokenized_input[0] != BOS_TOKEN:
            tokenized_input.insert(0, BOS_TOKEN)
        if tokenized_input[-1] != EOS_TOKEN:
            tokenized_input.append(EOS_TOKEN)
        ids_input = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        if verbose:
            print(f"Processing sentence: {tokenized_input}")
            # print(f"Sentence ids: {ids_input}")

        # sent_prob = 1
        sum_lp = 0
        # Mask non-special tokens and calculate their probabilities
        for i in range(1, len(tokenized_input) - 1):  # Ignore first and last tokens
            current_tokenized = tokenized_input[:]
            current_tokenized[i] = MASK_TOKEN
            masked_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(current_tokenized)])
            predictions = self.model(masked_input)[0]
            current_probs = sm(predictions[0, i])  # Softmax to get probabilities
            current_prob = current_probs[ids_input[i]]  # Prediction for masked word

            # sent_prob *= current_prob
            sum_lp += np.log(current_prob.detach().numpy())

            if verbose:
                print(current_tokenized)
                print(f"Word: {tokenized_input[i]} \t Prob: {current_prob}")
                self.print_top_predictions(current_probs)

        # print(f"\nSentence probability: {sent_prob.item()}\n")
        if verbose:
            print(f"\nNormalized sentence prob: log(P(sentence)) / sent_length: {sum_lp / sent_len}\n")

        return sum_lp / sent_len
