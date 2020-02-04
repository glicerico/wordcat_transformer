import numpy as np
import torch
from transformers import BertTokenizer, BertModel

BOS_TOKEN = '[CLS]'
EOS_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'


class BERT:
    def __init__(self, pretrained_model, device_number='cuda:2', use_cuda=False):
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        self.model = BertModel.from_pretrained(pretrained_model)
        with torch.no_grad():
            self.model.eval()

        if use_cuda:
            self.model.to(device_number)

    def print_top_predictions(self, probs, k=5):
        probs = probs.detach().numpy()
        top_indexes = np.argpartition(probs, -k)[-k:]
        sorted_indexes = top_indexes[np.argsort(-probs[top_indexes])]
        top_tokens = self.tokenizer.convert_ids_to_tokens(sorted_indexes)
        print(f"Ordered top predicted tokens: {top_tokens}")
        print(f"Ordered top predicted values: {probs[sorted_indexes]}")

    def get_sentence_prob(self, sentence, word, verbose=False):

        """
        Estimate the sentence probability if word is placed in sentence, in the masked location.
        Let S_i be a sentence with an empty slot, e.g. 'My racist ___ called me last night.', then this function
        takes the input 'word', places it in the empty slot, and returns the sentence probability P(S_i|___=word).
        This probability is composed by using the given transformer model's predictions, as follows:
        P(S_i) = Prod_i(P(w_i|w_0, w_1,..,w_i-1,w_i+1,..,w_N)),
        where N is the number of words in the sentence, and each P(w_i|...) is given by a transformer evaluation.

        :return:
        """
        sm = torch.nn.Softmax(dim=0)  # used to convert last hidden state to probs

        # Pre-process sentence, adding special tokens
        tokenized_input = self.tokenizer.tokenize(sentence)
        sent_len = len(tokenized_input)
        if tokenized_input[0] != BOS_TOKEN:
            tokenized_input.insert(0, BOS_TOKEN)
        if tokenized_input[-1] != EOS_TOKEN:
            tokenized_input.append(EOS_TOKEN)
        ids_input = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        print(f"Processing sentence: {tokenized_input}")
        # print(f"Sentence ids: {ids_input}")

        # sent_prob = 1
        sum_lp = 0
        # Mask non-special tokens and calculate their probabilities
        for i in range(1, len(tokenized_input) - 1):  # Ignore first and last tokens
            current_tokenized = tokenized_input[:]
            current_tokenized[i] = MASK_TOKEN
            if verbose:
                print(current_tokenized)
            masked_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(current_tokenized)])
            predictions = self.model(masked_input)[0]
            current_probs = sm(predictions[0, i])  # Softmax to get probabilities
            current_prob = current_probs[ids_input[i]]  # Prediction for masked word

            # sent_prob *= current_prob
            sum_lp += np.log(current_prob.detach().numpy())

            print(f"Word: {tokenized_input[i]} \t Prob: {current_prob}")
            if verbose:
                self.print_top_predictions(current_probs)

        # print(f"\nSentence probability: {sent_prob.item()}\n")
        print(f"\nNormalized sentence prob: log(P(sentence)) / sent_length: {sum_lp / sent_len}\n")
        return sum_lp / sent_len
