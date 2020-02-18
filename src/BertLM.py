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

    def tokenize_sent(self, sentence):
        tokenized_input = self.tokenizer.tokenize(sentence)
        if tokenized_input[0] != BOS_TOKEN:
            tokenized_input.insert(0, BOS_TOKEN)
        if tokenized_input[-1] != EOS_TOKEN:
            tokenized_input.append(EOS_TOKEN)
        return tokenized_input

    def get_directional_prob(self, sm, tokenized_input, i, direction, verbose=False):
        current_tokens = tokenized_input[:]
        if direction == 'backwards':
            current_tokens[1:i + 1] = [MASK_TOKEN for j in range(i)]
        elif direction == 'forward':
            current_tokens[i:-1] = [MASK_TOKEN for j in range(len(tokenized_input) - 1 - i)]
        else:
            print("Direction can only be 'forward' or 'backwards'")
            exit()

        if verbose:
            print()
            print(current_tokens)

        masked_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(current_tokens)])
        predictions = self.model(masked_input)
        predictions = predictions[0]
        probs = sm(predictions[0, i])  # Softmax to get probabilities
        if verbose:
            self.print_top_predictions(probs)

        return probs  # Model predictions

    def get_sentence_prob_directional(self, tokenized_input, verbose=False):
        """
        Estimate the probability of sentence S: P(S).
        A forward one-directional sentence probability is defined as:
        P_f(S) = P(w_0, w_1, ..., w_N) = P(w_0) * P(w_1|w_0) * P(w_2|w_0, w_1) * ...
        where N is the number of words in the sentence, and each P(w_i|...) is given by a transformer masked
        word prediction with all words to its left masked.
        To take advantage of BERT's bi-directional capabilities, we also estimate the backwards probability:
        P_b(S) = P(w_0, w_1, ..., w_N) = P(w_N) * P(w_{N-1}|w_N) * P(w_{N-2}|w_{N-1}, w_N) * ...
        The sentence probability is the geometric-average of the two directional ones:
        P(S) = sqrt(P_f(S) * P_b(S))
        Hence, one sentence probability requires 2N masked word prediction evaluations.
        :param tokenized_input: Input sentence
        :param verbose: Print information about the obtained probabilities or not.
        :return: Log of geometric average of each prediction: sort of sentence prob. normalized by sentence length.
        """
        sm = torch.nn.Softmax(dim=0)  # used to convert last hidden state to probs

        # Pre-process sentence, adding special tokens
        sent_len = len(tokenized_input)
        ids_input = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        if verbose:
            print(f"Processing sentence: {tokenized_input}")

        sent_prob_forward = 1
        sent_prob_backwards = 1
        # Mask non-special tokens in forward and backwards directions; calculate their probabilities
        for i in range(1, len(tokenized_input) - 1):  # Don't loop first and last tokens
            probs_forward = self.get_directional_prob(sm, tokenized_input, i, 'forward', verbose=verbose)
            probs_backwards = self.get_directional_prob(sm, tokenized_input, i, 'backwards', verbose=verbose)
            prob_forward = probs_forward[ids_input[i]]  # Prediction for masked word
            prob_backwards = probs_backwards[ids_input[i]]  # Prediction for masked word
            sent_prob_forward *= np.power(prob_forward.detach().numpy(), 1 / sent_len)
            sent_prob_backwards *= np.power(prob_backwards.detach().numpy(), 1 / sent_len)

            if verbose:
                print(f"Word: {tokenized_input[i]} \t Prob_forward: {prob_forward}; Prob_backwards: {prob_backwards}")

        # Obtain geometric average of forward and backward probs
        geom_mean_sent_prob = np.sqrt(sent_prob_forward * sent_prob_backwards)
        if verbose:
            print(f"Geometric-mean forward sentence probability: {sent_prob_forward}")
            print(f"Geometric-mean backward sentence probability: {sent_prob_backwards}\n")
            print(f"Average normalized sentence prob: {geom_mean_sent_prob}\n")
        return geom_mean_sent_prob

    def get_sentence_prob(self, tokenized_input, verbose=False):

        """
        Estimate the sentence probability P(S), where S is a sentence.
        This probability is composed by using the given transformer model's predictions, as follows:
        P(S) = Prod_i(P(w_i|w_0, w_1,..,w_i-1,w_i+1,..,w_N)),
        where N is the number of words in the sentence, and each P(w_i|...) is given by a transformer masked
        word prediction.
        Hence, one sentence probability requires N masked word prediction evaluations.
        :param tokenized_input: Input sentence
        :param verbose: Print information about the obtained probabilities or not.
        :return: Log of geometric average of each prediction: sort of sentence prob. normalized by sentence length.
        """
        sm = torch.nn.Softmax(dim=0)  # used to convert last hidden state to probs

        # Pre-process sentence, adding special tokens
        sent_len = len(tokenized_input)
        ids_input = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        if verbose:
            print(f"Processing sentence: {tokenized_input}")

        sum_lp = 0
        # Mask non-special tokens and calculate their probabilities
        for i in range(1, len(tokenized_input) - 1):  # Ignore first and last tokens
            current_tokenized = tokenized_input[:]
            current_tokenized[i] = MASK_TOKEN
            masked_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(current_tokenized)])
            predictions = self.model(masked_input)[0]
            current_probs = sm(predictions[0, i])  # Softmax to get probabilities
            current_prob = current_probs[ids_input[i]]  # Prediction for masked word

            sum_lp += np.log(current_prob.detach().numpy())

            if verbose:
                print(current_tokenized)
                print(f"Word: {tokenized_input[i]} \t Prob: {current_prob}")
                self.print_top_predictions(current_probs)

        if verbose:
            print(f"\nNormalized sentence prob: log(P(sentence)) / sent_length: {sum_lp / sent_len}\n")

        return sum_lp / sent_len
