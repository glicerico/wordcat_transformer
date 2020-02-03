import numpy as np
import torch

def sentence_prob(tokenizer, model, sentence, word):
    """
    Estimate the sentence probability if word is placed in sentence, in the masked location.
    Let S_i be a sentence with an empty slot, e.g. 'My racist ___ called me last night.', then this function
    takes the input 'word', places it in the empty slot, and returns the sentence probability P(S_i|___=word).
    This probability is calculated using the given transformer 'model', as follows:
    P(S_i) = Prod()


    :return:
    """
# Load pre-trained model (weights)
with torch.no_grad():
    model = BertForMaskedLM.from_pretrained('bert-large-uncased')
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')