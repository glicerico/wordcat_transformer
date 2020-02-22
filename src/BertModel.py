import torch
from transformers import BertTokenizer, BertModel


class BERT:
    def __init__(self, pretrained_model='bert-large-uncased', device_number='cuda:2', use_cuda=True,
                 output_hidden_states=True):
        self.device_number = device_number
        self.use_cuda = use_cuda

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertModel.from_pretrained(pretrained_model, output_hidden_states=output_hidden_states)
        with torch.no_grad():
            self.model.eval()

        if use_cuda:
            self.model.to(device_number)

