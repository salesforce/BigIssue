import math
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import reduce
from transformers import RobertaModel, AutoConfig


class ModelWrapper(nn.Module):

    def __init__(self,
                 base_model,
                 config,
                 max_seq_length=512):

        super().__init__()
        self.base_model = base_model

        self._encoding_size = self.base_model.config.hidden_size

        positional_embeddings = torch.zeros(max_seq_length, self._encoding_size)

        for position in range(max_seq_length):
            for i in range(0, self._encoding_size, 2):
                positional_embeddings[position, i] = (
                    math.sin(position / (10000 ** ((2 * i) / self._encoding_size)))
                )
                positional_embeddings[position, i + 1] = (
                    math.cos(position / (10000 ** ((2 * (i + 1)) / self._encoding_size)))
                )

        positional_embeddings = positional_embeddings.unsqueeze(0)

        self.positional_embeddings = positional_embeddings

        self.classification_layer = Classification(config)

    def forward(self, input_ids, attention_mask, chunk_len=512):
        x = input_ids
        xs = torch.split(x, chunk_len, dim=1)
        attention_masks = torch.split(attention_mask, chunk_len, dim=1)
        xs2 = []
        for y, attention_mask in zip(xs, attention_masks):
            output = self.base_model(y, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
            xs2.append(output)

        x = torch.cat(xs2, dim=1)

        positional_embeddings = torch.tile(self.positional_embeddings, (x.shape[0], 1, 1)).to(x.device)
        x = x + positional_embeddings

        x = self.classification_layer(x)

        return x


class Classification(nn.Module):
    def __init__(self, config, classifier_dropout=None, hidden_dropout_prob=0.1):
        super().__init__()
        self.segment_len = config.segment_len
        hidden_size = config.hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            classifier_dropout if classifier_dropout is not None else hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.token_proj = nn.Linear(hidden_size, 1)

    def forward(self, features):
        x = reduce(features, 'b (l l2) d -> b l d', 'mean', l2=self.segment_len)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.token_proj(x)
        x = torch.squeeze(x, 2)
        return x


device = "cuda:0"

config = AutoConfig.from_pretrained("microsoft/codebert-base")
config.segment_len = 1
base_model = RobertaModel.from_pretrained("microsoft/codebert-base", add_pooling_layer=False)
model = ModelWrapper(base_model=base_model, config=config, max_seq_length=512)
state_dict = torch.load("/tmp/model")  # sample model file
new_state_dict = OrderedDict({k[k.index('.') + 1:]: v for k, v in state_dict.items()})
model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)
model.eval()
