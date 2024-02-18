"""
This files contains lighting AI module for training
"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Bert
from transformers import BertModel, BertTokenizer

from graph import Graph


class GraphEncoder(nn.Module):
    def __init__(self):
        super(GraphEncoder, self).__init__()

    def forward(self, x):
        pass


class GraphDecoder:
    """
    TODO: See if you can load a pretrained language decoder
    """

    def __init__(self):
        super(GraphDecoder, self).__init__()

    def forward(self, x):
        pass


class LitModel(L.LightningModule):
    def __init__(self, model, graph: Graph):
        super().__init__()
        # CHECK: May load pretrained

        self.graph = graph

        self.encoder = BertModel()

        self.decoder = GraphDecoder()  # Perhaps load some pretrained dcoder here

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        graph_trans_embeds = self.encoder(x)

        # Travel the Graph
        center_position = self.graph.navigate(graph_trans_embeds)

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
