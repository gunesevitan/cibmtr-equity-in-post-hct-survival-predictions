import torch
import torch.nn as nn


def init_weights(layer):

    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight, gain=1.0)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    if isinstance(layer, nn.Embedding):
        nn.init.xavier_normal_(layer.weight, gain=1.0)


class FullyConnectedBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super(FullyConnectedBlock, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for fc in [self.fc1, self.fc2]:
            init_weights(fc)

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SparseMLP(nn.Module):

    def __init__(self, input_dim, stem_dim, mlp_hidden_dim, n_blocks, output_dim):

        super(SparseMLP, self).__init__()

        self.stem = nn.Linear(input_dim, stem_dim, bias=True)
        self.mlp = nn.Sequential(
            *[
                FullyConnectedBlock(
                    input_dim=stem_dim,
                    hidden_dim=mlp_hidden_dim,
                ) for _ in range(n_blocks)
            ]
        )
        self.head = nn.Linear(stem_dim, output_dim)

    def forward(self, x):

        x = self.stem(x)
        x = self.mlp(x)
        outputs = self.head(x)

        return outputs


class EmbeddingMLP(nn.Module):

    def __init__(self, input_n_categories, input_cont_dim, embedding_dim, stem_dim, mlp_hidden_dim, n_blocks, output_dim):

        super(EmbeddingMLP, self).__init__()

        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(n_category, embedding_dim) for n_category in input_n_categories
        ])
        for embedding in self.categorical_embeddings:
            init_weights(embedding)

        embedding_output_dim = len(input_n_categories) * embedding_dim

        self.cont_fc = nn.Linear(input_cont_dim, input_cont_dim * 2, bias=True)
        init_weights(self.cont_fc)

        total_input_dim = embedding_output_dim + (input_cont_dim * 2)
        self.stem = nn.Linear(total_input_dim, stem_dim, bias=True)

        self.mlp = nn.Sequential(*[
            FullyConnectedBlock(input_dim=stem_dim, hidden_dim=mlp_hidden_dim)
            for _ in range(n_blocks)
        ])

        self.head = nn.Linear(stem_dim, output_dim)


    def forward(self, x):

        x_cat = x[:, :-2].long()
        x_cont = x[:, -2:]

        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.categorical_embeddings)]
        embedded = torch.cat(embedded, dim=1)

        cont_out = self.cont_fc(x_cont)
        x = torch.cat([embedded, cont_out], dim=1)

        x = self.stem(x)
        x = self.mlp(x)
        outputs = self.head(x)

        return outputs
