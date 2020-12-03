import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim, drop_rate=0.0, onlyEn=False):
        super(AutoEncoder, self).__init__()
        self.drop_rate = drop_rate
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.only_encoder = onlyEn
        # dummpy variable
        self.out_class_yns = None

        self.encoder_net = []
        in_size = in_dim
        for i, next_size in enumerate(hidden_dims):
            fc = nn.Linear(in_features=in_size, out_features=next_size)
            in_size = next_size
            self.__setattr__('enc{}'.format(i), fc)
            self.encoder_net.append(fc)

        self.latent_view = nn.Linear(in_features=in_size, out_features=latent_dim)

        if not self.only_encoder:
            rev_hidden = hidden_dims.copy()
            rev_hidden.reverse()
            self.decoder_net = []
            in_size = latent_dim
            for i, next_size in enumerate(rev_hidden):
                fc = nn.Linear(in_features=in_size, out_features=next_size)
                in_size = next_size
                self.__setattr__('dec{}'.format(i), fc)
                self.decoder_net.append(fc)
            self.output_view = nn.Linear(in_features=in_size, out_features=in_dim)

    def forward(self, X):
        for i, fc in enumerate(self.encoder_net):
            X = F.dropout(F.relu(fc(X)), p=self.drop_rate)

        latent = F.sigmoid(self.latent_view(X))

        if self.only_encoder:
            return [latent]
        else:
            X = latent
            for i, fc in enumerate(self.decoder_net):
                X = F.dropout(F.relu(fc(X)), p=self.drop_rate)
            output = F.sigmoid(self.output_view(X))
            return [output]
