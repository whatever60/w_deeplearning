import torch
from torch import nn


class DeepCountAutoencoder(nn.Module):

    def __init__(self, in_dim, hid_dim, bottle_dim, out_dim, mode):
        super().__init__()
        assert mode in ('zinb', 'nb', 'poisson'), "Unknown mode"
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hid_dim, bottle_dim),
            nn.BatchNorm1d(bottle_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottle_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
        )
        self.mean = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            ClippedExp(min=1e-5, max=1e6)
        )
        if mode == 'nb' or mode == 'zinb':
            self.disp = nn.Sequential(
                nn.Linear(hid_dim, out_dim),
                ClippedSoftplus(min=1e-4, max=1e3),
            )
        if mode == 'zinb':
            self.dropout = nn.Sequential(
                nn.Linear(hid_dim, out_dim),
                nn.Sigmoid()
            )

    def encode(self, x):
        '''
        Input: expression profile
        Return: latent representation
        '''
        return self.bottleneck(self.encoder(x))

    def decode(self, x, size_factors):
        '''
        Input: latent representation
        Return: mean, dispersion, dropout
        '''
        x = self.decoder(x)
        mu = self.mean(x)
        size_factors_scaled = size_factors.unsqueeze(dim=1)
        mu_scaled = mu * size_factors_scaled

        if self.mode == 'poisson':
            return mu_scaled
        else:
            theta = self.disp(x)
            if self.mode == 'nb':
                return mu_scaled, theta
            elif self.mode == 'zinb':
                pi = self.dropout(x)
            return mu_scaled, theta, pi


def weight_init(m):
    if isinstance(m, (nn.Linear)):
        nn.init.xavier_uniform_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class ClippedExp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    def forward(self, x):
        return torch.clamp(torch.exp(x), self.min, self.max)


class ClippedSoftplus(nn.Module):
    def __init__(self, min, max, beta=1, threshold=20):
        super().__init__()
        self.min = min
        self.max = max
        self.model = nn.Softplus(beta=beta, threshold=threshold)
        
    def forward(self, x):
        return torch.clamp(self.model(x), min=self.min, max=self.max)


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        # hid_dim = 64
        # out_dim = in_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(),
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, acts, shallow=False):
        # hid_dim = 32, out_dim = 64
        super().__init__()
        self.shallow = shallow
        if not shallow:
            self.decoder = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.PReLU(),
            )
        else:
            hid_dim = in_dim
        self.decoder1 = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            acts[0]()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            acts[1]()
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            acts[2]()
        )
        self.apply(weight_init)
    
    def forward(self, x, size_factors=None):
        if self.shallow:
            x = self.decoder(x)
        retval1 = self.decoder1(x)
        retval2 = self.decoder2(x)
        retval3 = self.decoder3(x)
        if size_factors is not None:
            retval1 *= size_factors.unsqueeze(dim=1)
        return retval1, retval2, retval3


class ChromEncoder(nn.Module):
    def __init__(self, in_dims, hid_dim, out_dim):
        # 32, 32
        super().__init__()
        layers = nn.ModuleList()
        for in_dim in in_dims:
            layers.append(Encoder(in_dim, hid_dim, hid_dim // 2))
        self.layers = layers
        self.encoder = nn.Sequential(
            nn.Linear(hid_dim // 2 * len(layers), out_dim),
            nn.BatchNorm1d(out_dim),
            nn.PReLU()
        )
        self.apply(weight_init)

    def forward(self, x):
        assert len(x) == len(self.layers)
        enc_chroms = torch.cat([layer(x0) for layer, x0 in zip(self.layers, x)], dim=1)
        return self.encoder(enc_chroms)


class ChromDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims, acts):
        # in_dim = 32, hid_dim = 16
        # acts = (ClippedExp, ClippedSoftplus, nn.Identity)
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, len(out_dims) * hid_dim),
            nn.BatchNorm1d(len(out_dims) * hid_dim),
            nn.PReLU()
        )
        self.layers = nn.ModuleList([Decoder(hid_dim, hid_dim * 2, out_dim, acts=acts) for out_dim in out_dims])
        self.apply(weight_init)

    def forward(self, x):
        x = self.decoder(x)
        x_chunked = torch.chunk(x, chunks=len(self.layers), dim=1)
        retvals = [layer(x) for layer, x in zip(self.layers, x_chunked)]
        return [torch.cat(retval, dim=1) for retval in list(zip(*retvals))]
        

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, acts):
        super().__init__()
        self.encoder = Encoder(in_dim, 64, hid_dim)
        self.decoder = Decoder(hid_dim, 64, out_dim, acts=acts)

    def forward(self, x, size_factors=None):
        encoded = self.encoder(x)
        mean, dispersion, dropout = self.decoder(encoded, size_factors=size_factors)[0]
        return encoded, mean, dispersion, dropout
    
    def decode(self, encoded, size_factors=None):
        mean, dispersion, dropout = self.decoder(encoded, size_factors=size_factors)[0]
        return mean, dispersion, dropout


class ChromAutoEncoder(nn.Module):
    """
    Autoencoders that maps per chromsome input to per chromsome output
    """
    def __init__(self, in_dims, hid_dim, mode, acts=(ClippedExp, nn.Softplus, nn.Sigmoid)):
        super().__init__()
        assert mode in ('possion', 'nb', 'zinb')
        self.mode = mode
        self.encoder = ChromEncoder(in_dims, 32, hid_dim)
        self.decoder = ChromDecoder(hid_dim, 16, in_dims, acts=acts)
    
    def forward(self, x):
        encoded = self.encoder(x)
        mean, dispersion, dropout = self.decoder(encoded)
        if self.mode == 'possion':
            return encoded, mean
        elif self.mode == 'nb':
            return encoded, mean, dispersion
        elif self.mode == 'zinb':
            return encoded, mean, dispersion, dropout


class GenomeChromAutoEncoder(nn.Module):
    """RNA -> ATAC
    """
    def __init__(self, in_dim, hid_dim, out_dim, mode, acts=(ClippedExp, nn.Softplus, nn.Sigmoid)):
        super().__init__()
        assert mode in ('possion', 'nb', 'zinb')
        self.mode = mode
        if isinstance(in_dim, int):
            self.encoder = Encoder(in_dim, 64, hid_dim)
        else:
            self.encoder = ChromEncoder(hid_dim, 32, hid_dim)
        if isinstance(out_dim, int):
            self.decoder = Decoder(hid_dim, 32, out_dim)
        else:
            self.decoder = ChromDecoder(hid_dim, 16, out_dim, acts=acts)
    
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, *self.from_encoded(encoded)

    def from_encoded(self, encoded):
        mean, dispersion, dropout = self.decoder(encoded)
        if self.mode == 'possion':
            return encoded, mean
        elif self.mode == 'nb':
            return encoded, mean, dispersion
        elif self.mode == 'zinb':
            return encoded, mean, dispersion, dropout


class PairedAutoEncoder(nn.Module):
    def __init__(self, model_rna, model_atac):
        super().__init__()
        self.model_rna = model_rna
        self.model_atac = model_atac

    def forward(self, x):
        rna, atac = x
        rna_decoded = self.model_rna(rna)
        atac_decoded = self.model_atac(atac)
        return rna_decoded, atac_decoded

    def rna_to_atac(self, rna_encoded):
        return self.model_atac.from_encoded(rna_encoded)

    def atac_to_rna(self, atac_encoded):
        return self.model_rna.from_encoded(atac_encoded)


class InvertibleAutoEncoder(nn.Module):
    """
    Similar to paired autoencoder, but with an additional invertible network linking
    the latent dimensions
    An example invertible network might be RealNVP
    """
    
    def __init__(self, model_rna, model_atac, link_model):
        super().__init__()
        self.model_rna = model_rna
        self.model_atac = model_atac
        self.link_model = link_model

    def forward(self, x):
        x1, x2 = x  # Unpack, expects tuple from PairedDataset
        # Run through models
        y1 = self.model_rna(x1)
        y2 = self.model_atac(x2)
        # Run encoded representations through link model
        enc1 = y1[0]
        enc2 = y2[0]
        enc2_pred = self.link_model.forward(enc1, mode="direct")
        enc1_pred = self.link_model.forward(enc2, mode="inverse")
        return (y1, y2, (enc1_pred, enc2_pred))

    def rna_to_atac(self, rna_encoded):
        """Using invertible layer, translate domain 1 to domain 2"""
        atac_encoded = self.link_model.forward(rna_encoded, mode="direct")[0]
        return self.model_atac.from_encoded(atac_encoded)

    def atac_to_rna(self, atac_encoded):
        """Using invertbile layer, translate domain 2's latent representation to domain 1"""
        rna_encoded = self.link_model.forward(atac_encoded, mode="inverse")[0]
        return self.model_rna.from_encoded(rna_encoded)
    

class CatAutoEncoder(nn.Module):
    pass


class SplicedAutoEncoder(nn.Module):
    """
    Spliced Autoencoder - where we have 4 parts (2 encoders, 2 decoders) that are all combined
    This does not work when you have chromsome split features
    """
    def __init__(self, in_dim_rna, in_dim_atac, hid_dim, acts_rna=(ClippedExp, nn.Softplus, nn.Identity), acts_atac=(ClippedExp, nn.Softplus, nn.Sigmoid)):
        super().__init__()
        self.encoder_rna = Encoder(in_dim_rna, 64, hid_dim)
        self.encoder_atac = Encoder(in_dim_atac, 64, hid_dim)
        self.decoder_rna = Decoder(hid_dim, 32, in_dim_rna, acts=acts_rna)
        self.decoder_atac = Decoder(hid_dim, 32, in_dim_atac, acts=acts_atac)

    def split_catted_input(self, x):
        pass
    
    def _combine_output_and_encoded(self, encoded, decoded, num_outputs):
        pass

    def forward_single(self, x, size_factors, in_domian, out_domain):
        pass

    def forward(self, rna, atac):
        """
        RNA -> RNA
        ATAC -> RNA
        RNA -> ATAC
        ATAC -> ATAC
        """
        rna_encoded = self.encoder_rna(rna)
        atac_encoded = self.encoder_atac(atac)

        rna_from_rna = self.decoder_rna(atac_encoded)
        rna_from_atac = self.decoder_rna(rna_encoded)

        atac_from_rna = self.decoder_atac(rna_encoded)
        atac_from_atac = self.decoder_atac(atac_encoded)

        return rna_from_rna, rna_from_atac, atac_from_rna, atac_from_atac


class NaiveSpliceAutoEncoder(nn.Module):
    """
    Naive "spliced" autoencoder that does not use shared branches and instead simply
    trains four separate models
    """


class AssymSplicedAutoEncoder(nn.Module):
    """
    Assymmetric spliced autoencoder where branch 2 is a chrom AE
    """


class PerChromSplicedAutoEncoder(SplicedAutoEncoder):
    """
    Autoencoder that treats each chromsome completely separately
    Essentially a series of 22 independent autoencoders
    """


class AutoEncoderSkorchNet:
    pass


class PairedAutoEncoderSkorchNet:
    pass


class SplicedAutoEncoderSkorchNet(PairedAutoEncoderSkorchNet):
    pass


class PairedInvertibleAutoEncoderSkorchNet(PairedAutoEncoderSkorchNet):
    pass



class Loss(nn.Module):
    def __init__(self, y, prop_reg_lambda, criterion):
        super().__init__()
        self.prop_reg_lambda = prop_reg_lambda
        self.criterion = criterion

    def forward(self, y_pred, y_true, X=None):
        if len(y_true) == 2:
            y, y_cluster_pbulk = y_true
        else:
            y = y_true
        loss = self.criterion(y_pred, y)

        if self.prop_reg_lambda != 0.0:
            if isinstance(y_true, tuple) or isinstance(y_true, list):
                loss += self.prop_reg_lambda * self.get_prop_reg_pbulk(
                    y_pred[0], y_cluster_pbulk
                )
            else:
                loss += self.prop_reg_lambda * self.get_prop_reg(y_pred[0], y_true)
        
        return loss

    def get_prop_reg(self, y_pred, y_true):
        """
        Compute regularization based on the overall proportion of each gene in the batch
        """
        per_gene_counts = torch.sum(y_true, axis=0)  # Sum across the batch
        per_gene_counts_norm = per_gene_counts / torch.sum(per_gene_counts)

        per_gene_pred_counts = torch.sum(y_pred, axis=0)
        per_gene_pred_counts_norm = per_gene_pred_counts / torch.sum(
            per_gene_pred_counts
        )

        # L2 distance between the two
        # d = F.pairwise_distance(per_gene_pred_counts_norm, per_gene_counts_norm, p=2, eps=1e-6, keepdim=False)
        d2 = torch.pow(per_gene_counts_norm - per_gene_pred_counts_norm, 2).mean()
        # d = torch.sqrt(d2)
        return d2