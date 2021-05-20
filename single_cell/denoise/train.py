import numpy as np
from scipy import sparse as ss
import pandas as pd
import datatable as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class LinearBasicBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )
        self.final_act = nn.LeakyReLU(0.2, inplace=True)

        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        connection = self.downsample(x)
        out = self.model(x) + connection
        return self.final_act(out)


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(inplace=True)
            ShiftedReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class ShiftedReLU(nn.Module):
    def __init__(self, shift=0.5, inplace=True):
        super().__init__()
        self.model = nn.ReLU(inplace=inplace)
        self.shift = shift
    
    def forward(self, x):
        return self.model(x - self.shift)


class RectifiedTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Tanh()
    
    def forward(self, x):
        return self.model(x.clip(0))


class UNET(nn.Module):
    def __init__(self, in_dim, out_dim, features=(128, 64, 32), block=LinearBlock):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(block(in_dim, feature))
            in_dim = feature

        feature = in_dim // 2
        self.bottleneck = block(in_dim, feature)
        in_dim = feature

        for feature in features[::-1]:
            self.decoder.append(nn.Linear(in_dim, feature))
            self.decoder.append(block(feature * 2, feature))
            in_dim = feature
        # self.final_linear = nn.Linear(in_dim, out_dim)
        self.final_linear = nn.Sequential(nn.Linear(in_dim, out_dim), RectifiedTanh())
        # self.apply(weight_init)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        for i, skip_connection in enumerate(reversed(skip_connections)):
            x = self.decoder[i * 2](x)
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder[i * 2 + 1](x)
        return self.final_linear(x)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight)
        # nn.init.zeros_(m.weight)
        if m.bias is not None:
            # nn.init.eye_(m.bias)
            nn.init.zeros_(m.bias)
        # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
    # if isinstance(m, nn.BatchNorm2d):
    #     torch.nn.init.normal_(m.weight, 0.0, 0.02)
    #     torch.nn.init.constant_(m.bias, 0)


def test():
    device = "cuda:5"
    in_dim = 30000
    model = UNET(in_dim, in_dim)
    profiles = torch.randn((32, in_dim))
    model.to(device)
    profiles = profiles.to(device)
    preds = model(profiles)
    print(preds.shape)  # (32, in_dim)
    print(preds[:, 1])


class RNAProfile(Dataset):
    def __init__(self, X1, X2, mask_p) -> None:
        super().__init__()
        self.X1 = torch.from_numpy(X1)
        self.X2 = torch.from_numpy(X2)
        self.len = X1.shape[0]
        self.dim = X1.shape[1]
        self.mask_p = mask_p

    def __len__(self):
        return len(self.X1) * 2

    def __getitem__(self, index):
        if index < self.len:
            input_ = self.X1[index]
            target = self.X2[index]
        else:
            input_ = self.X2[self.len - index]
            target = self.X1[self.len - index]
        mask = torch.from_numpy(np.random.binomial(1, self.mask_p, self.dim)).bool()
        input_ = input_.masked_fill(mask, 0)
        return input_, target, mask


def gen_dataset(data_dir, binomial_p=0.85, mask_p=0, cache_dir="./data"):
    # It takes time to import this, so do lazy import.
    from pipeline import qc, svd_with_sparse
    df = dt.fread(data_dir, skip_to_line=2, header=False).to_pandas()
    gene_num, cell_num, total_count = df.iloc[0]
    df.columns = ["gene", "cell", "counts"]
    df.drop(index=0, inplace=True)
    df.cell -= 1
    df.gene -= 1
    X = ss.csr_matrix(
        (df.counts, (df.cell, df.gene)), shape=(cell_num, gene_num), dtype="int16"
    )
    X = qc(X, gene_min_cells=50)
    X1, X2 = pseudo_replicate(X, binomial_p)
    # median = np.median(X.sum(axis=1).A.flatten())
    # return X1, X2

    def normalize(X_sparse):
        # X = preprocessing.minmax_scale(X_sparse.log1p().toarray())
        # X = (X - 0.5) / 0.5
        # X_sparse = preprocessing.normalize(X_sparse.log1p(), norm="l1")
        X_sparse = (preprocessing.normalize(X_sparse, norm="l1") * 5000).log1p()  # l1 first
        X_sparse /= X_sparse.max()
        X = X_sparse.toarray()
        # X, _ = svd_with_sparse(X_sparse, 2000)
        # X = (X - X.min()) / (X.max() - X.min())
        # X = (X - 0.5) / 0.5
        return X.astype("float32")
        # return preprocessing.StandardScaler().fit_transform(X)

    X1, X2 = normalize(X1), normalize(X2)
    X1_train, X1_val, X2_train, X2_val = train_test_split(X1, X2, test_size=0.2)
    np.save(f"{cache_dir}/x1_train.npy", X1_train)
    np.save(f"{cache_dir}/x1_val.npy", X1_val)
    np.save(f"{cache_dir}/x2_train.npy", X2_train)
    np.save(f"{cache_dir}/x2_val.npy", X2_val)
    train_dataset = RNAProfile(X1_train, X2_train, mask_p)
    val_dataset = RNAProfile(X1_val, X2_val, mask_p)
    return train_dataset, val_dataset


def pseudo_replicate(X_sparse, p):
    replicate_data = np.array([np.random.binomial(i, p, 2) for i in X_sparse.data])
    replicate1 = X_sparse.copy()
    replicate1.data = replicate_data[:, 0]
    replicate2 = X_sparse.copy()
    replicate2.data = replicate_data[:, 1]
    return replicate1, replicate2


class RNADataLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        binomial_p: float,
        mask_p: float,
        data_dir: str,
        cache_dir: str = "./data",
        use_cache: bool =True,
    ):
        """
        Args:
            batch_size: Batch size.
            binomial_p: p of the binomial sampling, so that each count in the original
                profile has a probability p to be included in the generated pesudo profile.
            mask_p: Probability of applying mask on each gene.
            data_dir: Path to your `.mtx` profile file. Ignored when `use_cache` is True.
            cache_dir: The directory of cached ready-to-use data. If `use_cache` is True,
                files in `cache_dir` will be directly used for training. When `use_cache`
                is False, processed data will be stored in `cache_dir`.
            use_cache: If True, skip data processing and use cache. If False, process raw
                data and cache it.
        """
        super().__init__()
        self.batch_size = batch_size
        self.binomial_p = binomial_p
        self.mask_p = mask_p
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache

    def prepare_data(self, stage=None):
        if not self.use_cache:
            self.train_dataset, self.val_dataset = gen_dataset(
                self.data_dir, self.binomial_p, self.mask_p, self.cache_dir
            )
        else:
            X1_train = np.load(f"{self.cache_dir}/x1_train.npy")
            X1_val = np.load(f"{self.cache_dir}/x1_val.npy")
            X2_train = np.load(f"{self.cache_dir}/x2_train.npy")
            X2_val = np.load(f"{self.cache_dir}/x2_val.npy")
            self.train_dataset = RNAProfile(X1_train, X2_train, self.mask_p)
            self.val_dataset = RNAProfile(X1_val, X2_val, self.mask_p)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )


class Net(pl.LightningModule):
    def __init__(
        self, batch_size, lr, ssl_weight, in_dim, features, binomial_p, mask_p, sparse_thres
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNET(in_dim, in_dim, features)
        self.criterion_recon = nn.MSELoss()
        self.criterion_ssl = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        input_, target, mask = batch
        pred = self(input_)
        loss_reconstruction = self.criterion_recon(target, pred)
        # loss_ssl = self.criterion_ssl(
        #     pred.masked_fill(~mask, 0), target.masked_fill(~mask, 0)
        # )
        # loss = loss_reconstruction + self.hparams.ssl_weight * loss_ssl
        loss = loss_reconstruction
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.shared_step(batch)
        self.log("trian_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.shared_step(batch)
        self.log_dict(dict(val_loss=loss, sparsity=(pred < self.hparams.sparse_thres).float().mean()))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        # return optimizer
        return dict(
            optimizer=optimizer,
            lr_scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5),
            interval="epoch",
        )


if __name__ == "__main__":
    # version 14
    # version 15: L1Loss -> MSELoss
    # version 16: Normalizer: MINMAX -> l1 norm
    # version 17: MSELoss -> L1Loss, as a proof of concept
    # version 18: Continue training of version 17
    # version 19: Learning rate: 2e-4 -> 1e-3
    # version 20: StepLR (20, 0.5)
    # version 21: Scale after normalization
    # version 22: Continue training of version 21
    # version 23: Add Tanh as the last activation
    # version 24: Continue training of version 23
    # version 25: L1Loss -> MSELoss
    # version 26: Mutual prediction dataset
    # version 27: add qc
    # version 28: raw count to PCA

    # version 30: New init, replace LinearBasicBlock, simpler preprocess
    # version 31: Forgot to init bias in version 30, higher batch size (128)
    # version 32: Remove (x - 0.5) / 0.5 shift, replace tanh with sigmoid
    # version 33: Batch size back to 64, learning rate doubled (2e-3)
    # version 34: Continue training of version 33.
    # version 35: Change MSELoss to L1Loss
    # version 36: Add sparsity monitor
    # version 37: Remove sigmoid
    # version 38: Increase learning rate to 1e-2
    # version 39: Learning rate back to 2e-3, add ReLU as the final activation
    # version 40: Replace Adam with SGD
    # version 41: Huge learning rate (1)
    # version 42: Learning rate 10
    # version 43: Learning rate 1e-2, loss multiplied by 1e4
    # version 44: Remove init.
    # version 45: Adam instead of SGD, ReLU instead of LeakyReLU
    # version 46: Profile multiplied by 10 after normalization
    # version 47: Uniform init.
    # version 48: L1 first, smaller model.
    # version 49: QC, learning rate 1e-3, replace ReLU with ELU
    # version 50: Replace ELU with ReLU
    # version 51: Add shift back, replace final ReLU with Tanh
    # version 52: Remove shift, activation ShiftedReLU, final ReLU
    # version 53: Add exp when computing loss, replace L1Loss with MSELoss
    # Something is wrong with version number at this point.
    # version 54: Final RectifiedTanh
    pl.seed_everything(42)

    DATA_DIR = "/home/tiankang/wusuowei/data/single_cell/babel/snareseq_GSE126074/GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz"
    CACHD_DIR = "./data2"
    BATCH_SIZE = 64
    LR = 1e-3
    SSL_WEIGHT = 0
    IN_DIM = 13183
    FEATURES = [128, 32]
    BINOMIAL_P = 0.85
    MASKED_P = 0
    SPARSE_THRES = 0.05

    dataloader = RNADataLoader(
        BATCH_SIZE,
        BINOMIAL_P,
        MASKED_P,
        DATA_DIR,
        CACHD_DIR,
        use_cache=True,
    )

    CHECKPOINT = "" # "/home/tiankang/wusuowei/deeplearning/single_cell/babel/denoise/lightning_logs/version_52/checkpoints/epoch=29-step=7739.ckpt"
    TRAIN = True
    model = Net(
        BATCH_SIZE,
        LR,
        SSL_WEIGHT,
        IN_DIM,
        FEATURES,
        BINOMIAL_P,
        MASKED_P,
        SPARSE_THRES
    )
    if not CHECKPOINT:
        trainer = pl.Trainer(
            max_epochs=100,
            gpus=[5],
            # precision=16,
            deterministic=True,
        )
    else:
        trainer = pl.Trainer(
            resume_from_checkpoint=CHECKPOINT,
            max_epochs=200,
            gpus=[9],
            # precision=16,
            deterministic=True,
        )
    if TRAIN:
        trainer.fit(model, dataloader)
    else:
        trainer.test(model, dataloader)
