import pdb
import lightning as L
import torch
import torch.nn as nn

import dataset

from model import encoder, decoder
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


class DDRSA(L.LightningModule):
    def __init__(self, feature_size, learning_rate):
        super(DDRSA, self).__init__()
        self.encoder = encoder(feature_size)
        self.decoder = decoder()
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def DRSA_loss(self, probs, ys, p=True):
        # pdb.set_trace()
        loss = torch.tensor(0.0, requires_grad=True)
        trade_off_factor = 0.75
        batch_size = len(ys)
        # if p == False:
        #     pdb.set_trace()
        for i in range(0, batch_size):
            y = ys[i]
            prob_seq = probs[:, i, 0]  # batch at second index

            # find the index of one
            one_idx = []
            for i in range(0, len(y)):
                if y[i] == 1.0:
                    one_idx.append(i)

            if len(one_idx) == 0:
                # the failure is censored
                log_one_minus_h = torch.log(1.0 - prob_seq)
                log_l_c = torch.sum(log_one_minus_h)
                loss = loss + (trade_off_factor - 1.0) * log_l_c
                self.log("log_l_c", log_l_c)
            elif len(one_idx) == 1:
                # the failure is uncensored
                if p == False:
                    # pass
                    print(prob_seq)
                    print(y)
                l = one_idx[0]
                log_one_minus_h = torch.log(1.0 - prob_seq)
                log_l_z = torch.sum(log_one_minus_h[0:l]) + torch.log(prob_seq[l])
                log_l_u = torch.log(1 - torch.prod(1 - prob_seq).double()).float()
                loss = loss + (
                    -1 * trade_off_factor * log_l_z + (trade_off_factor - 1) * log_l_u
                )
                self.log("log_l_z", log_l_z)
                self.log("log_l_u", log_l_u)
            else:
                assert False

        # return the averaged loss over the batch
        # pdb.set_trace()
        return loss / batch_size
        # return loss

    def training_step(self, batch):
        # print("training step is called")
        xs, lengths, ys = batch
        # print(f"xs shape {xs.shape}")

        out = self.encoder((xs, lengths))
        print("hidden state ", out)
        # pdb.set_trace()
        probs = self.decoder(out)  # shape should be (512, 64)

        assert probs.shape[1] == len(ys)

        loss = self.DRSA_loss(probs, ys, False)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        # print("validation step is called")
        xs, lengths, ys = batch

        out = self.encoder((xs, lengths))
        probs = self.decoder(out)  # shape should be (512, 64)

        # pdb.set_trace()
        # need second dim of probs since batch_first=False
        assert probs.shape[1] == len(ys)

        loss = self.DRSA_loss(probs, ys, p=False)
        self.log("val_loss", loss)
        return loss


def train(model, train_loader, valid_loader):
    trainer = L.Trainer(
        limit_train_batches=2,
        # callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        accelerator="cpu",
        log_every_n_steps=1,
        overfit_batches=1,
        max_epochs=5000,
    )
    # the train should automatically use gpu for training when available
    trainer.fit(model, train_loader)


def pad_sequence(data):
    # pdb.set_trace()
    xs, lengths, ys = zip(*data)
    xs = nn.utils.rnn.pad_sequence(xs)
    return xs, lengths, ys


if __name__ == "__main__":
    print("#### start process the data ####")
    train_dataset, valid_dataset, test_dataset = dataset.process_data(
        "../AMLWorkshop/Data/features_15h.csv"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        collate_fn=pad_sequence,
        num_workers=2,
    )
    # train_loader = DataLoader(
        # dataset.oneDataSet(), batch_size=2, collate_fn=pad_sequence
    # )
    pdb.set_trace()
    valid_loader = DataLoader(
        valid_dataset, batch_size=16, collate_fn=pad_sequence, num_workers=2
    )
    print("#### finish process the data ####")

    print("#### start build the model ####")
    model = DDRSA(feature_size=16, learning_rate=0.01)
    print("#### finish build the model ####")

    print("#### start training ####")
    train(model, train_loader, valid_loader)
    print("#### finish training ####")
