import lightning as L
import torch

import dataset

from model import encoder, decoder
from torch.utils.data import DataLoader


class DDRSA(L.LightningModule):
    def __init__(self, feature_size, learning_rate):
        super(DDRSA, self).__init__()
        self.encoder = encoder(feature_size)
        self.decoder = decoder()
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        pass


def train(model, train_loader, valid_loader):
    trainer = L.Trainer()
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main":
    train_dataset, valid_dataset, test_dataset = dataset.process_data(
        "../AMLWorkshop/Data/features_15h.csv"
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True)

    model = DDRSA(feature_size=16, learning_rate=0.01)

    train(model, train_loader, valid_loader)
