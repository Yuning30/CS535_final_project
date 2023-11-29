import lightning as L
import torch

from model import encoder, decoder

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