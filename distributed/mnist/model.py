import torch
import pytorch_lightning as pl
import os

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        x = self.softmax(x)
        return x

class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, lr_rate):
        super(LightningMNISTClassifier, self).__init__()
        self.model = Model()
        self.lr_rate = lr_rate

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {
            'scheduler':
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            'name':
                'expo_lr'
        }
        return [optimizer], [lr_scheduler]


def prepare_data():
    # transforms for images
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    # prepare transforms standard to MNIST
    mnist_train = MNIST(os.getcwd(),
                        train=True,
                        download=True,
                        transform=transform)
    mnist_train = [mnist_train[i] for i in range(2200)]

    mnist_train, mnist_val = random_split(mnist_train, [2000, 200])

    mnist_test = MNIST(os.getcwd(),
                       train=False,
                       download=True,
                       transform=transform)
    mnist_test = [mnist_test[i] for i in range(3000, 4000)]

    return mnist_train, mnist_val, mnist_test


train, val, test = prepare_data()
train_loader = DataLoader(train, batch_size=64)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)

model = LightningMNISTClassifier(lr_rate=1e-3)

# saves checkpoints to 'dirpath' whenever 'val_loss' has a new min
dirpath = 'model'
os.makedirs(dirpath, exist_ok=True)

trainer = pl.Trainer(max_epochs=30,
                     default_root_dir=dirpath)  #gpus=1

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(dataloaders=test_loader, ckpt_path=None)
