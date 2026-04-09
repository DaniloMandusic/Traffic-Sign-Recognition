import pytorch_lightning as pl

class TrafficSignClassifier(pl.LightningModule):
    def __init__(self, config, model, criterion, optimizer):
        super().__init__()

        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.save_hyperparameters(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = logits.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        return self.optimizer



