from base.base_trainer import BaseTrainer

class BertTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None):
        super().__init__(model, optimizer, loss_fn, train_loader, val_loader)
