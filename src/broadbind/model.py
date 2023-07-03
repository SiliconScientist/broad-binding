from torch.nn import Module, Linear, ReLU, Dropout, MSELoss
from torch.optim import Optimizer
from torch_geometric.nn import XConv, global_mean_pool
from lightning.pytorch.utilities.grads import grad_norm
from lightning import LightningModule


class GNN(Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        spatial_dimension: int,
        k_neighbors: int,
        hidden_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv = XConv(
            in_channels=input_dimension,
            out_channels=hidden_dimension,
            dim=spatial_dimension,
            kernel_size=k_neighbors,
            hidden_channels=hidden_channels,
        )
        self.relu = ReLU()
        self.linear = Linear(
            in_features=hidden_dimension, out_features=output_dimension
        )
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, data):
        x = self.conv(x=data.x, pos=data.pos, batch=data.batch)
        x = self.relu(x)
        x = global_mean_pool(x=x, batch=data.batch)
        return self.linear(x)


class BroadBindNetwork(LightningModule):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: MSELoss,
        scheduler_config: dict,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler_config = scheduler_config

    def step(self, batch, step_type):
        predictions = self.model(batch)
        loss = self.criterion(predictions, batch.y)
        self.log(f"{step_type}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):
        return self.step(test_batch, "test")

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_config}

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
