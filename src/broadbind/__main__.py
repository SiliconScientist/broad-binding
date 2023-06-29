import pickle
import toml
import polars as pl
import numpy as np
from torch import tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
from ase.visualize import view

from torch.optim import SGD
from torch.nn import MSELoss

from broadbind.config import Config
from broadbind.dataset import make_dataframe
from broadbind.model import GNN


class BroadBindDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.root = root

    def len(self):
        return len(list(self.root.glob("*")))

    def get(self, idx):
        df = pl.read_parquet(self.root / f"{idx}.parquet")
        y = (
            tensor([df.drop_in_place("electron_volts").to_numpy()[0]])
            .float()
            .unsqueeze(dim=1)
        )
        position_columns = ["x", "y", "z"]
        x = tensor(df.select(pl.exclude(position_columns)).to_numpy()).float()
        pos = tensor(df.select(pl.col(position_columns)).to_numpy()).float()
        data = Data(x=x, y=y, pos=pos)
        return data


def main():
    config = Config(**toml.load("config.toml"))
    with open("data/temp.pickle", "rb") as f:
        smol_reactions = pickle.load(f)
    big_df = make_dataframe(
        reactions=smol_reactions,
        bound_sites=config.bound_sites,
        properties=config.properties,
    )
    big_df.write_parquet("data/dataset.parquet")
    data = pl.read_parquet(source="data/dataset.parquet")
    for name, df in data.groupby("index"):
        df = df.select(pl.exclude(["symbol", "index"]))
        df.write_parquet(f"data/dataset/{name}.parquet")

    dataset = BroadBindDataset(root=config.path)
    loader = DataLoader(
        dataset=dataset, batch_size=config.training.batch_size, shuffle=True
    )
    n_features = len(config.properties)
    model = GNN(
        input_dimension=n_features,
        output_dimension=1,
        spatial_dimension=3,
        **config.model.dict(),
    )
    optimizer = SGD(params=model.parameters(), lr=config.training.learning_rate)
    criterion = MSELoss()
    for _ in range(config.training.max_epoch):
        for i, data in enumerate(loader):
            optimizer.zero_grad()  # Clear gradients.
            y_pred = model(data)  # Forward pass.
            loss = criterion(y_pred, data.y)  # Loss computation.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update model parameters.
            if i // 100 == 0:
                print(loss.item())


if __name__ == "__main__":
    main()
