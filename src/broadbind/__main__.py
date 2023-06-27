import pickle
import toml
import polars as pl
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
from broadbind.dataset import make_dataframe
from config import Config


class BroadBindDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path

    def len(self):
        return len(self.path.glob("*"))

    def get(self, idx):
        df = pl.read_parquet(self.path / f"{idx}.parquet")
        y = df.drop_in_place("electron_volts")[0]
        position_columns = ["x", "y", "z"]
        x = df.select(pl.exclude(position_columns))
        pos = df.select(pl.col(position_columns))
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

    dataset = BroadBindDataset(path=config.path)

    print(dataset.get(50))

    # y = data.drop_in_place("electron_volts")[0]
    # print(y)
    # break


if __name__ == "__main__":
    main()
