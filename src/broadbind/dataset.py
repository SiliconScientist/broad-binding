import polars as pl
from pymatgen.core.periodic_table import Element
from ase.atoms import Atoms
from torch import tensor
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from pathlib import Path
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split


class BroadBindDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = Path(root)
        self.data = pl.read_parquet(source=self.root).partition_by("index")

    def len(self):
        return len(self.data)

    def get(self, idx):
        df = self.data[idx]
        y = (
            tensor([df.drop_in_place("electron_volts").to_numpy()[0]])
            .float()
            .unsqueeze(dim=1)
        )
        position_columns = ["x", "y", "z"]
        columns_to_exclude = position_columns + ["index", "symbol"]
        x = tensor(df.select(pl.exclude(columns_to_exclude)).to_numpy()).float()
        pos = tensor(df.select(pl.col(position_columns)).to_numpy()).float()
        data = Data(x=x, y=y, pos=pos)
        return data


class BroadBindDataModule(LightningDataModule):
    def __init__(
        self, train: Path, validation: Path, test: Path, batch_size: int
    ) -> None:
        super().__init__()
        self.train_dataset = BroadBindDataset(root=train)
        self.val_dataset = BroadBindDataset(root=validation)
        self.test_dataset = BroadBindDataset(root=test)
        self.batch_size = batch_size
        self.num_workers = cpu_count()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def split_train_val_test(
    data: list[pl.DataFrame],
    random_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train, test = train_test_split(data, train_size=0.8, random_state=random_seed)
    test, validation = train_test_split(test, train_size=0.5, random_state=random_seed)
    return pl.concat(train), pl.concat(validation), pl.concat(test)


def get_unique_element_symbols(systems: list[Atoms]):
    atom_symbols = []
    for atoms in systems:
        symbols = atoms.get_chemical_symbols()
        for symbol in symbols:
            atom_symbols.append(symbol)
    return list(set(atom_symbols))


def get_element_properties(symbols: list[str], properties: list[str]) -> pl.LazyFrame:
    element_properties = []
    for symbol in symbols:
        property_row = {}
        element = Element(symbol)
        for property_str in properties:
            element_property = getattr(element, property_str)
            property_row[property_str] = element_property
        element_properties.append(property_row)
    df = pl.LazyFrame(element_properties).with_columns(symbol=pl.Series(symbols))
    return df


def process_system(atoms: Atoms, energy: float) -> pl.LazyFrame:
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    return pl.LazyFrame(positions, schema=["x", "y", "z"]).with_columns(
        electron_volts=pl.lit(energy), symbol=pl.Series(symbols)
    )


def filter_reactions(
    reactions: list[dict], bound_sites: list[str]
) -> tuple[list[Atoms], list[float]]:
    """
    Filter out any reaction not specified in the products lists
    """
    systems = []
    energies = []
    for reaction in tqdm(reactions, desc="filter_reactions"):
        for key in reaction["reactionSystems"]:
            if key in bound_sites:
                systems.append(reaction["reactionSystems"][key])
                energies.append(reaction["reactionEnergy"])
    return systems, energies


def make_dataframe(
    reactions: list[dict], bound_sites: list[str], properties: list[str]
) -> pl.DataFrame:
    systems, energies = filter_reactions(reactions=reactions, bound_sites=bound_sites)
    element_symbols = get_unique_element_symbols(systems=systems)
    element_properties = get_element_properties(
        symbols=element_symbols, properties=properties
    )
    dfs = []
    for i, (atoms, energy) in tqdm(
        enumerate(zip(systems, energies)), total=len(systems)
    ):
        df = process_system(atoms=atoms, energy=energy).with_columns(index=pl.lit(i))
        dfs.append(df)
    return (
        pl.concat(dfs)
        .join(other=element_properties, on="symbol", how="inner")
        .select(
            pl.col(["symbol", "electron_volts"]),
            pl.exclude(["symbol", "electron_volts"]),
        )
    ).collect()
