import pickle
import toml
import polars as pl
from torch_geometric.loader import DataLoader
from torch.optim import SGD
from torch.nn import MSELoss
from broadbind.config import Config
from broadbind.dataset import BroadBindDataset, make_dataframe
from broadbind.model import GNN


def main():
    config = Config(**toml.load("config.toml"))
    if not config.dataset_path.is_file():
        with open("data/reactions.pkl", "rb") as f:
            reactions = pickle.load(f)
        df = make_dataframe(
            reactions=reactions,
            bound_sites=config.bound_sites,
            properties=config.properties,
        )
        print(df.estimated_size(unit="gb"))
        df.write_parquet(config.dataset_path)
    dataset = BroadBindDataset(root=config.dataset_path)
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
