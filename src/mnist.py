import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils import pad_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTPointCloudDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = sample[0]
        points = sample[1:].reshape(-1, 3)  # Reshape into (x, y, v) format
        # Filter out points where x, y, v are all -1
        valid_points = points[~(points == -1).all(axis=1)]
        valid_points = torch.tensor(valid_points, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return valid_points, label


def load_mnist(batch_size=64):
    # Load the datasets
    train_dataset = MNISTPointCloudDataset("MNISTPointCloud/train.csv")
    test_dataset = MNISTPointCloudDataset("MNISTPointCloud/test.csv")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )
    return train_loader, test_loader


def sample_points():
    # Sample points from the dataset
    train_loader, _ = load_mnist(batch_size=64)
    batch = next(iter(train_loader))
    points, _ = zip(*batch)
    points = pad_batch(points)[:, :, :2]  # Keep only x, y coordinates
    points = points.to(device)

    return points
