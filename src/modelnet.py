import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

from utils import normalize_points

DATA_PATH = "/path/to/dataset/"

def read_off(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()

        # Check if the first line starts with "OFF"
        if first_line.startswith('OFF'):
            # Try to split the first line in case it also contains the counts
            parts = first_line[3:].strip()  # Strip "OFF" and get the rest of the line
            if parts:  # If the line contains numbers after "OFF"
                n_verts, _, _ = map(int, parts.split())
            else:  # Otherwise, read the next line for counts
                n_verts, _, _ = map(int, f.readline().strip().split())
        else:
            raise ValueError(f"Not a valid OFF file: {file_path}")

        # Read vertices
        vertices = []
        for _ in range(n_verts):
            vertex = f.readline().strip().split()
            if len(vertex) != 3:
                raise ValueError(f"Error reading vertices in file: {file_path}")
            vertices.append(list(map(float, vertex)))

        vertices = np.array(vertices)
        
    return vertices


def voxel_grid_downsample(points, leaf_size):
    """
    Optimized downsample of a point cloud using voxel grid filter.
    
    Args:
    points (numpy.ndarray): Input point cloud of shape (N, 3), where N is the number of points.
    leaf_size (float): The size of the voxels (grid cell size).
    
    Returns:
    numpy.ndarray: Downsampled point cloud.
    """
    # Compute the minimum bound of the point cloud
    min_bound = np.min(points, axis=0)

    # Compute voxel indices for each point
    voxel_indices = np.floor((points - min_bound) / leaf_size).astype(np.int32)

    # Get unique voxel indices and their inverse map
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    # Compute the centroids for each unique voxel
    # Use bincount to sum the points in each voxel, then divide by the number of points per voxel to get the centroid
    downsampled_points = np.zeros((unique_voxels.shape[0], points.shape[1]))
    
    for dim in range(points.shape[1]):
        downsampled_points[:, dim] = np.bincount(inverse_indices, weights=points[:, dim]) / np.bincount(inverse_indices)

    return downsampled_points

class ModelNetDataset(Dataset):
    def __init__(self, data_path, split='train', max_samples=None, leaf_size_list=None, min_points=500, max_points=800, normalize=True, strict_min=False):
        self.data_path = data_path
        self.split = split
        self.min_points = min_points
        self.max_points = max_points
        self.normalize = normalize
        self.strict_min = strict_min
        
        # Default leaf sizes if none are provided
        if leaf_size_list is None:
            self.leaf_size_list = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        else:
            self.leaf_size_list = leaf_size_list
        
        # Create class-to-label and label-to-class mappings
        self.class_to_label, self.label_to_class = self.create_class_mapping()
        
        # Get the names of all data files for the given split
        self.file_paths, self.labels = self.get_all_file_paths()

        # If max_samples is provided, subsample the files
        if max_samples is not None and max_samples < len(self.file_paths):
            selected_indices = random.sample(range(len(self.file_paths)), max_samples)
            self.file_paths = [self.file_paths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]

        # Read point clouds
        self.point_clouds = [read_off(fp) for fp in self.file_paths]

        # Normalize point clouds if necessary
        if self.normalize:
            self.point_clouds = [normalize_points(pc) for pc in self.point_clouds]

        # Process and downsample point clouds as needed
        self.point_clouds = self.filter_and_downsample(self.point_clouds)

        # Normalize point clouds again if required after downsampling
        if self.normalize:
            self.point_clouds = [normalize_points(pc) for pc in self.point_clouds]

        # Compute min and max sizes
        self.min_size = min([pc.shape[0] for pc in self.point_clouds])
        self.max_size = max([pc.shape[0] for pc in self.point_clouds])

    def create_class_mapping(self):
        """Create mappings between class names and integer labels."""
        class_names = sorted(os.listdir(self.data_path))  # Sort to ensure consistent ordering
        class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}
        label_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}
        return class_to_label, label_to_class

    def get_all_file_paths(self):
        """Collect all file paths and generate labels based on folder names."""
        all_file_paths = []
        all_labels = []
        for class_folder in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_folder, self.split)
            if os.path.isdir(class_path):
                label = self.class_to_label[class_folder]  # Get the label for the class folder
                for file in os.listdir(class_path):
                    if file.endswith('.off'):
                        all_file_paths.append(os.path.join(class_path, file))
                        all_labels.append(torch.tensor(label, dtype=torch.long))  # Convert to torch.long
        return all_file_paths, all_labels

    def filter_and_downsample(self, point_clouds):
        """Filter point clouds and downsample them based on the min and max point limits."""
        filtered_point_clouds = []
        for pc in point_clouds:
            num_points = pc.shape[0]
            
            # If the point cloud is within the allowed range, include it without downsampling
            if self.min_points <= num_points <= self.max_points:
                filtered_point_clouds.append(pc)
                continue
            
            # Set starting leaf size based on number of points
            starting_leaf_size = self.get_initial_leaf_size(num_points)
            if starting_leaf_size is None:
                continue  # Discard if no starting leaf size is found

            # Try downsampling with different leaf sizes
            downsampled_pc = self.downsample_until_range(pc, starting_leaf_size)
            if downsampled_pc is not None:
                filtered_point_clouds.append(downsampled_pc)

        return filtered_point_clouds

    def get_initial_leaf_size(self, num_points):
        """Determine the initial leaf size based on the number of points."""
        if num_points < 5000:
            return 0.01
        elif 5000 <= num_points < 30000:
            return 0.1
        elif 30000 <= num_points < 80000:
            return 0.2
        elif num_points >= 80000:
            return 0.3
        return None

    def downsample_until_range(self, point_cloud, initial_leaf_size):
        """Downsample a point cloud until it is within the min_points and max_points range."""
        # Find the index of the initial leaf size
        current_leaf_size_idx = self.leaf_size_list.index(initial_leaf_size)
        
        while True:
            downsampled_pc = voxel_grid_downsample(point_cloud, self.leaf_size_list[current_leaf_size_idx])
            num_points = downsampled_pc.shape[0]
            
            # Check if the number of points is within the acceptable range
            if self.min_points <= num_points <= self.max_points:
                return downsampled_pc
            if not self.strict_min and num_points < self.min_points:
                return downsampled_pc
            elif num_points < self.min_points:
                # Go to a smaller leaf size if available
                if current_leaf_size_idx > 0:
                    next_downsampled_pc = voxel_grid_downsample(point_cloud, self.leaf_size_list[current_leaf_size_idx - 1])
                    next_num_points = next_downsampled_pc.shape[0]

                    # Binary search if the next leaf size crosses the min/max boundary
                    if next_num_points > self.max_points:
                        return self.binary_search_leaf_size(point_cloud, self.leaf_size_list[current_leaf_size_idx], self.leaf_size_list[current_leaf_size_idx - 1])
                    current_leaf_size_idx -= 1
                else:
                    return None  # Discard if no smaller leaf size is available
            else:
                # Go to a larger leaf size if available
                if current_leaf_size_idx < len(self.leaf_size_list) - 1:
                    next_downsampled_pc = voxel_grid_downsample(point_cloud, self.leaf_size_list[current_leaf_size_idx + 1])
                    next_num_points = next_downsampled_pc.shape[0]

                    # Binary search if the next leaf size crosses the min/max boundary
                    if next_num_points < self.min_points:
                        return self.binary_search_leaf_size(point_cloud, self.leaf_size_list[current_leaf_size_idx], self.leaf_size_list[current_leaf_size_idx + 1])
                    current_leaf_size_idx += 1
                else:
                    return None  # Discard if no larger leaf size is available

    def binary_search_leaf_size(self, point_cloud, small_leaf_size, large_leaf_size, max_iterations=3):
        """
        Perform binary search between two leaf sizes to find the optimal one that lands within the point range.
        
        Args:
            point_cloud (np.ndarray): The point cloud to downsample.
            small_leaf_size (float): The smaller leaf size that yields too many points.
            large_leaf_size (float): The larger leaf size that yields too few points.
            max_iterations (int): The maximum number of binary search iterations.
        
        Returns:
            np.ndarray: The downsampled point cloud if successful, or None if no valid size is found.
        """
        for _ in range(max_iterations):
            mid_leaf_size = (small_leaf_size + large_leaf_size) / 2.0
            downsampled_pc = voxel_grid_downsample(point_cloud, mid_leaf_size)
            num_points = downsampled_pc.shape[0]
            
            if self.min_points <= num_points <= self.max_points:
                return downsampled_pc
            elif num_points < self.min_points:
                large_leaf_size = mid_leaf_size  # Too few points, increase size
            else:
                small_leaf_size = mid_leaf_size  # Too many points, decrease size
        
        return None  # Return None if no valid size is found after binary search

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.point_clouds)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset along with its label."""
        point_cloud = torch.tensor(self.point_clouds[idx], dtype=torch.float32)
        label = self.labels[idx]
        return point_cloud, label

# Load ModelNet function with labels
def load_modelnet(data_path=DATA_PATH, batch_size=64, n_train=None, n_test=None, **dataset_kwargs):
    # Create train and test datasets
    train_dataset = ModelNetDataset(data_path, split='train', max_samples=n_train, **dataset_kwargs)
    test_dataset = ModelNetDataset(data_path, split='test', max_samples=n_test, **dataset_kwargs)

    # Create DataLoaders for train and test sets
    # We need to shuffle the test set because otherwise it's sorted by classes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    return train_loader, test_loader, train_dataset.label_to_class

def load_modelnet_saved(data_path="/n/holyscratch01/dam_lab/Users/jgeuter/DEQ-GFs/modelnet", size="s", batch_size=64):
    if size == "s":
        train_dataset = torch.load(f"{data_path}/train_dataset_s.pth")
        test_dataset = torch.load(f"{data_path}/test_dataset_s.pth")
    if size == "m":
        train_dataset = torch.load(f"{data_path}/train_dataset.pth")
        test_dataset = torch.load(f"{data_path}/test_dataset.pth")
    elif size == "l":
        train_dataset = torch.load(f"{data_path}/train_dataset_l.pth")
        test_dataset = torch.load(f"{data_path}/test_dataset_l.pth")

    # We need to shuffle the test set because otherwise it's sorted by classes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    return train_loader, test_loader, train_dataset.label_to_class


def compute_avg_points_per_class(dataset):
    """
    Compute the average number of points in the point clouds for each class in the dataset.
    
    Args:
        dataset (ModelNetDataset): An instance of ModelNetDataset.
    
    Returns:
        dict: A dictionary where the keys are class names and the values are the average
              number of points in the point clouds for that class.
    """
    
    # Dictionary to accumulate total points per class and count of samples per class
    points_per_class = defaultdict(list)

    # Iterate over all samples in the dataset
    for idx in range(len(dataset)):
        point_cloud, label = dataset[idx]  # Get the point cloud and its corresponding label
        num_points = point_cloud.shape[0]  # Number of points in the point cloud (N in N*d)

        # Ensure the label is converted to an integer
        label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        
        points_per_class[label].append(num_points)

    # Compute the average number of points per class
    avg_points_per_class = {
        label: sum(points) / len(points) for label, points in points_per_class.items()
    }

    # Map the integer class labels to class names
    avg_points_per_class_named = {
        dataset.label_to_class[label]: avg_points_per_class[label] for label in avg_points_per_class
    }
    
    return avg_points_per_class_named
