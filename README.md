# Distributional Deep Equilibrium Models
Welcome to the DDEQ repository!

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To use this project, you first need to download the datasets, namely [MNIST Point Cloud](https://www.kaggle.com/datasets/cristiangarcia/pointcloudmnist2d)
and [ModelNet40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset/data). The MNIST files should be saved in the `MNISTPointCloud` folder.
Save the ModelNet40 dataset somewhere and change `DATA_PATH` in `src/modelnet.py` accordingly. The dataset can then be created using the `load_modelnet` function in that file. Since creating the dataset takes some time, it's a good idea to save the dataset, which can then be loaded with the `load_modelnet_saved` function.

Once the datasets are set up, simply run the `train_torchdeq.py` file.

## License
This project is licensed under the [GNU License](LICENSE).

