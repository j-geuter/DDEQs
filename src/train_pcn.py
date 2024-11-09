# adapted from:
# https://github.com/qinglew/PCN-PyTorch/blob/master/train.py

import argparse
import os
import datetime
import pandas as pd

import torch
import torch.optim as Optim
from torch.utils.data import Dataset

from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from pcn import PCN
from utils import mask_batch, target_completion, chamfer_distance

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
    train_dataset = MNISTPointCloudDataset("../MNISTPointCloud/train.csv")
    test_dataset = MNISTPointCloudDataset("../MNISTPointCloud/test.csv")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )
    return train_loader, test_loader, len(train_dataset), len(test_dataset)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.category)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.category, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name), False)
    log(log_fd, "Logger directory: {}".format(logger_path), False)
    log(log_fd, str(params), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params):
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    log(log_fd, 'Loading Data...')

    train_dataloader, val_dataloader, len_train, len_val = load_mnist(batch_size=params.batch_size)
    log(log_fd, "Dataset loaded!")

    # model
    model = PCN(dim=2, num_dense=params.num_dense, latent_dim=params.latent_dim, grid_size=params.grid_size).to(device)

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.999))
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    step = len(train_dataloader) // params.log_frequency

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    for epoch in range(1, params.epochs + 1):
        # hyperparameter alpha
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif train_step < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        # training
        model.train()
        for i, batch in enumerate(train_dataloader):
            points, _ = mask_batch(batch, mask=True, dim=2)
            target = points.clone()
            points = target_completion(points, return_input=False)
            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(points)
            
            # loss function
            if params.coarse_loss == 'cd':
                loss1 = chamfer_distance(coarse_pred, target)
            else:
                raise ValueError('Not implemented loss {}'.format(params.coarse_loss))
                
            loss2 = chamfer_distance(dense_pred, target)
            loss = loss1 + alpha * loss2

            # back propagation
            loss.backward()
            optimizer.step()

            if (i + 1) % step == 0:
                log(log_fd, "Training Epoch [{:03d}/{:03d}] - Iteration [{:03d}/{:03d}]: coarse loss = {:.6f}, dense l1 cd = {:.6f}, total loss = {:.6f}"
                    .format(epoch, params.epochs, i + 1, len(train_dataloader), loss1.item() * 1e3, loss2.item() * 1e3, loss.item() * 1e3))
            
            train_writer.add_scalar('coarse', loss1.item(), train_step)
            train_writer.add_scalar('dense', loss2.item(), train_step)
            train_writer.add_scalar('total', loss.item(), train_step)
            train_step += 1
        
        lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():

            for i, batch in enumerate(val_dataloader):
                points, _ = mask_batch(batch, mask=True, dim=2)
                target = points.clone()
                points = target_completion(points, return_input=False)
                coarse_pred, dense_pred = model(points)
                total_cd_l1 += chamfer_distance(dense_pred, target).item()

            total_cd_l1 /= len_val
            val_writer.add_scalar('l1_cd', total_cd_l1, val_step)
            val_step += 1

            log(log_fd, "Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, params.epochs, total_cd_l1 * 1e3))
        
        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_l1_cd.pth'))
            
    log(log_fd, 'Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))
    log_fd.close()
    return model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCN')
    parser.add_argument('--exp_name', type=str, default='test', help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loader')
    parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers for data loader')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--num_dense', type=int, default=16384, help='Number of dense points')
    parser.add_argument('--latent_dim', type=int, default=1024, help='Latent dimension')
    parser.add_argument('--grid_size', type=int, default=4)
    params = parser.parse_args()
    
    model = train(params)
