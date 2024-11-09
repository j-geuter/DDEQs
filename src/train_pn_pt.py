from __future__ import print_function

import argparse
import os
import random

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from mnist import load_mnist
from modelnet import load_modelnet_saved
from pointnet import PointNetCls, ShallowPointNetCls, feature_transform_regularizer
from point_transformer import PointTransformerCls
from networks import *
from utils import mask_batch

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
parser.add_argument(
    "--nepoch", type=int, default=1, help="number of epochs to train for"
)
parser.add_argument("--outf", type=str, default="cls", help="output folder")
parser.add_argument("--model_path", type=str, default="", help="model path")
parser.add_argument(
    "--feature_transform", action="store_true", help="use feature transform"
)
parser.add_argument(
    "--pad_to_max",
    type=bool,
    default=False,
    help="if True, pads all batches "
    "to the same size; else, batch size is adaptive to the batch",
)
parser.add_argument(
    "--shallow_net", action="store_true", help="uses shallower network instead"
)
parser.add_argument("--model", type=str, default="pointnet", help="model to use, 'pointnet' or 'pointtransformer'")
parser.add_argument("--batch_per_epoch", type=int, default=1000000)
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--masking", action="store_true")
parser.add_argument("--small_model", action="store_true")
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blue = lambda x: "\033[94m" + x + "\033[0m"

args.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)


if args.dataset == "modelnet-s":
    train_loader, test_loader, class_to_label = load_modelnet_saved(
        batch_size=args.batchSize,
        size="s",
    )
    p = 3
    n_classes = 40
    scheduler_stepsize = 8
    scheduler_gamma = 0.1
elif args.dataset == "modelnet":
    train_loader, test_loader, class_to_label = load_modelnet_saved(
        batch_size=args.batchSize,
        size="m",
    )
    p = 3
    n_classes = 40
    scheduler_stepsize = 8
    scheduler_gamma = 0.1
elif args.dataset == "modelnet-l":
    train_loader, test_loader, class_to_label = load_modelnet_saved(
        batch_size=args.batchSize,
        size="l",
    )
    p = 3
    n_classes = 40
    scheduler_stepsize = 8
    scheduler_gamma = 0.1
elif args.dataset == "mnist":
    train_loader, test_loader = load_mnist(batch_size=args.batchSize)
    p = 2
    n_classes = 10
    scheduler_stepsize = 2
    scheduler_gamma = 0.1

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.model == "pointnet":
    loss_fn = F.nll_loss
    if args.shallow_net:
        classifier = ShallowPointNetCls(k=n_classes, d=p, feature_transform=args.feature_transform)
    else:
        classifier = PointNetCls(k=n_classes, d=p, feature_transform=args.feature_transform)
elif args.model == "pointtransformer":
    loss_fn = F.cross_entropy
    if not args.small_model:
        classifier = PointTransformerCls(dim=p, num_classes=n_classes, masking=args.masking)
    else:
        classifier = PointTransformerCls(dim=p, num_classes=n_classes, masking=args.masking, channels=[32, 48, 96, 192, 256])
if args.model_path != "":
    classifier.load_state_dict(torch.load(args.model))

# torch.autograd.set_detect_anomaly(True)
classifier.to(device)
if args.model != "pointtransformer":
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_stepsize, gamma=scheduler_gamma)
else:
    optimizer = torch.optim.SGD(
        classifier.parameters(),    # Model parameters
        lr=0.05,               # Learning rate (adjust as needed)
        momentum=0.9,          # Set momentum to 0.9
        weight_decay=0.0001    # Set weight decay (L2 regularization) to 0.0001
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[int(0.6 * args.nepoch), int(0.8 * args.nepoch)],  # Epochs at which to drop the learning rate
        gamma=0.1               # Multiply LR by 0.1 (i.e., divide by 10) at each milestone
    )


num_batch = len(train_loader)
if args.pad_to_max:
    max_pad = max(max(sample[0].shape[0] for sample in batch) for batch in train_loader)
else:
    max_pad = None

for epoch in range(args.nepoch):
    for i, data in enumerate(train_loader):
        if i >= args.batch_per_epoch:
            break
        points, target = mask_batch(data, mask=True, dim=p)
        target = torch.tensor(target, dtype=torch.long).to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = loss_fn(pred, target)
        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        if i % 50 == 0:
            print(
                "[%d: %d/%d] train loss: %f accuracy: %f"
                % (
                    epoch,
                    i,
                    num_batch,
                    loss.item(),
                    correct.item() / float(args.batchSize),
                )
            )
            j, data = next(enumerate(test_loader))
            points, target = mask_batch(data, mask=True, dim=p)
            target = torch.tensor(target, dtype=torch.long).to(device)
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = loss_fn(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print(
                "[%d: %d/%d] %s loss: %f accuracy: %f"
                % (
                    epoch,
                    i,
                    num_batch,
                    blue("test"),
                    loss.item(),
                    correct.item() / float(args.batchSize),
                )
            )
    scheduler.step()
    torch.save(classifier.state_dict(), "%s/cls_model_%d.pth" % (args.outf, epoch))

total_correct = 0
total_testset = 0
classifier.eval()
for i, data in tqdm(enumerate(test_loader)):
    points, target = mask_batch(data, mask=True, dim=p)
    target = torch.tensor(target, dtype=torch.long).to(device)
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
