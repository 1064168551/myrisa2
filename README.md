# myrisa2
deep learning2
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from torchvision import datasets, transforms
from sklearn import preprocessing
import time
from glob import glob
#import cv2
from torch.autograd import Variable
from PIL import Image

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def load_data(trainPath, testPath, labelpath):
    # nfile = 0
    # for Name in os.listdir(trainPath):
    #     if Name.split(".")[-1] == "dat":
    #         # print(Name)
    #         fileName = os.path.join(trainPath, Name)
    #         data = np.loadtxt(fileName)[:, 0]
    #         data = np.expand_dims(data, axis=0)
    #         if nfile == 0:
    #             train_data = data
    #         else:
    #             train_data = np.concatenate((train_data, data), axis=0)
    #         nfile += 1

    # np.save(os.path.join(trainPath, 'train_data'), train_data)

    # nfile = 0
    # for Name in os.listdir(testPath):
    #     if Name.split(".")[-1] == "dat":
    #         # print(Name)
    #         fileName = os.path.join(testPath, Name)
    #         data = np.loadtxt(fileName)[:, 0]
    #         data = np.expand_dims(data, axis=0)
    #         if nfile == 0:
    #             test_data = data
    #         else:
    #             test_data = np.concatenate((test_data, data), axis=0)
    #         nfile += 1

    # np.save(os.path.join(testPath, 'test_data'), test_data)

    train_data = np.load(os.path.join(trainPath, 'train_data_3500.npy'))[1499:3499, :]
    
    # for Name in os.listdir(trainPath):
    #     if Name.split(".")[-1] == "dat":
    #         # print(Name)
    #         fileName = os.path.join(trainPath, Name)
    #         data = np.loadtxt(fileName)[:, 0]
    #         data = np.expand_dims(data, axis=0)
    #         train_data = np.concatenate((train_data, data), axis=0)

    # np.save(os.path.join(trainPath, 'train_data_3500'), train_data)
    test_data = np.load(os.path.join(testPath, 'test_data.npy'))

    nfile = 0
    fileName = os.path.join(trainPath, labelpath, 'deff.dat')
    train_label = np.loadtxt(fileName)[1499:3499]*100

    # train_label_fid = open(os.path.join(trainPath, labelpath, 'train_labels.dat'), 'w')
    # np.savetxt(train_label_fid, train_labels)
    # train_label_fid.close()
            
    nfile = 0
    fileName = os.path.join(testPath, labelpath, 'deff.dat')
    test_label = np.loadtxt(fileName)*100

    # test_label_fid = open(os.path.join(testPath, labelpath, 'test_labels.dat'), 'w')
    # np.savetxt(test_label_fid, test_labels)
    # test_label_fid.close()

    # train_labels = np.loadtxt(os.path.join(trainPath, labelpath, 'train_labels.dat'))
    # test_labels = np.loadtxt(os.path.join(testPath, labelpath, 'test_labels.dat'))

    print("training samples:{}".format(np.shape(train_data)))
    print("testing samples: {}".format(np.shape(test_data)))
    return train_data, test_data, train_label, test_label
    
class FeatureVisualization():
    def __init__(self, model, layer, input_scaler, label_scaler, device, epoch, interval, savepath, data, target):
        self.layer=layer
        self.model = model
        self.input_scaler = input_scaler
        self.label_scaler = label_scaler
        self.device = device
        self.epoch = epoch
        self.interval = interval
        self.savepath = savepath
        self.data = data
        self.target = target
            

def train(args, model, device, train_loader, criterion, optimizer, epoch, savepath):
    model.train()
    train_Loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # try:
        #     output = model(data)
        # except RuntimeError as exception:
        #     if "out of memory" in str(exception):
        #         # print("Warning: out of memory")
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
        #         else:
        #             raise exception
        output = model(data)
        train_Loss += criterion(output, target).item()

    train_Loss /= len(train_loader.dataset)
    
    # myClass = FeatureVisualization(model, 16, input_scaler, label_scaler, device, epoch, interval, savepath)
    # myClass.get_feature()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # def closure():
        #     optimizer.zero_grad()
        #     output = model(data)
        #     loss = F.smooth_l1_loss(output, target)
        #     loss.backward()
        #     return loss
        # optimizer.step(closure)
        optimizer.zero_grad()
        # try:
        #     output = model(data)
        # except RuntimeError as exception:
        #     if "out of memory" in str(exception):
        #         # print("Warning: out of memory")
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
        #         else:
        #             raise exception
        output = model(data)
        loss = F.smooth_l1_loss(output, target)
        loss.backward()
        optimizer.step()
    
    return output, target, train_Loss
              

def test(args, model, device, test_loader, criterion, epoch, savepath):
    # model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # try:
            #     output = model(data)
            # except RuntimeError as exception:
            #     if "out of memory" in str(exception):
            #         # print("Warning: out of memory")
            #         if hasattr(torch.cuda, 'empty_cache'):
            #             torch.cuda.empty_cache()
            #         else:
            #             raise exception
            output = model(data)
            # test_loss += criterion(output, target).item()  # sum up batch loss
            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader.dataset)

    return output, target, test_loss

def save_model(model_dir, epoch, model):
    print("[*] Save models to {}...".format(model_dir))

    torch.save(model.state_dict(), '{}/CNN_{}.pth'.format(model_dir, epoch))

def load_model(model_dir, model, use_cuda):
    print("[*] Load models from {}...".format(model_dir))

    paths = glob(os.path.join(model_dir, '*.pth'))
    paths.sort()

    if len(paths) == 0:
        print("[!] No checkpoint found in {}...".format(model_dir))
        return

    idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
    epoch = max(idxes)

    if not use_cuda:
        map_location = lambda storage, loc: storage
    else: 
        map_location = None

    CNN_filename = '{}/CNN_{}.pth'.format(model_dir, epoch)
    model.load_state_dict(
        torch.load(CNN_filename, map_location=map_location))
    print("[*] network loaded: {}".format(CNN_filename))

    return epoch

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='effective diffusivity of porous media')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--dataset', type=str, default="test3", metavar='dataSet',
                        help='the name of dataset')
    parser.add_argument('--labelpath', type=str, default="target", metavar='labelpath',
                        help='the file name of labels (default: target)')
    parser.add_argument('--input_height', type=int, default=100, metavar='ih',
                        help='input height of the matrix (default: 32)')
    parser.add_argument('--input_width', type=int, default=100, metavar='iw',
                        help='input width of the matrix (default: 32)')
    parser.add_argument('--input_depth', type=int, default=100, metavar='iw',
                        help='input depth of the matrix (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='b1',
                        help='Adam beta1 (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load_path', type=str, default='', metavar='loadpath',
                        help='where to load trained model')
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    trainPath = os.path.join(".", "data", args.dataset)
    testPath = os.path.join(trainPath, "test")
    labelpath = args.labelpath
    ih = args.input_height
    iw = args.input_width
    ide = args.input_depth
    train_data, test_data, train_labels, test_labels = load_data(trainPath, testPath, labelpath)

    ntrain = len(train_data)
    ntest = len(test_data)
    assert(ntrain == len(train_labels))
    assert(ntest == len(test_labels))

    train_data = train_data.reshape((ntrain, 1,  ih, iw, ide))
    test_data = test_data.reshape((ntest, 1, ih, iw, ide))
    print("Input training samples:{}".format(np.shape(train_data)))
    print("Input testing samples: {}".format(np.shape(test_data)))


    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()
    train_labels = torch.from_numpy(train_labels).float()
    test_labels = torch.from_numpy(test_labels).float()

    custom_TrainSet = data.TensorDataset(train_data, train_labels)
    custom_TestSet = data.TensorDataset(test_data, test_labels)

    train_loader = data.DataLoader(
        custom_TrainSet, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(
        custom_TestSet, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = nn.Sequential(
        nn.Conv3d(1, 20, kernel_size = (5,5,3)),
        # nn.Dropout3d(0.5),
        # nn.BatchNorm3d(20),
        nn.MaxPool3d(kernel_size = 2, padding=0, dilation=1, stride=2),
        nn.ReLU(),
        nn.Conv3d(20, 40, kernel_size = (5,5,3)),
        # nn.Dropout3d(0.5),
        # nn.BatchNorm3d(40),
        nn.MaxPool3d(kernel_size = 2, padding=0, dilation=1, stride=2),
        # nn.MaxPool3d(kernel_size = 2, padding=0, dilation=1, stride=2),
        nn.ReLU(),
        nn.Conv3d(40, 60, kernel_size = (5,5,3)),
        nn.Dropout3d(0.5),
        # nn.BatchNorm3d(60),
        # nn.MaxPool3d(kernel_size = 2, padding=0, dilation=1, stride=2),
        nn.ReLU(),
        nn.Conv3d(60, 80, kernel_size = (5,5,3)),
        nn.Dropout3d(0.5),
        # nn.BatchNorm3d(80),
        # nn.MaxPool3d(kernel_size = 2, padding=0, dilation=1, stride=2),
        nn.ReLU(),
        nn.Conv3d(80, 100, kernel_size = (5,5,3)),
        nn.Dropout3d(0.5),
        # nn.BatchNorm3d(80),
        # nn.MaxPool3d(kernel_size = 2, padding=0, dilation=1, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(170000, 1)
        # nn.Linear(17500, 1)
    ).to(device)

    print(model)
    train_criterion = nn.MSELoss(size_average=True)
    test_criterion = nn.MSELoss(size_average=True)

#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.LBFGS(model.parameters())
    
    out_dir = os.path.join(".", "result", args.dataset, time.strftime('%m%d_%H%M%S', time.localtime(time.time())))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    LossFile = os.path.join(out_dir, "Loss.dat")
    str_TrainLoss = ["{:>16s}".format("TrainLoss")]
    str_TrainLoss = "\t".join(str_TrainLoss)
    str_TestLoss = ["{:>16s}".format("TestLoss")]
    str_TestLoss = "\t".join(str_TestLoss)
    loss_fid = open(LossFile, "a")
    loss_fid.write("{:>16s}{}{}\n".format("ID", str_TrainLoss, str_TestLoss))

    model_dir = os.path.join(out_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.load_path:
        epoch = load_model(args.load_path, model, use_cuda) + 1
    else:
        epoch = 1

    while epoch <= args.epochs + 1:
        outfile = "{}_test_epoch{:>05d}.dat".format(args.dataset.strip(), epoch)
        outfile = os.path.join(out_dir, outfile)
        outfile1 = "{}_train_epoch{:>05d}.dat".format(args.dataset.strip(), epoch)
        outfile1 = os.path.join(out_dir, outfile1)
        output, target, test_loss = test(args, model, device, test_loader, test_criterion, epoch, out_dir)
        output1, target1, train_loss = train(args, model, device, train_loader, train_criterion, optimizer, epoch, out_dir)

        np.savetxt(loss_fid, np.expand_dims(np.r_[epoch-1, train_loss, test_loss], axis=-1).T, fmt='%16.7f', delimiter='\t')
        
        loss_fid.flush()

        fid = open(outfile, "a")
        fid1 = open(outfile1, "a")
        str_ground = ["{:>16s}".format("groundTruth")]
        str_ground = "\t".join(str_ground)
        str_predict = ["{:>16s}".format("predict")]
        str_predict = "\t".join(str_predict)
        fid.write("{:>16s}{}{}\n".format("ID", str_ground, str_predict))
        fid1.write("{:>16s}{}{}\n".format("ID", str_ground, str_predict))
        
        # # ID = np.expand_dims(np.arange(1, len(output)+1), -1)
        # print(np.shape(target.detach().cpu().numpy()))
        # print(np.shape(output.detach().cpu().numpy()))
        np.savetxt(fid, np.c_[range(args.test_batch_size), np.squeeze(target.detach().cpu().numpy()), np.squeeze(output.detach().cpu().numpy())], fmt='%16.7f', delimiter='\t')
        fid.close()
        np.savetxt(fid1, np.c_[range(args.batch_size), np.squeeze(target1.detach().cpu().numpy()), np.squeeze(output1.detach().cpu().numpy())], fmt='%16.7f', delimiter='\t')
        fid1.close()

        if epoch % 500 == 0:
            save_model(model_dir, epoch, model)

        epoch += 1

    loss_fid.close()

if __name__ == '__main__':
    main()
