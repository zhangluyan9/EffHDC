from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
dimension_ = 64
load_ = "MNIST_d64_twolayers_retrain_step1.pt"
load_True = True
test_hdc = True
######################################################################

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=10):
        ctx.constant = constant
        return torch.floor(tensor)
        #return torch.div(torch.floor(torch.mul(tensor, constant)), constant)
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
        x =  torch.clamp(torch.div(torch.floor(torch.mul(tensor, 5)), 5),min=0, max=0.2)
        x = x*5
        #print(x)
        return x


class Quantization_integer_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=1000):
        ctx.constant = constant
        x = tensor
        x_ = torch.where(x>=0 , torch.div(torch.ceil(torch.mul(x, 1)), 1), x)
        x = torch.where(x_<0 , torch.div(torch.floor(torch.mul(x_, 1)), 1), x_)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_integer = Quantization_integer_.apply

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(784, dimension_, bias=False)
        self.fc0_ = nn.Linear(dimension_, dimension_, bias=False)
        self.Bn0 = nn.BatchNorm1d(dimension_)

        self.fc0_1 = nn.Linear(dimension_, dimension_, bias=False)
        self.Bn0_ = nn.BatchNorm1d(dimension_)

        self.fc1_0 = nn.Linear(dimension_, dimension_, bias=False)
        self.fc1 = nn.Linear(dimension_, 10, bias=False)
        self.dropout3 = nn.Dropout2d(0.3)
        self.relu = nn.ReLU()


    def forward(self, x):
        
        x = torch.flatten(x, 1)
        #print(x.shape)
        self.fc0.weight.data = Quantization_integer(self.fc0.weight.data)
        x = self.fc0(x)
        x = self.Bn0(x)
        #x =  self.relu(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x, 4)

        self.fc0_.weight.data = Quantization_integer(self.fc0_.weight.data)
        x = self.fc0_(x)
        x = self.Bn0_(x)
        #x =  self.relu(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x, 4)
        return x


def test_represent(model, device, train_loader,test_loader,threshold_):
    model.eval()
    test_loss = 0
    correct = 0
    threshold_ = threshold_
    represent_ = torch.zeros((10,dimension_)).cuda()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            for i in range(target.shape[0]):
                represent_[int(target[i])]+=output[i]

        represent_np_ = np.array(represent_.cpu())
        #np.savez("represent_d_64_MNIST_2layers.npz", represent_ = represent_np_)

        my_ones = torch.ones(represent_.shape[0],represent_.shape[1]).cuda()
        my_zeros = -1*torch.ones(represent_.shape[0],represent_.shape[1]).cuda()
        represent_ = torch.where(represent_<=threshold_, my_zeros, represent_)
        represent_ = torch.where(represent_>=threshold_ , my_ones, represent_)

        count = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_minones = -1* torch.ones(output.shape[0],output.shape[1]).cuda()
            output = torch.where(output==0 , output_minones, output)
            testing_inference = torch.mm(output,represent_.T)
            out = torch.argmax(testing_inference, dim=1)
            count+=torch.sum(torch.eq(out,target))
        print(count)
        represent_np = np.array(represent_.cpu())

        #np.savez("represent_d_64_MNIST_2layers_binary.npz", represent_ = represent_np)
    test_loss /= len(test_loader.dataset)

    return represent_

def train_(args, model, device, train_loader, optimizer, epoch,represent_):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target_mine = torch.zeros((output.shape[0],output.shape[1])).cuda()

        for i in range(output.shape[0]):
            target_mine[i] = represent_[int(target[i])]

        loss = F.mse_loss(output, target_mine.float())

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test_represent_only_test(model, device,test_loader_,repre):
    model.eval()
    test_loss = 0
    correct = 0
    represent_ = repre
    with torch.no_grad():
        sum_all = 0
        for data, target in test_loader_:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_minones = -1* torch.ones(output.shape[0],output.shape[1]).cuda()
            output = torch.where(output==0 , output_minones, output)
            testing_inference = torch.mm(output,represent_.T)
            out = torch.argmax(testing_inference, dim=1)
            sum_all += torch.sum(torch.eq(out,target))
        print(sum_all)
    return sum_all

def test_represent_test(model, device,test_loader_,represent_):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for data, target in test_loader_:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_minones = -1* torch.ones(output.shape[0],output.shape[1]).cuda()
            output = torch.where(output==0 , output_minones, output)
            testing_inference = torch.mm(output,represent_.T)
            out = torch.argmax(testing_inference, dim=1)
            correct+=torch.sum(torch.eq(out,target))

    print(correct)
    return correct

def test_represent_new(model, device, train_loader,test_loader,repre_binary,repre_nb):

    model.eval()
    test_loss = 0
    correct = 0
    represent_ = repre_nb
    represent_b = repre_binary
    threshold = 3500*torch.ones(dimension_).cuda()

    with torch.no_grad():
        my_ones = torch.ones(represent_.shape[1]).cuda()
        my_zeros = torch.zeros(represent_.shape[1]).cuda()
        sum_predifine  = 0
        for times in range(500):
            sum_all = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output_minones = -1* torch.ones(output.shape[0],output.shape[1]).cuda()
                output_ = torch.where(output==0 , output_minones, output)
                testing_inference = torch.mm(output_,represent_b.T)
                testing_inference_ = torch.argmax(testing_inference, dim=1)
                #60,000*10
                for i in range(testing_inference_.shape[0]):
                    if testing_inference_[i]!=target[i]:
                        represent_[int(testing_inference_[i])]-=0.1*output[i]
                        represent_[int(target[i])]+=0.1*output[i]

                represent_b = represent_.clone()
                for i in range(represent_.shape[0]):
                    represent_b[i] = torch.where(represent_b[i]<=threshold[i] , my_zeros, represent_b[i])
                    represent_b[i] = torch.where(represent_b[i]>=threshold[i] , my_ones, represent_b[i])

                #print(out.shape,out.shape[0])
                #represent_9112 = np.array(represent_.cpu())
                #np.savez("represent_d_64_MNIST_2layers_binary_9112.npz", represent_ = represent_9112)
                sum_all += torch.sum(torch.eq(testing_inference_,target))
            #print(sum_all)
            sum_all = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output_minones = -1* torch.ones(output.shape[0],output.shape[1]).cuda()
                output = torch.where(output==0 , output_minones, output)
                testing_inference = torch.mm(output,represent_b.T)
                out = torch.argmax(testing_inference, dim=1)
                sum_all += torch.sum(torch.eq(out,target))
            #print(sum_all)
            print("second time",sum_all/10000)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--T', type=int, default=5, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddQuantization()
        ])

    transform=transforms.Compose([
        transforms.ToTensor(),
        AddQuantization()
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset1,batch_size = 512)

    for i in range(3):
        transform_train_1 = transforms.Compose([
            transforms.ToTensor(),
            AddQuantization()
        ])
        dataset1 = dataset1+ datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train_1)

    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader_ = torch.utils.data.DataLoader(dataset1,batch_size = 512)

    test_loader = torch.utils.data.DataLoader(dataset2, batch_size = 512)

    model = Net().to(device)
    if load_True:
        model.load_state_dict(torch.load(load_), strict=False)

    if args.resume != None:
        load_model(torch.load(args.resume), model)
    for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #1. Train the HDC encoder 
    repre = test_represent(model, device, train_loader,test_loader,3500)
    #test_represent(model_, device, train_loader_,test_loader)
    cor_ = 0
    for epoch in range(1, args.epochs + 1):
        train_(args, model, device, train_loader, optimizer, epoch,repre)
        correct = test_represent_only_test(model, device,test_loader,repre)
        if correct>cor_:
            cor_ = correct
            print(cor_,correct)
            #torch.save(model.state_dict(), "MNIST_d64_twolayers_retrain_step1.pt")
    
    #2. Retrain the hyperVector:
    """
    f = np.load('represent_d_64_MNIST_2layers_binary.npz')
    repre_binary = f['represent_']
    repre_binary = torch.tensor(repre_binary).cuda()
    f = np.load('represent_d_64_MNIST_2layers.npz')
    repre_nb = f['represent_']
    repre_nb = torch.tensor(repre_nb).cuda()
    test_represent_new(model, device, train_loader,test_loader,repre_binary,repre_nb)
    """

    #3. Final Test:
    repre = test_represent(model, device, train_loader,test_loader,3500)
    f = np.load('MNIST_d64_twolayers_9112.npz')
    repre_binary = f['represent_']
    repre_binary = torch.tensor(repre_binary).cuda()
    test_represent_only_test(model, device,test_loader,repre_binary)




if __name__ == '__main__':
    main()
