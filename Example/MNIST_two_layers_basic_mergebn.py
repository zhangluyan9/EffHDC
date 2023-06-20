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

# Setting 
dimension_ = 64
load_ = "MNIST_d64_twolayers.pt"
#MNIST_d64_twolayers
load_True = True
#For training,set test_hdc as False. For inference, set it as True.
test_hdc = True
factor_t = [-1,-1]
#save_ = "dimension_32_test_MNIST_binary_weight_1_2layers_1_1_bn_1.pt"
######################################################################

def quantize_to_bit_(x, nbit):
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    return x

def fold_bn_into_fc_inter(fc_layer, bn_layer,index):

    assert isinstance(fc_layer, nn.Linear) and isinstance(bn_layer, nn.BatchNorm1d), \
        "Expecting PyTorch layers nn.Linear and nn.BatchNorm1d."

    # Fetch parameters
    bn_st_dict = bn_layer.state_dict()
    scale = bn_st_dict['weight']
    shift = bn_st_dict['bias']
    running_mean = bn_st_dict['running_mean']
    running_var = bn_st_dict['running_var']

    # Fold BN parameters into FC parameters
    scale_var_shift = scale / torch.sqrt(running_var + bn_layer.eps)
    fc_layer.weight.data *= scale_var_shift.unsqueeze(-1)
    #Because we did not use bias before, so we set it as (0-running_mean)
    fc_layer.bias.data = scale_var_shift * (0- running_mean) + shift

    factor = torch.max(torch.abs(fc_layer.weight.data))
        
    if factor<torch.max(torch.abs(fc_layer.bias.data)):
        factor = torch.max(torch.abs(fc_layer.bias.data))

    if factor_t[index]==-1:
        factor_t[index] = factor
        fc_layer.weight.data = nn.Parameter(quantize_to_bit_(fc_layer.weight.data/factor_t[index], 32))
        fc_layer.bias.data = nn.Parameter(quantize_to_bit_(fc_layer.bias.data/factor_t[index], 32))
        #print(factor_t)
        #x = self.fc0_(x)
        #my_ones = torch.ones(x.shape[0],x.shape[1]).cuda()
        #my_zeros = 0*torch.ones(x.shape[0],x.shape[1]).cuda()
        #x = torch.where(x<=(2.0**(32-1))/factor_t[1], my_zeros, x)
        #x = torch.where(x>=(2.0**(32-1))/factor_t[1] , my_ones, x)
    return fc_layer

#def my_fc(fc_layer, x):


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
        x =  torch.clamp(torch.div(torch.floor(torch.mul(tensor, 5)), 5),min=0, max=0.2)
        x = x*5
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
        self.fc0 = nn.Linear(784, dimension_, bias=True)
        self.fc0_ = nn.Linear(dimension_, dimension_, bias=True)
        self.Bn0 = nn.BatchNorm1d(dimension_)

        self.Bn0_ = nn.BatchNorm1d(dimension_)

        self.fc1 = nn.Linear(dimension_, 10, bias=False)
        self.dropout3 = nn.Dropout2d(0.3)
        self.relu = nn.ReLU()


    def forward(self, x):
        #training tricks: Step1: tain only fc,bn. Step2: replace relu with clamp and quantiztion. 
        #Step3: quantizaed weights layer by layer and retrain layer by layer
        x = torch.flatten(x, 1)
        #print(x.shape)
        self.fc0.weight.data = Quantization_integer(self.fc0.weight.data)
        x = self.fc0(x)
        #print(x)
        x = self.Bn0(x)
        self.fc0 = fold_bn_into_fc_inter(self.fc0,self.Bn0,0)
        #print(x)
        #x =  self.relu(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x, 4)

        self.fc0_.weight.data = Quantization_integer(self.fc0_.weight.data)
        x = self.fc0_(x)
        x = self.Bn0_(x)
        self.fc0_ = fold_bn_into_fc_inter(self.fc0_,self.Bn0_,1)

        #x =  self.relu(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x, 4)
        #print(x)

        if not test_hdc:
            x = self.fc1(x)
        return x


class Net_plain(nn.Module):
    def __init__(self):
        super(Net_plain, self).__init__()
        self.fc0 = nn.Linear(784, dimension_, bias=True)
        self.fc0_ = nn.Linear(dimension_, dimension_, bias=True)
        self.fc1 = nn.Linear(dimension_, 10, bias=False)
        self.dropout3 = nn.Dropout2d(0.3)
        self.relu = nn.ReLU()


    def forward(self, x):

        x = torch.flatten(x, 1)

        x = self.fc0(x)
        my_ones = torch.ones(x.shape[0],x.shape[1]).cuda()
        my_zeros = 0*torch.ones(x.shape[0],x.shape[1]).cuda()
        x = torch.where(x<=(2.0**(32-1))/factor_t[0], my_zeros, x)
        x = torch.where(x>=(2.0**(32-1))/factor_t[0] , my_ones, x)

        x = self.fc0_(x)
        my_ones = torch.ones(x.shape[0],x.shape[1]).cuda()
        my_zeros = 0*torch.ones(x.shape[0],x.shape[1]).cuda()
        x = torch.where(x<=(2.0**(32-1))/factor_t[1], my_zeros, x)
        x = torch.where(x>=(2.0**(32-1))/factor_t[1] , my_ones, x)

        #print(x)

        if not test_hdc:
            x = self.fc1(x)
        return x



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, onehot.type(torch.float))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test_pass(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
    return correct

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct
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
        print(count/100)

    test_loss /= len(test_loader.dataset)

    return correct



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
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

    test_loader = torch.utils.data.DataLoader(dataset2, batch_size = 10000)

    model = Net().to(device)
    model_plain = Net_plain().to(device)
    if load_True:
        model.load_state_dict(torch.load(load_), strict=False)

    ##### merge bn
    
    if args.resume != None:
        load_model(torch.load(args.resume), model)
    
    for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    ACC = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader_, optimizer, epoch)
        ACC_ = test(model, device, test_loader)
        if ACC_>ACC or ACC_ == ACC:
            ACC = ACC_
            #torch.save(model.state_dict(), save_)
        
        scheduler.step()

    test_pass(model, device, test_loader)
    #x = model(torch.tensor([0,1]))
    torch.save(model.state_dict(), "merged_bn.pt")
    model_plain.load_state_dict(torch.load("merged_bn.pt"), strict=False)
    #load_model(model.state_dict(),model_plain)

    if test_hdc:
        repre = test_represent(model_plain, device, train_loader,test_loader,500)
        repre = test_represent(model_plain, device, train_loader,test_loader,1000)
        repre = test_represent(model_plain, device, train_loader,test_loader,1500)
        repre = test_represent(model_plain, device, train_loader,test_loader,2000)
        repre = test_represent(model_plain, device, train_loader,test_loader,2500)
        repre = test_represent(model_plain, device, train_loader,test_loader,3000)
        repre = test_represent(model_plain, device, train_loader,test_loader,3500)
        repre = test_represent(model_plain, device, train_loader,test_loader,4000)
        repre = test_represent(model_plain, device, train_loader,test_loader,4500)
    else:
        test(model_plain, device, test_loader)



if __name__ == '__main__':
    main()
