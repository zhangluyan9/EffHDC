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
import pickle 
dimension_ = 128
load_ = "models/128_test_ucihar_3layers_step4.pt"
load_True = True
test_hdc = True
save_ = "128_test_ucihar_3layers_step4_1.pt"
factor_t = [-1,-1,-1]

######################################################################
def test_pass(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
    return correct

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
    
    return fc_layer

class Net_plain(nn.Module):
    def __init__(self):
        super(Net_plain, self).__init__()
        self.fc0 = nn.Linear(2244, dimension_, bias=True)
        self.Bn0 = nn.BatchNorm1d(dimension_)

        self.fc0_ = nn.Linear(dimension_, dimension_, bias=True)
        self.Bn0_ = nn.BatchNorm1d(dimension_)

        self.fc0_1 = nn.Linear(dimension_, dimension_, bias=True)
        self.Bn0_1 = nn.BatchNorm1d(dimension_)

        self.fc1 = nn.Linear(dimension_, 6, bias=False)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = torch.flatten(x, 1)
        #self.fc0.weight.data = Quantization_integer(self.fc0.weight.data)

        x = self.fc0(x)
        #x = self.Bn0(x)
        #x = torch.clamp(x, min=0, max=1)
        #x = Quantization_(x, 4)

        my_ones = torch.ones(x.shape[0],x.shape[1]).cuda()
        my_zeros = 0*torch.ones(x.shape[0],x.shape[1]).cuda()
        x = torch.where(x<=(2.0**(32-1))/factor_t[0], my_zeros, x)
        x = torch.where(x>=(2.0**(32-1))/factor_t[0] , my_ones, x)
        #self.fc0_.weight.data = Quantization_integer(self.fc0_.weight.data)
        x = self.fc0_(x)
        #x = self.Bn0_(x)
        #x = torch.clamp(x, min=0, max=1)
        #x = Quantization_(x, 4)

        my_ones = torch.ones(x.shape[0],x.shape[1]).cuda()
        my_zeros = 0*torch.ones(x.shape[0],x.shape[1]).cuda()
        x = torch.where(x<=(2.0**(32-1))/factor_t[1], my_zeros, x)
        x = torch.where(x>=(2.0**(32-1))/factor_t[1] , my_ones, x)

        # #self.fc0_1.weight.data = Quantization_integer(self.fc0_1.weight.data)
        # x = self.fc0_1(x)
        # #x = self.Bn0_1(x)
        # #x = torch.clamp(x, min=0, max=1)
        # #x = Quantization_(x, 4)
        
        # my_ones = torch.ones(x.shape[0],x.shape[1]).cuda()
        # my_zeros = 0*torch.ones(x.shape[0],x.shape[1]).cuda()
        # x = torch.where(x<=(2.0**(32-1))/factor_t[2], my_zeros, x)
        # x = torch.where(x>=(2.0**(32-1))/factor_t[2] , my_ones, x)
        
        if not test_hdc:
            x = self.fc1(x)
        return x



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
        #return torch.floor(tensor)
        x = tensor
        #x = torch.sign(x)
        x_ = torch.where(x>=0 , torch.div(torch.ceil(torch.mul(x, 1)), 1), x)
        x = torch.where(x_<0 , torch.div(torch.floor(torch.mul(x_, 1)), 1), x_)
        #print(x)
        return x
        #return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_integer = Quantization_integer_.apply

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(2244, dimension_, bias=True)
        self.Bn0 = nn.BatchNorm1d(dimension_)

        self.fc0_ = nn.Linear(dimension_, dimension_, bias=True)
        self.Bn0_ = nn.BatchNorm1d(dimension_)

        self.fc0_1 = nn.Linear(dimension_, dimension_, bias=True)
        self.Bn0_1 = nn.BatchNorm1d(dimension_)

        self.fc1 = nn.Linear(dimension_, 6, bias=True)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = torch.flatten(x, 1)
        #print(x.shape)
        self.fc0.weight.data = Quantization_integer(self.fc0.weight.data)
        x = self.fc0(x)
        x = self.Bn0(x)
        # x =  self.relu(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x, 4)
        self.fc0 = fold_bn_into_fc_inter(self.fc0,self.Bn0,0)


        self.fc0_.weight.data = Quantization_integer(self.fc0_.weight.data)
        x = self.fc0_(x)
        x = self.Bn0_(x)
        # x =  self.relu(x)
        x = torch.clamp(x, min=0, max=1)
        x = Quantization_(x, 4)
        self.fc0_ = fold_bn_into_fc_inter(self.fc0_,self.Bn0_,1)

        # self.fc0_1.weight.data = Quantization_integer(self.fc0_1.weight.data)
        # x = self.fc0_1(x)
        # x = self.Bn0_1(x)
        # # x =  self.relu(x)
        # x = torch.clamp(x, min=0, max=1)
        # x = Quantization_(x, 4)


        if not test_hdc:
            x = self.fc1(x)
        return x





def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 6)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, onehot.type(torch.float))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 6)
            output = model(data)
            test_loss += F.cross_entropy(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct

def test_represent(model, device, train_loader,test_loader,threshold_):
    model.eval()
    test_loss = 0
    correct = 0
    threshold_ = threshold_
    represent_ = torch.zeros((6,dimension_)).cuda()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            #onehot = torch.nn.functional.one_hot(target, 10)
            #print(onehot.shape)
            output = model(data)
            #print(output.shape,target.shape)
            
            for i in range(target.shape[0]):
                represent_[int(target[i])]+=output[i]

        my_ones = torch.ones(represent_.shape[0],represent_.shape[1]).cuda()
        my_zeros = -1*torch.ones(represent_.shape[0],represent_.shape[1]).cuda()
        print(torch.max(represent_))
        represent_ = torch.where(represent_<=threshold_, my_zeros, represent_)
        represent_ = torch.where(represent_>=threshold_ , my_ones, represent_)

            #represent_final = torch.sign(represent_)
        print("finishing encoding")
        count = 0
        for data, target in test_loader:
            print(data.shape)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_minones = -1* torch.ones(output.shape[0],output.shape[1]).cuda()
            output = torch.where(output==0 , output_minones, output)
            testing_inference = torch.mm(output,represent_.T)
            out = torch.argmax(testing_inference, dim=1)
            count+=torch.sum(torch.eq(out,target))
        print(count / len(test_loader.dataset))
        # print(count)
        # print(len(test_loader.dataset))

    # test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=5120, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=2, metavar='LR',
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
    parser.add_argument('--raw-data-dir', type=str, default='./Hyperdimensional-Computing/dataset')
                        
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

    
    class HDDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return self.data.size(0)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    def quantize(data, precision=8):
    # assume min and max of the data is -1 and 1
        # assume min and max of the data is -1 and 1
        scaling_factor = 2 ** (precision - 1) - 1
        data = torch.round(data * scaling_factor)
    # shift the quantized data to positive and rescale to [0, 1.0]
        data =  (data + scaling_factor) / 255.0
        level1=1.5
        x1 =  torch.clamp(torch.div(torch.floor(torch.mul(data, level1)), level1),min=0, max=1/level1)
        x1 = x1*level1
        level2=2
        x2 =  torch.clamp(torch.div(torch.floor(torch.mul(data, level2)), level2),min=0, max=1/level2)
        x2 = x2*level2

        level3=4
        x3 =  torch.clamp(torch.div(torch.floor(torch.mul(data, level3)), level3),min=0, max=1/level3)
        x3 = x3*level3

        # level 4 is only used by d=128, if used, the input size should be 2244 instead of 1683
        level4=6
        x4 =  torch.clamp(torch.div(torch.floor(torch.mul(data, level4)), level4),min=0, max=1/level4)
        x4 = x4*level4
        x = torch.cat((x1,x2,x3,x4),1)
        return x
    
    x_train_path = f'./{args.raw_data_dir}/ucihar/train/x_train.txt'
    y_train_path = f'./{args.raw_data_dir}/ucihar/train/y_train.txt'
    x_test_path = f'./{args.raw_data_dir}/ucihar/test/x_test.txt'
    y_test_path = f'./{args.raw_data_dir}/ucihar/test/y_test.txt'

    def load_data(feature_file_path, label_file_path):
        # load training features from txt
        x_train = open(feature_file_path, 'r')
        x_train = x_train.readlines()
        for idx in range(len(x_train)):
            x_train[idx] = x_train[idx].split()
        x_train = np.array(x_train, dtype=np.float32)
        # load test features from txt
        y_train = open(label_file_path, 'r')
        y_train = y_train.readlines()
        y_train = np.array(y_train, dtype=np.int32) - 1
        return x_train, y_train

    x_train, y_train = load_data(x_train_path, y_train_path)
    x_test, y_test = load_data(x_test_path, y_test_path)
    x_train, y_train = quantize(torch.tensor(x_train), precision=8).unsqueeze(1), torch.tensor(y_train).long()
    x_test, y_test = quantize(torch.tensor(x_test), precision=8).unsqueeze(1), torch.tensor(y_test).long()
    trainset = HDDataset(x_train, y_train)
    testset = HDDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size = args.batch_size)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 10000)
    #print(test_loader[0])
    #snn_loader = torch.utils.data.DataLoader(snn_dataset, **kwargs)

    model = Net().to(device)
    if load_True:
        model.load_state_dict(torch.load(load_), strict=False)

    #snn_model = CatNet(args.T).to(device)
    #model.load_state_dict(torch.load("dimension_4.pt"), strict=False)

    if args.resume != None:
        load_model(torch.load(args.resume), model)
    for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)

    # ACC = 0
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     ACC_ = test(model, device, test_loader)
    #     if ACC_>ACC or ACC_ == ACC:
    #         ACC = ACC_
    #         torch.save(model.state_dict(), save_)
        
    #     scheduler.step()
    # print('save', ACC)
    model_plain = Net_plain().to(device)
    test_pass(model, device, test_loader)
    #x = model(torch.tensor([0,1]))
    torch.save(model.state_dict(), "merged_bn.pt")
    model_plain.load_state_dict(torch.load("merged_bn.pt"), strict=False)
    #load_model(model.state_dict(),model_plain)

    if test_hdc:
        repre = test_represent(model_plain, device, train_loader,test_loader,600)
        repre = test_represent(model_plain, device, train_loader,test_loader,685)
        repre = test_represent(model_plain, device, train_loader,test_loader,690)
        repre = test_represent(model_plain, device, train_loader,test_loader,695)
        repre = test_represent(model_plain, device, train_loader,test_loader,300)
        repre = test_represent(model_plain, device, train_loader,test_loader,310)    
        
    else:
        test(model, device, test_loader)




if __name__ == '__main__':
    main()
