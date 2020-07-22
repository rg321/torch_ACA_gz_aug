
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
# from tensorboardX import SummaryWriter
import math
import sys
import os
import shutil
from torch_ACA import odesolve_endtime as odesolve

from data_helper import get_galaxyZoo_loaders

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 60:
        optim_factor = 2
    elif epoch > 30:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))


parser = argparse.ArgumentParser()
parser.add_argument('--network', type = str, choices = ['resnet', 'sqnxt'], default = 'resnet')
parser.add_argument('--method', type = str, choices=['Euler', 'RK2', 'RK4','RK23','RK45','RK12','Dopri5'], default = 'RK12')
parser.add_argument('--num_epochs', type = int, default = 50)
parser.add_argument('--start_epoch', type = int, default = 0)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--h', type=float, default = None, help='Initial Stepsize')
parser.add_argument('--t0', type=float, default = 0.0, help='Initial time')
parser.add_argument('--t1', type=float, default = 1.0, help='End time')
parser.add_argument('--rtol', type=float, default = 1e-2, help='Releative tolerance')
parser.add_argument('--atol', type=float, default = 1e-2, help='Absolute tolerance')
parser.add_argument('--print_neval', type=bool, default = False, help='Print number of evaluation or not')
parser.add_argument('--neval_max', type=int, default = 50000, help='Maximum number of evaluation in integration')

parser.add_argument('--batch_size', type = int, default = 20)
parser.add_argument('--test_batch_size', type = int, default = 10)
parser.add_argument('--dataset', type = str, choices = ['CIFAR10', 'GalaxyZoo', 'MTVSO'], default = 'CIFAR10')
args = parser.parse_args()
if args.network == 'sqnxt':
    from cifar_classification.models.sqnxt import SqNxt_23_1x
    # writer = SummaryWriter('sqnxt/'+args.network +'_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')
elif args.network == 'resnet':
    from cifar_classification.models.resnet import ResNet18
    # writer = SummaryWriter('resnet/'+args.network+'_'+ args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')


num_epochs = int(args.num_epochs)
lr           = float(args.lr)
start_epoch  = int(args.start_epoch)
batch_size   = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'method':args.method})
        self.options.update({'h': args.h})
        self.options.update({'t0': args.t0})
        self.options.update({'t1': args.t1})
        self.options.update({'rtol': args.rtol})
        self.options.update({'atol': args.atol})
        self.options.update({'print_neval': args.print_neval})
        self.options.update({'neval_max': args.neval_max})
        print(self.options)

    def forward(self, x):
        out = odesolve(self.odefunc, x, self.options, regenerate_graph = False)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        

# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# train_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_train, train = True, download = True)
# test_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_test, train = False, download = True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, num_workers = 4, shuffle = False)

train_loader, test_loader, train_dataset = get_galaxyZoo_loaders(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

if args.dataset == 'MTVSO':
    num_classes = 20
else:
    num_classes = 10

if args.network == 'sqnxt':
    net = SqNxt_23_1x(num_classes, ODEBlock)
elif args.network == 'resnet':
    net = ResNet18(ODEBlock, num_classes=num_classes)

net.apply(conv_init)
print(net)
if is_use_cuda:
    net.cuda()#to(device)
    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer  = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 5e-4)

def train(epoch):
    net.train()
    train_loss = 0
    correct0    = 0
    correct1    = 0
    correct2    = 0
    total      = 0
    
    print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), [l.cuda() for l in labels]
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            outputs = net(inputs)
            # loss = criterion(outputs, labels)
            loss0 = criterion(outputs[0], labels[0])
            loss1 = criterion(outputs[1], labels[1])
            loss2 = criterion(outputs[2], labels[2])
            loss = loss0+loss1+loss2
            loss.backward()

        optimizer.step()
        # writer.add_scalar('Train/Loss', loss.item(), epoch* 50000 + batch_size * (idx + 1)  )
        train_loss += loss.item()
        _, predict0 = torch.max(outputs[0], 1)
        _, predict1 = torch.max(outputs[1], 1)
        _, predict2 = torch.max(outputs[2], 1)
        total += labels[0].size(0)
        # total += labels[1].size(0)
        # total += labels[2].size(0)
        correct0 += predict0.eq(labels[0]).cpu().sum().double()
        correct1 += predict1.eq(labels[1]).cpu().sum().double()
        correct2 += predict2.eq(labels[2]).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Loss: %.4f Acc0: %.3f Acc1: %.3f Acc2: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, 
                           # idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), 
                          correct0 / total, correct1 / total, correct2 / total))
        sys.stdout.flush()
    # writer.add_scalar('Train/Accuracy', correct / total, epoch )
        
def test(epoch):
    net.eval()
    test_loss = 0
    correct0 = 0
    correct1 = 0
    correct2 = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), [l.cuda() for l in labels]
        outputs = net(inputs)
        # loss = criterion(outputs, labels)
        loss0 = criterion(outputs[0], labels[0])
        loss1 = criterion(outputs[1], labels[1])
        loss2 = criterion(outputs[2], labels[2])
        loss = loss0+loss1+loss2
        
        test_loss  += loss.item()
        _, predict0 = torch.max(outputs[0], 1)
        _, predict1 = torch.max(outputs[1], 1)
        _, predict2 = torch.max(outputs[2], 1)
        total += labels[0].size(0)
        # total += labels[1].size(0)
        # total += labels[2].size(0)
        correct0 += predict0.eq(labels[0]).cpu().sum().double()
        correct1 += predict1.eq(labels[1]).cpu().sum().double()
        correct2 += predict2.eq(labels[2]).cpu().sum().double()
        # writer.add_scalar('Test/Loss', loss.item(), epoch* 50000 + test_loader.batch_size * (idx + 1)  )
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch  [%d/%d] Loss: %.4f Acc0: %.3f Acc1: %.3f Acc2: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, test_loss / (100 * (idx + 1)),
                           correct0 / total, correct1 / total, correct2 / total))
        sys.stdout.flush()

    acc = [correct0 / total, correct1 / total, correct2 /total]
    # writer.add_scalar('Test/Accuracy', acc, epoch )
    return acc

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

best_acc0 = 0.0
best_acc1 = 0.0
best_acc2 = 0.0
best_total = 0.0

if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc0 = checkpoint['best_acc0']
        best_acc1 = checkpoint['best_acc1']
        best_acc2 = checkpoint['best_acc2']
        best_total = checkpoint['best_total']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()

    _lr = lr_schedule(args.lr, _epoch)
    adjust_learning_rate(optimizer, _lr)

    train(_epoch)
    print()
    test_acc = test(_epoch)
    print()
    print()
    end_time   = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

    # save model
    is_best = sum(test_acc) > best_total
    # is_best0 = test_acc[0] > best_acc0
    # is_best1 = test_acc[1] > best_acc1
    # is_best2 = test_acc[2] > best_acc2
    best_acc0 = max(test_acc[0], best_acc0)
    best_acc1 = max(test_acc[1], best_acc1)
    best_acc2 = max(test_acc[2], best_acc2)
    save_checkpoint({
        'epoch': _epoch + 1,
        'state_dict': net.state_dict(),
        'acc0': test_acc[0],
        'acc1': test_acc[1],
        'acc2': test_acc[2],
        'best_acc0': best_acc0,
        'best_acc1': best_acc1,
        'best_acc2': best_acc2,
        'best_total': best_total,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=args.checkpoint+'_'+args.method+'_'+args.network)

print('Best Acc0: %.4f' % (best_acc0 * 100))
print('Best Acc1: %.4f' % (best_acc1 * 100))
print('Best Acc2: %.4f' % (best_acc2 * 100))
# writer.close()
