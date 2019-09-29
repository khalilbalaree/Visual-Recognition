import torch
from torch import nn, optim
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

epochs = 10
batch_size_train = 128
batch_size_test = 1000
learining_rate = 1e-3
lambda_val_MNIST = 3e-5
lambda_val_CIFAR10 = 3e-5


class logistic_model(nn.Module):
    
  def __init__(self, dim, n_class):
    super(logistic_model, self).__init__()
    self.Linear = nn.Linear(dim, n_class)
    
  def forward(self, x):
    x = x.view(x.size(0), -1)
    x= self.Linear(x)
    # The sigmoid function is used for the two-class logistic regression, 
    # whereas the softmax function is used for the multiclass logistic regression
    x = F.softmax(x)
    return x

def logistic_regression(dataset_name):

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if dataset_name == "MNIST":
        MNIST_training = datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
        
        MNIST_test = datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
        
        MNIST_training_dataset, MNIST_validation_dataset = random_split(MNIST_training, [55000, 5000])

        train_loader  = DataLoader(MNIST_training_dataset, batch_size=batch_size_train, shuffle=True)
        validation_loader  = DataLoader(MNIST_validation_dataset, batch_size=batch_size_train, shuffle=True)
        test_loader = DataLoader(MNIST_test, batch_size=batch_size_test, shuffle=True)

        # images.shape = [batch_size,channel=1,height=28,width=28]
        model = logistic_model(28 * 28, 10).to(device)
        criterion = nn.CrossEntropyLoss()
        # adam better than sgd
        optimizer = optim.Adam(model.parameters(), lr=learining_rate, weight_decay=lambda_val_MNIST)
      
    elif dataset_name == "CIFAR10":
        CIFAR10_training = datasets.CIFAR10("/CIFAR10_dataset/",train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        
        CIFAR10_test = datasets.CIFAR10("/CIFAR10_dataset/",train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        
        CIFAR10_training_dataset, CIFAR10_validation_dataset = random_split(CIFAR10_training, [40000, 10000])

        train_loader  = DataLoader(CIFAR10_training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)
        validation_loader  = DataLoader(CIFAR10_validation_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)
        test_loader = DataLoader(CIFAR10_test, batch_size=batch_size_test, shuffle=True, num_workers=2)

        # images.shape = [batch_size,channel=3,height=32,width=32]
        model = logistic_model(32 * 32 * 3, 10).to(device)
        criterion = nn.CrossEntropyLoss()
        # adam better than sgd
        optimizer = optim.Adam(model.parameters(), lr=learining_rate, weight_decay=lambda_val_CIFAR10)
        

    # shape of data 
    examples = enumerate(test_loader)
    _, (example_data, _) = next(examples)
    print(example_data.shape)

    # train 
    for epoch in range(1, epochs + 1):    
        model.train()
        for _, (images, labels) in enumerate(train_loader): 
            images = images.to(device) 
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            # l2_reg = 0
            # for param in model.parameters():
            #     l2_reg += torch.norm(param)
            # loss = criterion(output, labels) + lambda_val * l2_reg
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # validation 
        if epoch%5== 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    output = model(images)
                    _, pred = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

            accuracy = 100. * correct / total
            print("{}. epoch: {}. Accuracy: {}.".format(dataset_name, epoch, accuracy))

    # test 
    model.eval()
    out1 = []
    out2 = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, pred = torch.max(output.data, 1)  
            pred = pred.cpu().numpy()
            labels = labels.cpu().numpy()
            out1.append(pred)
            out2.append(labels)

    return torch.tensor(out1), torch.tensor(out2)