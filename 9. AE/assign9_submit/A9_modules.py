import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TrainParams:
    """
    :ivar optim_type: 0: SGD, 1: ADAM

    :ivar load_weights:
        0: train from scratch,
        1: load and test
        2: load if it exists and continue training

    :ivar save_criterion:  when to save a new checkpoint
        0: max validation accuracy
        1: min validation loss,
        2: max training accuracy
        3: min training loss

    :ivar vis: visualize the input and reconstructed images during validation and testing
    """

    def __init__(self):
        self.batch_size = 128
        self.optim_type = 0
        self.lr = 0.000001
        self.momentum = 0.9
        self.n_epochs = 67
        self.weight_decay = 0.1
        self.c0 = 1
        self.save_criterion = 0
        self.load_weights = 2
        self.weights_path = './checkpoints/model.pt'
        self.vis = 0


class MNISTParams(TrainParams):
    def __init__(self):
        super(MNISTParams, self).__init__()
        self.weights_path = './checkpoints/mnist/model.pt'


class FMNISTParams(TrainParams):
    def __init__(self):
        super(FMNISTParams, self).__init__()
        self.weights_path = './checkpoints/fmnist/model.pt'


class CompositeLoss(nn.Module):
    def __init__(self, device):
        super(CompositeLoss, self).__init__()
        # self.weight = nn.Parameter(torch.Tensor(1)).to(device)
        self.linear = nn.Linear(1, 1)
        self.relu = nn.ReLU(True)

    def init_weights(self):
        # self.weight.data.fill_(0.01)
        pass

    def forward(self, reconstruction_loss, classification_loss):
        t = torch.Tensor([classification_loss]).to(device)
        loss = reconstruction_loss + self.relu(self.linear(t))
        return loss


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 300)
        self.linear2 = nn.Linear(300, 148)
        self.relu = nn.ReLU(True)
        self.weight1 = self.linear1.weight
        self.weight2 = self.linear2.weight
        self.bias1 = self.linear1.bias
        self.bias2 = self.linear2.bias

    def get_weights(self):
        return [self.weight1, self.weight2]

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        self.bias1.data.fill_(0.01)
        self.bias2.data.fill_(0.01)

    def forward(self, enc_input):
        x = enc_input.view(-1, 784)
        x = self.relu(F.linear(x, self.weight1, self.bias1))
        x = self.relu(F.linear(x, self.weight2, self.bias2))
        return x


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()         
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.bias1 = nn.Linear(120, 300).bias
        self.bias2 = nn.Linear(300, 784).bias

    def init_weights(self, shared_weights):
        self.weight1 = shared_weights[1].t()
        self.weight2 = shared_weights[0].t()
        self.bias1.data.fill_(0.01)
        self.bias2.data.fill_(0.01)
        
    def forward(self, dec_input):
        x = self.relu(F.linear(dec_input, self.weight1, self.bias1))
        x = self.sigmoid(F.linear(x, self.weight2, self.bias2))
        return x.view(-1, 1, 28, 28)


class Classifier(nn.Module):
    def __init__(self, device):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(148, 32)
        self.linear2 = nn.Linear(32, 10)
        self.relu = nn.ReLU(True)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x