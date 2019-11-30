import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import os

class TrainParams:
    def __init__(self):
        self.batch_size = 100
        self.n_epochs = 30
        self.c0 = 1
        self.load_weights = 1
        self.root_path = ''
        self.weights_path = os.path.join(self.root_path, './checkpoints/model.pt')


def adjust_lr_classifier(optimizer, epoch):
    lr = 0.01 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1))
        self.layer.add_module("Bn", nn.BatchNorm2d(out_channels))

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential()
            self.skip.add_module("Conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1))
            self.skip.add_module("Bn", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.layer(x)
        out += self.skip(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block):
        super(ResNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1))
        self.layer1.add_module("Bn", nn.BatchNorm2d(64))
        self.layer1.add_module("Relu", nn.ReLU(True))

        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.layer2 = nn.Sequential(
            block(64, 64, 1),
            block(64, 64, 1),
        )

        self.layer4 = nn.Sequential(
            block(64, 128, 2),
            block(128, 128, 1),
        )

        self.layer5 = nn.Sequential(
            block(128, 256, 2),
            block(256, 256, 1),
        )

        self.layer6 = nn.Sequential(
            block(256, 512, 2),
            block(512, 512, 1),
        )

        self.layer7 = nn.Sequential(
            block(512, 1024, 2),
            block(1024, 1024, 1),
        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 37*4+20),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        cls = x[:,0:20]
        bb1 = x[:,20:]
        return cls.view(-1, 10, 2), bb1.view(-1, 4, 37)


def train(device, param):
    root_dir = param.root_path
    x_train = torch.Tensor(np.load(os.path.join(root_dir,'train_X.npy')))
    x_train = x_train.view(55000, 1, 64, 64)
    x_valid = torch.Tensor(np.load(os.path.join(root_dir,'valid_X.npy')))
    x_valid = x_valid.view(5000, 1, 64, 64)
    y_train = torch.Tensor(np.load(os.path.join(root_dir,'train_Y.npy'))).type(torch.LongTensor)
    y_valid = torch.Tensor(np.load(os.path.join(root_dir,'valid_Y.npy'))).type(torch.LongTensor)
    train_bboxes = torch.Tensor(np.load(os.path.join(root_dir,'train_bboxes.npy'))).type(torch.LongTensor)
    valid_bboxes = torch.Tensor(np.load(os.path.join(root_dir,'valid_bboxes.npy'))).type(torch.LongTensor)

    train_dataset = utils.TensorDataset(x_train, y_train, train_bboxes)
    train_loader = utils.DataLoader(train_dataset, batch_size=param.batch_size)

    valid_dataset = utils.TensorDataset(x_valid, y_valid, valid_bboxes)
    valid_loader = utils.DataLoader(valid_dataset, batch_size=param.batch_size)

    model = ResNet(BasicBlock).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.97, weight_decay=0.001)

    for epoch in range(1, param.n_epochs+1):
        model.train()
        adjust_lr_classifier(optimizer, epoch)
        for _, (images, labels, bbs) in enumerate(train_loader): 
            images = images.to(device)
            labels = labels.to(device)
            bbs = bbs.to(device)
            optimizer.zero_grad()
            cls, bb1 = model(images)
            loss1 = criterion(torch.sort(cls)[0], labels)
            loss2 = criterion(bb1[:,0,:], bbs[:,0,0])
            loss3 = criterion(bb1[:,1,:], bbs[:,0,1])
            loss4 = criterion(bb1[:,2,:], bbs[:,1,0])
            loss5 = criterion(bb1[:,3,:], bbs[:,1,1])
            loss = loss1 + param.c0 * (loss2 + loss3 + loss4 + loss5)
            loss.backward()
            optimizer.step()
        
        # validation 
        if epoch%5 == 0:
            model.eval()
            correct = 0
            total = 0
            pred_bb = []
            truth_bb = []
            with torch.no_grad():
                for images, labels, bbs in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    cls, bb1 = model(images)
                    cls = torch.sort(cls)[0]
                    _, pred = torch.max(cls.data, 1)
                    c = (pred == labels).sum(1)
                    cc = c == 2
                    correct += cc.sum()
                    total += labels.size(0)

                    bb1_pred = torch.max(bb1[:,0,:].data, 1)[1]
                    bb2_pred = torch.max(bb1[:,1,:].data, 1)[1]
                    bb3_pred = torch.max(bb1[:,2,:].data, 1)[1]
                    bb4_pred = torch.max(bb1[:,3,:].data, 1)[1]

                    for i in range(param.batch_size):
                        this_bb = np.empty((2, 4))
                        this_bb[0][0] = bb1_pred[i]
                        this_bb[0][1] = bb2_pred[i]
                        this_bb[0][2] = bb1_pred[i]+28
                        this_bb[0][3] = bb2_pred[i]+28
                        this_bb[1][0] = bb3_pred[i]
                        this_bb[1][1] = bb4_pred[i]
                        this_bb[1][2] = bb3_pred[i]+28
                        this_bb[1][3] = bb4_pred[i]+28
                        pred_bb.append(this_bb)
                        truth_bb.append(np.array(bbs[i]))

            accuracy = 100. * correct / total
            # iou = compute_iou(np.array(pred_bb), np.array(truth_bb))
            # just 0 here
            iou = 0.0
            print("epoch: {}. Accuracy: {:.4f}. iou: {:.4f}.".format(epoch, accuracy, iou))

    return model


def classify_and_detect(images):
    """

    :param np.ndarray images: N x 4096 array containing N 64x64 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """

    N = images.shape[0]
    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes
    param = TrainParams()

    if param.load_weights:
        print("loading...")
        model = ResNet(BasicBlock)
        model.load_state_dict(torch.load(param.weights_path, map_location=torch.device('cpu')))
    else:
        print("training ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        model = train(device, param)
        print("Saving...") 
        torch.save(model.state_dict(), param.weights_path)
        model.to(torch.device("cpu"))

    print("testing...")
    model.eval()
    with torch.no_grad():
      for i, image in enumerate(images):
          image = torch.Tensor(image).view(1, 1, 64, 64)
          cls, bb1 = model(image)
          cls = torch.sort(cls)[0]
          pred_cls = torch.max(cls.data, 1)[1]
          pred_class[i] = pred_cls.cpu().detach().numpy()
          
          bb1_pred = torch.max(bb1[:,0,:].data, 1)[1]
          bb2_pred = torch.max(bb1[:,1,:].data, 1)[1]
          bb3_pred = torch.max(bb1[:,2,:].data, 1)[1]
          bb4_pred = torch.max(bb1[:,3,:].data, 1)[1]

          this_bb = np.empty((2, 4))
          this_bb[0][0] = bb1_pred
          this_bb[0][1] = bb2_pred
          this_bb[0][2] = bb1_pred+28
          this_bb[0][3] = bb2_pred+28
          this_bb[1][0] = bb3_pred
          this_bb[1][1] = bb4_pred
          this_bb[1][2] = bb3_pred+28
          this_bb[1][3] = bb4_pred+28
          pred_bboxes[i] = this_bb

    return pred_class, pred_bboxes