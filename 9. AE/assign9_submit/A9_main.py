import os
import cv2
import sys
import re
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets

from A9_modules import Encoder, Decoder, Classifier
from A9_modules import CompositeLoss
from A9_modules import MNISTParams, FMNISTParams


class A9_Params:
    """

    :ivar dataset:
        0: MNIST
        1: FMNIST
    """

    def __init__(self):
        self.use_cuda = 1
        self.dataset = 0
        self.train_split = 0.8
        self.labeled_split = 0.2
        self.mnist = MNISTParams()
        self.fmnist = FMNISTParams()


class PartiallyLabeled(Dataset):
    """
    :param Dataset _dataset:
    """

    def __init__(self, dataset, all_idx, labeled_percent):
        """

        :param Dataset dataset:
        :param all_idx:
        :param float labeled_percent:
        """
        self._dataset = dataset
        self._n_data = len(all_idx)
        self.n_labeled_data = int(labeled_percent * self._n_data)
        self._is_labeled = np.zeros((self._n_data, 1), dtype=np.bool)
        labeled_images = np.random.permutation(all_idx)[:self.n_labeled_data]
        self._is_labeled[labeled_images] = 1

    def __len__(self):
        return self._dataset.__len__()

    def __getitem__(self, idx):
        assert idx < self._n_data, "Invalid idx: {} for _n_data: {}".format(idx, self._n_data)

        input, target = self._dataset.__getitem__(idx)
        is_labeled = self._is_labeled[idx]
        return input, target, is_labeled


def get_psnr(x, x_test):
    mse = np.mean((np.reshape(x_test, [-1, 28, 28]) - np.reshape(x, [-1, 28, 28])) ** 2)
    psnr = -100.0 * np.log10(mse)
    return psnr


def eval(modules, data_loader, criteria, vis, device):
    modules.eval()

    encoder, decoder, classifier = modules
    criterion_rec, criterion_cls = criteria
    mean_loss_sum = 0
    _psnr_sum = 0
    total = 0
    correct = 0

    n_batches = 0
    _pause = 1

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs_enc = encoder(inputs)
            outputs_rec = decoder(outputs_enc)
            outputs_cls = classifier(outputs_enc)

            loss_rec = criterion_rec(outputs_rec, inputs)
            loss_cls = criterion_cls(outputs_cls, targets)

            loss = loss_rec + loss_cls

            mean_loss = loss.item()

            mean_loss_sum += mean_loss

            _, predicted = outputs_cls.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            n_batches += 1

            inputs_np = inputs.detach().cpu().numpy()
            outputs_np = outputs_rec.detach().cpu().numpy()

            _psnr = get_psnr(inputs_np, outputs_np)
            _psnr_sum += _psnr

            if vis:
                comcat_imgs = []
                for i in range(data_loader.batch_size):
                    input_img = inputs_np[i, ...].squeeze()
                    output_img = outputs_np[i, ...].squeeze()

                    comcat_img = np.concatenate((input_img, output_img), axis=1)
                    comcat_imgs.append(comcat_img)

                comcat_imgs = np.concatenate(comcat_imgs, axis=0)
                cv2.imshow('comcat_imgs', comcat_imgs)
                k = cv2.waitKey(1 - _pause)
                if k == 27:
                    sys.exit(0)
                elif k == ord('q'):
                    vis = 0
                    cv2.destroyWindow('comcat_imgs')
                    break
                elif k == 32:
                    _pause = 1 - _pause

    overall_mean_loss = mean_loss_sum / n_batches
    mean_psnr = _psnr_sum / n_batches
    acc = 100. * correct / total

    if vis:
        cv2.destroyWindow('comcat_imgs')

    return overall_mean_loss, acc, mean_psnr


def main():
    params = A9_Params()

    # optional command line argument parsing
    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    # init device
    if params.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Training on CPU')

    # load dataset
    if params.dataset == 0:
        print('Using MNIST dataset')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        ])
        train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('data', train=False, download=True, transform=transform)
        valid_set = datasets.MNIST('data', train=True, download=True, transform=transform)
        train_params = params.mnist
    elif params.dataset == 1:
        print('Using Fashion MNIST dataset')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        train_set = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        valid_set = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        train_params = params.fmnist
    else:
        raise IOError('Invalid db_type: {}'.format(params.dataset))

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(params.train_split * num_train))

    train_idx, valid_idx = indices[:split], indices[split:]
    train_set = PartiallyLabeled(train_set, train_idx, labeled_percent=params.labeled_split)

    print('Training samples: {}\n'
          'Validation samples: {}\n'
          'Labeled training samples: {}'
          ''.format(
        len(train_idx),
        len(valid_idx),
        train_set.n_labeled_data
    ))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_params.batch_size, sampler=train_sampler,
                                                   num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=24, sampler=valid_sampler,
                                                   num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, num_workers=4)

    # create modules
    encoder = Encoder(device).to(device)
    decoder = Decoder(device).to(device)
    classifier = Classifier(device).to(device)

    assert isinstance(encoder, nn.Module), 'encoder must be an instance of nn.Module'
    assert isinstance(decoder, nn.Module), 'decoder must be an instance of nn.Module'
    assert isinstance(classifier, nn.Module), 'classifier must be an instance of nn.Module'

    modules = nn.ModuleList((encoder, decoder, classifier))

    # init weights
    encoder.init_weights()
    decoder.init_weights(encoder.get_weights())
    classifier.init_weights()

    # create losses
    criterion_rec = torch.nn.MSELoss().to(device)
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)

    parameters = list(modules.parameters())
    if train_params.c0 == 0:
        composite_loss = CompositeLoss(device)
        composite_loss.init_weights()
        assert isinstance(composite_loss, nn.Module), 'composite_loss must be an instance of nn.Module'
        parameters += list(composite_loss.parameters())
    else:
        def composite_loss(x, y):
            return x + train_params.c0 * y

    # create optimizer
    if train_params.optim_type == 0:
        optimizer = torch.optim.SGD(parameters, lr=train_params.lr, momentum=train_params.momentum,
                                    weight_decay=train_params.weight_decay)
    elif train_params.optim_type == 1:
        optimizer = torch.optim.Adam(parameters, lr=train_params.lr, weight_decay=train_params.weight_decay)
    else:
        raise IOError('Invalid optim_type: {}'.format(train_params.optim_type))

    weights_dir = os.path.dirname(train_params.weights_path)
    weights_name = os.path.basename(train_params.weights_path)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    start_epoch = 0
    max_valid_acc_epoch = 0
    max_valid_acc = 0
    max_train_acc = 0
    min_valid_loss = np.inf
    min_train_loss = np.inf

    # load weights
    if train_params.load_weights:
        matching_ckpts = [k for k in os.listdir(weights_dir) if
                          os.path.isfile(os.path.join(weights_dir, k)) and
                          k.startswith(weights_name)]
        if not matching_ckpts:
            msg = 'No checkpoints found matching {} in {}'.format(weights_name, weights_dir)
            if train_params.load_weights == 1:
                raise IOError(msg)
            print(msg)
        else:
            matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

            weights_path = os.path.join(weights_dir, matching_ckpts[-1])

            chkpt = torch.load(weights_path, map_location=device)  # load checkpoint

            print('Loading weights from: {} with:\n'
                  '\tepoch: {}\n'
                  '\ttrain_loss: {}\n'
                  '\ttrain_acc: {}\n'
                  '\tvalid_loss: {}\n'
                  '\tvalid_acc: {}\n'
                  '\ttimestamp: {}\n'.format(
                weights_path, chkpt['epoch'],
                chkpt['train_loss'], chkpt['train_acc'],
                chkpt['valid_loss'], chkpt['valid_acc'],
                chkpt['timestamp']))

            encoder.load_state_dict(chkpt['encoder'])
            decoder.load_state_dict(chkpt['decoder'])
            classifier.load_state_dict(chkpt['classifier'])
            optimizer.load_state_dict(chkpt['optimizer'])

            if train_params.c0 == 0 and 'composite_loss' in chkpt:
                composite_loss.load_state_dict(chkpt['composite_loss'])

            max_valid_acc = chkpt['valid_acc']
            min_valid_loss = chkpt['valid_loss']

            max_train_acc = chkpt['train_acc']
            min_train_loss = chkpt['train_loss']

            max_valid_acc_epoch = chkpt['epoch']
            start_epoch = chkpt['epoch'] + 1

    if train_params.load_weights != 1:
        # continue training
        for epoch in range(start_epoch, train_params.n_epochs):
            # Training
            modules.train()

            train_loss_rec = 0
            train_loss_cls = 0
            train_loss = 0
            train_total = 0
            train_correct = 0
            batch_idx = 0

            save_weights = 0

            for batch_idx, (inputs, targets, is_labeled) in tqdm(enumerate(train_dataloader)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                if not np.count_nonzero(is_labeled.detach().numpy()):
                    continue

                is_labeled = is_labeled.squeeze().to(device)

                optimizer.zero_grad()

                outputs_enc = encoder(inputs)
                outputs_rec = decoder(outputs_enc)
                outputs_cls = classifier(outputs_enc)

                loss_rec = criterion_rec(outputs_rec, inputs)
                loss_cls = criterion_cls(outputs_cls[is_labeled, :], targets[is_labeled])

                loss = composite_loss(loss_rec, loss_cls)

                mean_loss_rec = loss_rec.item()
                mean_loss_cls = loss_cls.item()
                train_loss_rec += mean_loss_rec
                train_loss_cls += mean_loss_cls

                loss.backward()
                optimizer.step()

                mean_loss = loss.item()
                train_loss += mean_loss

                _, predicted = outputs_cls.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            mean_train_loss_rec = train_loss_rec / (batch_idx + 1)
            mean_train_loss_cls = train_loss_cls / (batch_idx + 1)
            mean_train_loss = train_loss / (batch_idx + 1)

            train_acc = 100. * train_correct / train_total

            valid_loss, valid_acc, valid_psnr = eval(
                modules, valid_dataloader, (criterion_rec, criterion_cls), train_params.vis, device)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                max_valid_acc_epoch = epoch
                if train_params.save_criterion == 0:
                    save_weights = 1

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                if train_params.save_criterion == 1:
                    save_weights = 1

            if train_acc > max_train_acc:
                max_train_acc = train_acc
                if train_params.save_criterion == 2:
                    save_weights = 1

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                if train_params.save_criterion == 3:
                    save_weights = 1

            print(
                'Epoch: %d Train-Loss: %.6f (rec: %.6f, cls: %.6f) | Train-Acc: %.3f%% | '
                'Validation-Loss: %.6f | Validation-Acc: %.3f%% | Validation-PSNR: %.3f | '
                'Max Validation-Acc: %.3f%% (epoch: %d)' % (
                    epoch, mean_train_loss, mean_train_loss_rec, mean_train_loss_cls, train_acc,
                    valid_loss, valid_acc, valid_psnr, max_valid_acc, max_valid_acc_epoch))

            # Save checkpoint.
            if save_weights:
                model_dict = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': mean_train_loss,
                    'train_acc': train_acc,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'epoch': epoch,
                    'timestamp': datetime.now().strftime("%y/%m/%d %H:%M:%S"),
                }
                if train_params.c0 == 0:
                    model_dict['composite_loss'] = composite_loss.state_dict()

                weights_path = '{}.{:d}'.format(train_params.weights_path, epoch)
                print('Saving weights to {}'.format(weights_path))
                torch.save(model_dict, weights_path)

    print('Testing...')
    start_t = time.time()
    test_loss, test_acc, test_psnr = eval(
        modules, test_dataloader, (criterion_rec, criterion_cls), train_params.vis, device)
    end_t = time.time()
    test_time = end_t - start_t

    print('Test-Loss: %.6f | Test-Acc: %.3f%% | Test-PSNR: %.3f%% | Test-Time: %.3f sec' % (
        test_loss, test_acc, test_psnr, test_time))


if __name__ == '__main__':
    main()
