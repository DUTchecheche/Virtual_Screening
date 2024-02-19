# -*- coding: utf-8 -*-
# Requires a computer with at least 32G RAM
# The code was tested in a winodws11 system with the following conda environment:
# python 3.7
# torch 1.13.1
# numpy 1.21.6
# matplotlib 3.3.2
# seaborn 0.11.1
# scikit-learn 0.24.2

import os
import re
import random
import torch
import torch.nn.functional as F
import torch.optim as op
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from pylab import mpl


def org_data(targets_path):
    all_actives = []
    all_decoys = []
    all_inactives = []
    targets = os.listdir(targets_path)
    for target in targets:
        index = None
        actives = []
        inactives = []
        file = open(targets_path + os.sep + target + '\\actives_adj_matrix.txt', 'r')
        for line in file:
            if line.startswith('#'):
                actives.append([])
                inactives.append([])
                index = int(re.search(' \d+ ', line)[0]) - 1
            else:
                actives[index].extend(list(map(int, line.strip().split(' '))))
                inactives[index].extend(list(map(int, line.strip().split(' '))))
                for i in range(46 - len(line) // 2):
                    actives[index].append(0)
                    inactives[index].append(0)
        for active in actives:
            for j in range(46 - len(active) // 46):
                for k in range(46):
                    active.append(0)
        for inactive in inactives:
            for j in range(46 - len(inactive) // 46):
                for k in range(46):
                    inactive.append(0)
        file.close()

        index = None
        decoys = []
        file = open(targets_path + os.sep + target + '\\decoys_adj_matrix_used_in_this_work.txt', 'r')
        for line in file:
            if line.startswith('#'):
                decoys.append([])
                index = int(re.search(' \d+ ', line)[0]) - 1
            else:
                decoys[index].extend(list(map(int, line.strip().split(' '))))
                for i in range(46 - len(line) // 2):
                    decoys[index].append(0)
        for decoy in decoys:
            for j in range(46 - len(decoy) // 46):
                for k in range(46):
                    decoy.append(0)
        file.close()

        index = None
        file = open(targets_path + os.sep + target + '\\actives_feature_matrix.txt', 'r')
        for line in file:
            if line.startswith('#'):
                index = int(re.search(' \d+ ', line)[0])-1
            else:
                actives[index].extend(list(map(int, line.strip().split(' '))))
                inactives[index].extend(list(map(int, line.strip().split(' '))))
        for active in actives:
            for j in range(46-(len(active)-46*46)//25):
                for k in range(25):
                    active.append(0)
        for inactive in inactives:
            for j in range(46-(len(inactive)-46*46)//25):
                for k in range(25):
                    inactive.append(0)
        file.close()

        index = None
        file = open(targets_path + os.sep + target + '\\decoys_feature_matrix_used_in_this_work.txt', 'r')
        for line in file:
            if line.startswith('#'):
                index = int(re.search(' \d+ ', line)[0])-1
            else:
                decoys[index].extend(list(map(int, line.strip().split(' '))))
        for decoy in decoys:
            for j in range(46-(len(decoy)-46*46)//25):
                for k in range(25):
                    decoy.append(0)
        file.close()

        file = open(targets_path + os.sep + target + '\\pocket_adj_matrix.txt', 'r')
        for line in file:
            for active in actives:
                active.extend(list(map(int, line.strip().split(' '))))
                for i in range(53-len(line)//2):
                    active.append(0)
            for decoy in decoys:
                decoy.extend(list(map(int, line.strip().split(' '))))
                for i in range(53-len(line)//2):
                    decoy.append(0)
        for active in actives:
            for j in range(53-(len(active)-46*46-46*25)//53):
                for k in range(53):
                    active.append(0)
        for decoy in decoys:
            for j in range(53-(len(decoy)-46*46-46*25)//53):
                for k in range(53):
                    decoy.append(0)
        file.close()

        file = open(targets_path + os.sep + target + '\\pocket_feature_matrix.txt', 'r')
        for line in file:
            for active in actives:
                active.extend(list(map(int, line.strip().split(' '))))
            for decoy in decoys:
                decoy.extend(list(map(int, line.strip().split(' '))))
        for active in actives:
            for j in range(53-(len(active)-46*46-46*25-53*53)//113):
                for k in range(113):
                    active.append(0)
        for decoy in decoys:
            for j in range(53-(len(decoy)-46*46-46*25-53*53)//113):
                for k in range(113):
                    decoy.append(0)
        file.close()

        for inactive in inactives:
            inactive_target = random.choice(targets)
            while inactive_target == target:
                inactive_target = random.choice(targets)
            file = open(targets_path + os.sep + inactive_target + '\\pocket_adj_matrix.txt', 'r')
            for line in file:
                inactive.extend(list(map(int, line.strip().split(' '))))
                for i in range(53 - len(line) // 2):
                    inactive.append(0)
            for j in range(53 - (len(inactive) - 46 * 46 - 46 * 25) // 53):
                for k in range(53):
                    inactive.append(0)
            file.close()
            file = open(targets_path + os.sep + inactive_target + '\\pocket_feature_matrix.txt', 'r')
            for line in file:
                inactive.extend(list(map(int, line.strip().split(' '))))
            for j in range(53 - (len(inactive) - 46 * 46 - 46 * 25-53*53) // 113):
                for k in range(113):
                    inactive.append(0)
            file.close()

        all_actives.extend(actives)
        all_decoys.extend(decoys)
        all_inactives.extend(inactives)

    return all_actives, all_decoys, all_inactives
    # organize original data. For more details, please see our paper.


def shuffle_data(targets_path):
    samples = org_data(targets_path)
    actives_x = np.array(samples[0])
    decoys_x = np.array(samples[1])
    inactives_x = np.array(samples[2])
    actives_y = np.ones((actives_x.shape[0], 1), dtype=int)
    actives_xy = np.append(actives_x, actives_y, axis=1)
    decoys_y = np.zeros((decoys_x.shape[0], 1), dtype=int)
    decoys_xy = np.append(decoys_x, decoys_y, axis=1)
    inactives_y = np.zeros((inactives_x.shape[0], 1), dtype=int)
    inactives_xy = np.append(inactives_x, inactives_y, axis=1)
    data = np.append(actives_xy, decoys_xy, axis=0)
    data = np.append(data, inactives_xy, axis=0)
    np.random.shuffle(data)
    return data


class dataset(Dataset):
    def __init__(self, shuffled_data):
        self.len = shuffled_data.shape[0]
        self.x_data = torch.FloatTensor(shuffled_data[:, :-1])
        self.y_data = torch.LongTensor(shuffled_data[:, -1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 4))
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(12, 12))
        self.conv4 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(12, 4))
        self.conv5 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 5))
        self.conv6 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 5))
        self.pooling = torch.nn.AvgPool2d(2)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(576, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x1 = x[:, :46*46].reshape((len(x), 1, 46, 46))
        # Adjacency matrix of small molecules (46*46)
        x2 = x[:, 46*46:(46*46+46*25)].reshape((len(x), 1, 46, 25))
        # Feature matrix of small molecules (46*25)
        x3 = x[:, (46*46+46*25):(46*46+46*25+53*53)].reshape((len(x), 1, 53, 53))
        # Adjacency matrix of binding sites (53*53)
        x4 = x[:, (46*46+46*25+53*53):].reshape((len(x), 1, 53, 113))
        # Feature matrix of binding sites (53*113)
        x1 = F.relu(self.pooling(self.conv1(x1)))
        # n*8*21*21
        x2 = F.relu(self.pooling(self.conv2(x2)))
        # n*8*21*11
        x3 = F.relu(self.pooling(self.conv3(x3)))
        # n*8*21*21
        x4 = F.relu(self.pooling(self.conv4(x4)))
        # n*8*21*55
        x = torch.cat((x1, x2, x3, x4), 3)
        # n*8*21*108
        x = F.relu(self.pooling(self.conv5(x)))
        # n*16*9*52
        x = self.dropout(x)
        x = F.relu(self.pooling(self.conv6(x)))
        # n*8*3*24
        x = x.view(len(x), -1)
        # n*576
        x = F.relu(self.fc(x))
        # n*32
        x = self.fc2(x)
        # n*2
        return x


def train(epoch):
    model.train()
    loss_epoch_list = []
    total = 0
    correct = 0
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss_epoch_list.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_list.append(epoch+1)
    loss_list.append(sum(loss_epoch_list)/len(loss_epoch_list))
    accuracy_train.append(correct/total)


def validate():
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy_validation.append(correct/total)


def test():
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.detach().cpu()
            labels = labels.detach().cpu()
            score_list.extend(outputs.data.numpy())
            label_list.extend(labels.numpy())
            probability_list.extend(F.softmax(outputs.data, dim=1).numpy()[:, 1])
            _, predicted = torch.max(outputs.data, dim=1)
            predicted_list.extend(predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct/total


if __name__ == '__main__':
    shuffled_samples = shuffle_data(r'.\DUD-E_dataset')
    samples_num = shuffled_samples.shape[0]
    batch_size = 64
    train_dataset = dataset(shuffled_samples[0:int(samples_num*0.9), :])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataset = dataset(shuffled_samples[int(samples_num*0.9):int(samples_num*0.95), :])
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataset = dataset(shuffled_samples[int(samples_num*0.95):, :])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # dataset ratio: train:validation:test = 18:1:1

    model = CNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = op.Adam(model.parameters())

    epoch_list = []
    loss_list = []
    accuracy_train = []
    accuracy_validation = []
    score_list = []
    label_list = []
    probability_list = []
    predicted_list = []
    # these lists are used for plotting
    print('Begin to train the CNN model......')
    for epoch in range(80):
        train(epoch)
        validate()
        print('Epoch: %s Loss: %s Accuracy_train: %s Accuracy_validation: %s'%(
            epoch_list[epoch], loss_list[epoch], accuracy_train[epoch], accuracy_validation[epoch]))
    accuracy = test()
    torch.save(model, r'.\CNN_model_parameters.pkl')

    pre1, rec1, f11, _ = metrics.precision_recall_fscore_support(label_list, predicted_list, average='micro')
    pre2, rec2, f12, _ = metrics.precision_recall_fscore_support(label_list, predicted_list, average='macro')
    print('micro:', '\nPrecision:', pre1, '\nRecall:', rec1, '\nF1-score:', f11)
    # In a binary task, the above indexes for 'micro average' are equal.
    print('------------------------------------')
    print('macro:', '\nPrecision:', pre2, '\nRecall:', rec2, '\nF1-score:', f12)
    print('------------------------------------')
    print('MCC: '+str(metrics.matthews_corrcoef(label_list, predicted_list)))
    print('Final accuracy: '+str(accuracy))
    print('------------------------------------')

    classification_num = 2
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], classification_num)
    label_onehot.scatter_(1, label_tensor, 1)
    label_onehot = np.array(label_onehot)
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(classification_num):
        fpr_dict[i], tpr_dict[i], _ = metrics.roc_curve(label_onehot[:, i], np.array(score_list)[:, i])
        roc_auc_dict[i] = metrics.auc(fpr_dict[i], tpr_dict[i])
    fpr_dict["micro"], tpr_dict["micro"], _ = metrics.roc_curve(label_onehot.ravel(), np.array(score_list).ravel())
    roc_auc_dict["micro"] = metrics.auc(fpr_dict["micro"], tpr_dict["micro"])
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(classification_num)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classification_num):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= classification_num
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = metrics.auc(fpr_dict["macro"], tpr_dict["macro"])
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['font.weight'] = 'bold'
    plt.figure(figsize=(6, 6), dpi=150)
    plt.title('ROC curve for test dataset', fontweight='bold')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.plot(fpr_dict['micro'], tpr_dict['micro'],
             label='micro average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict['micro']),
             color='red', lw=1.5)
    plt.plot(fpr_dict['macro'], tpr_dict['macro'],
             label='macro average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict['macro']),
             color='blue',  lw=1.5)
    colors = ['green', 'yellow']
    for i in range(classification_num):
        plt.plot(fpr_dict[i], tpr_dict[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]),
                 color=colors[i], lw=1)
    plt.legend(loc='lower right')
    plt.show()
    # plot ROC curve including 'micro average', 'macro average', classification 0, and classification 1 on test dataset

    plt.figure(figsize=(6, 6), dpi=150)
    confusion = metrics.confusion_matrix(label_list, predicted_list, labels=[0, 1])
    sns.heatmap(confusion.T, annot=True, cmap='YlGnBu', linewidths=1, square=True, fmt='d')
    plt.title('Confusion matrix for test dataset', fontweight='bold')
    plt.xlabel('Label', fontweight='bold')
    plt.ylabel('Prediction', fontweight='bold')
    plt.show()
    # plot confusion matrix on test dataset

    plt.figure(figsize=(5, 6), dpi=150)
    ax2 = plt.subplot(2, 1, 1)
    ax3 = plt.subplot(2, 1, 2)
    plt.sca(ax2)
    plt.title('Model Training Results', fontweight='bold')
    plt.plot(epoch_list, loss_list, color='red', label='training', lw=1)
    plt.legend(loc='upper right')
    plt.ylabel('CrossEntropyLoss', fontweight='bold')
    plt.sca(ax3)
    plt.plot(epoch_list, accuracy_train, color='red', label='training', lw=1)
    plt.plot(epoch_list, accuracy_validation, color='blue', label='validation', lw=1, ls='--')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.ylim(0, 1)
    plt.xlabel('Epoch', fontweight='bold')
    plt.show()
    # plot LOSS curve and Accuracy curve on training dataset
