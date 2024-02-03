# -*- coding: utf-8 -*-
import torch,time,os,subprocess,vs_metrics
from torch.utils.data import Dataset #导入数据
from torch.utils.data import DataLoader #批量加载数据集
import torch.nn.functional as F #激活函数
import torch.optim as op #优化器
import numpy as np #矩阵与数组处理
import matplotlib.pyplot as plt #绘图
from sklearn import metrics #绘制ROC曲线
from itertools import cycle #迭代器，用于绘制不同颜色的ROC曲线
from pylab import mpl #设置绘图字体
import seaborn as sns #绘制混淆矩阵
from organize_data import org_data #组织数据

def shuffle_data(targetspath):
    result=org_data(targetspath)
    actives_x = np.array(result[0])
    decoys_x = np.array(result[1])
    inactives_x = np.array(result[2])
    result='' #清内存
    actives_y = np.ones((actives_x.shape[0], 1), dtype=int)
    actives_xy = np.append(actives_x, actives_y, axis=1)
    decoys_y = np.zeros((decoys_x.shape[0], 1), dtype=int)
    decoys_xy = np.append(decoys_x, decoys_y, axis=1)
    inactives_y = np.zeros((inactives_x.shape[0], 1), dtype=int)
    inactives_xy = np.append(inactives_x, inactives_y, axis=1)
    data_np = np.append(actives_xy, decoys_xy, axis=0)
    data_np = np.append(data_np, inactives_xy, axis=0)
    data_np = data_np.reshape((data_np.shape[0], 1, 12065)) #batchsize*channel*列数
    np.random.shuffle(data_np)
    return data_np

class matrixdataset(Dataset):     #定义加载数据集方式
    def __init__(self, division_data_np):
        self.len = division_data_np.shape[0] #取样本数(行数)
        self.x_data = torch.from_numpy(division_data_np[...,:-1])
        self.x_data = self.x_data.type(torch.FloatTensor)
        self.y_data = torch.from_numpy(division_data_np[...,-1:].ravel())
#        self.y_data = self.y_data.type(torch.FloatTensor)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

class Net(torch.nn.Module):  #CNN模型
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(5,5))
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 4))
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(12, 12))
        self.conv4 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(12, 4))
        self.conv5 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4,5))
        self.conv6 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4,5))
        self.pooling = torch.nn.AvgPool2d(2)
        self.dropout=torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(576, 32)
        self.fc2=torch.nn.Linear(32,2)

    def forward(self, x):
        x1=x[...,:2116]
        x1=x1.reshape((len(x),1,46,46))
        x2=x[...,2116:3266]
        x2=x2.reshape((len(x),1,46,25))
        x3=x[...,3266:6075]
        x3=x3.reshape((len(x),1,53,53))
        x4=x[...,6075:]
        x4=x4.reshape((len(x),1,53,113))
        batch_size = x.size(0)
        x1=F.relu(self.pooling(self.conv1(x1)))
        x2 = F.relu(self.pooling(self.conv2(x2)))
        x3 = F.relu(self.pooling(self.conv3(x3)))
        x4 = F.relu(self.pooling(self.conv4(x4)))
        x=torch.cat((x1,x2,x3,x4),3)
        x=F.relu(self.pooling(self.conv5(x)))
        x=self.dropout(x)
        x = F.relu(self.pooling(self.conv6(x)))
        x=x.view(batch_size,-1)
        x=F.relu(self.fc(x))
        x=self.fc2(x)
        return x

def train(epoch):    #模型训练，一个epoch中所有batch_size的训练过程
    total = 0
    correct = 0
    loss=0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs,labels=inputs.to(device),labels.to(device) #把数据移到GPU计算
        optimizer.zero_grad() #梯度清零
        labels = labels.squeeze().long() #降维以适应CrossEntropyLoss的参数形式
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1) # max(,dim=1)函数返回每一行中概率最大的数据和标签，_占位原数据，我们只需要标签
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    loss_list.append(loss.item())   #记录第一个图:Loss曲线的y坐标值
    epoch_list.append(epoch+1)      #记录绘图的x坐标值
    accuracy_list1.append(correct/total)  #在训练集本身上测试训练结果，记录第二个图:Accuracy的y坐标值

def validate():                      #在验证集上测试准确率
    total = 0
    correct=0
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 把数据移到GPU计算
            labels=labels.squeeze().long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy_list2.append(correct/total)

def test():                      #在测试集上测试准确率
    total = 0
    correct=0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 把数据移到GPU计算
            labels=labels.squeeze().long()
            outputs = model(inputs)
            outputs=outputs.detach().cpu() # 把数据从GPU搬回CPU，以绘制ROC曲线
            labels=labels.detach().cpu() # 把数据从GPU搬回CPU，以绘制ROC曲线
            score_list.extend(outputs.data.numpy())   #绘制ROC曲线
            label_list.extend(labels.numpy())             #绘制ROC曲线
            probability_list.extend(F.softmax(outputs.data,dim=1).numpy()[:,1]) #计算其余虚拟筛选指标
            _, predicted = torch.max(outputs.data, dim=1)
            pre_list.extend(predicted.numpy())            #计算分类性能指标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct/total

if __name__ == '__main__':
    start_time=time.time()
    shuffle_result=shuffle_data(r'D:\DUD_E_done\all')
    batch_size=64
    num_sample=shuffle_result.shape[0]
    train_dataset = matrixdataset(shuffle_result[0:int(num_sample*0.9),...])  #向下取整
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    validation_dataset = matrixdataset(shuffle_result[int(num_sample*0.9):int(num_sample*0.95),...])
    validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=batch_size)
    test_dataset = matrixdataset(shuffle_result[int(num_sample*0.95):,...])
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    model = Net()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
#    model=torch.load('.\\model.pkl')
    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = op.SGD(model.parameters(), lr=0.01,momentum=0.5)
    optimizer = op.Adam(model.parameters(), lr=0.001)
    epoch_list=[]
    loss_list=[]
    accuracy_list1=[]
    accuracy_list2=[]
    score_list = []
    label_list = []
    probability_list=[]
    pre_list=[]
    for epoch in range(80):
        train(epoch)
        validate()
        if (epoch+1)%5==0:
            print('The %s epoch has finished.'%(epoch+1))
    accuracy_test=test()
    torch.save(model,'.\\model.pkl')

    pre1, rec1, f11, _ = metrics.precision_recall_fscore_support(label_list, pre_list,average='micro')
    pre2, rec2, f12, _ = metrics.precision_recall_fscore_support(label_list, pre_list, average='macro')
    print('micro:',"\n精确率:", pre1, "\n召回率:", rec1, "\nf1-score:", f11) #二分类的微平均结果一样
    print('------------------------------------')
    print('macro:',"\n精确率:", pre2, "\n召回率:", rec2, "\nf1-score:", f12) #宏平均易受样本不平衡影响，当样本不平衡时宜采用
    print('------------------------------------')
    print('MCC:'+str(metrics.matthews_corrcoef(label_list, pre_list)))
    print('Accuracy:'+str(accuracy_test))
    end_time=time.time()
    print('Time:'+str(int(end_time-start_time)))
    print('------------------------------------')
#    print('EF2%(fold):'+str(vs_metrics.enrichment_factor(np.array(label_list),np.array(probability_list),percentage=2,kind='fold')))
#    print('EF2%(percentage):' + str(vs_metrics.enrichment_factor(np.array(label_list), np.array(probability_list), percentage=2,kind='percentage')))
#    print('BEDROC80.5:'+str(vs_metrics.bedroc(np.array(label_list), np.array(probability_list),alpha=80.5)))
#    使用oddt计算以上值，需要先将各样本按从大到小排序

    num=2   #计算四条ROC曲线
    score_array = np.array(score_list)
    label_tensor = torch.tensor(label_list)    # 以下四行将label转换成onehot形式
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num)
    label_onehot.scatter_(1,label_tensor, 1)
    label_onehot = np.array(label_onehot)
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num):
        fpr_dict[i], tpr_dict[i], _ = metrics.roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = metrics.auc(fpr_dict[i], tpr_dict[i])
    fpr_dict["micro"], tpr_dict["micro"], _ = metrics.roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = metrics.auc(fpr_dict["micro"], tpr_dict["micro"])
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = metrics.auc(fpr_dict["macro"], tpr_dict["macro"])
    # 在测试数据集上，度量分类器对大类判别的有效性应该选择微平均(micro)，而度量分类器对小类判别的有效性则应该选择宏平均(macro)
    # 微平均是对数据集中的每一个示例不分类别进行统计建立全局混淆矩阵，然后计算相应的指标。宏平均是指所有类别的每一个统计指标值的算数平均值。

    mpl.rcParams['font.sans-serif'] = ['Times New Roman'] #修改字体
    mpl.rcParams['font.weight'] = 'bold' #字体加粗
    plt.figure(figsize=(6,6),dpi=150) # 绘制四条roc曲线，num条+宏平均+微平均
    plt.title('ROC Curve for Test',fontweight='bold')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate',fontweight='bold')
    plt.ylabel('True Positive Rate',fontweight='bold')
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='red', lw=1.5)
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='blue',  lw=1.5)
    colors = cycle(['green', 'yellow'])
    for i, color in zip(range(num), colors):
        plt.plot(fpr_dict[i], tpr_dict[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]),
                 color=color, lw=1)
    plt.legend(loc='lower right')
    plt.show()

    confusion = metrics.confusion_matrix(label_list, pre_list, labels=[0, 1]) #绘制混淆矩阵，证明样本不平衡对模型无影响
    sns.heatmap(confusion.T, annot=True, cmap='YlGnBu', linewidths=1, square=True,fmt='d')
    plt.title('Confusion matrix for test', fontweight='bold')
    plt.xlabel('Label', fontweight='bold')
    plt.ylabel('Prediction', fontweight='bold')
    plt.show()

    plt.figure(figsize=(5, 6),dpi=150) #绘制Loss曲线，Accuracy曲线
    ax2=plt.subplot(2,1,1)
    ax3=plt.subplot(2,1,2)
    plt.sca(ax2)
    plt.title('Model Training Results',fontweight='bold')
    plt.plot(epoch_list, loss_list,color='red',label='training',lw=1)
    plt.legend(loc='upper right')
    plt.ylabel('CrossEntropyLoss',fontweight='bold')
    plt.sca(ax3)
    plt.plot(epoch_list, accuracy_list1,color='red',label='training',lw=1)
    plt.plot(epoch_list, accuracy_list2, color='blue', label='validation',lw=1,ls='--')
#    plt.text(1, 0.1, 'Accuracy_test=' + str(round(accuracy_test, 3)))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy',fontweight='bold')
    plt.ylim(0,1)
    plt.xlabel('Epoch',fontweight='bold')
    plt.show()

#    logROCfile=open(r'logROCdata.txt','w') #使用rocker绘制logROC曲线
#    logROCfile.write('label_list probability_list\n')
#    for o in range(len(label_list)):
#        if label_list[o] == 1:
#            logROCfile.write('positive' + str(o) + ' ' + str(probability_list[o]) + '\n')
#        else:
#            logROCfile.write('negative' + str(o) + ' ' + str(probability_list[o]) + '\n')
#    logROCfile.close()
#    logAUC = vs_metrics.roc_log_auc(np.array(label_list), np.array(probability_list), ascending_score=False, log_min=0.01,
#                                    log_max=1.0)
#    logAUCrandom = vs_metrics.random_roc_log_auc(log_min=0.01, log_max=1.0)
#    os.system(
#        r'rocker.exe logROCdata.txt -an positive -c 2 -s 5 5 -lp 0.01 -la "False Positive Rate" "True Positive Rate" -li CNN(area=%s) random(area=%s) -l 4 -cl red -les 10 -EF 2.0 -BR 80.5 -p logROC.png'%(round(logAUC,2),round(logAUCrandom,2)))
#    subprocess.Popen(['start','logROC.png'],shell=True)
    #记得仔细筛选drugbank中的药物确保能适应模型运行

