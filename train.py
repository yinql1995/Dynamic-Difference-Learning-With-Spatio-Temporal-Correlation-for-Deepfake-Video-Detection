import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import datetime
from torchsummary import summary
import torch
from dataset import MyDataset_multiframe
from model import DDLmodel
from sklearn.metrics import roc_auc_score, f1_score

batch_size = 12
frame_num = 8
continue_train = 0
retrain_epoch = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4'
model_name = ''
epoches = 20
train = 1
test = 1
video_level = 1
image_size = 256
time = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
summary_path = os.path.join('summary/weight/', model_name)
model_path = os.path.join(summary_path, '20.pth')

path_save_figure1 = 'summary/figure/'+model_name+'.jpg'


result_train_score = []
result_train_loss = []

result_val_score = []
result_val_loss = []


class Saver(object):
    def __init__(self, path_figure1 ):
        self.path_figure1 = path_figure1
        self.figure_saver()

    def figure_saver(self):
        # # drawing tool
        print('---------------------------Drawing...--------------------------')
        epochs = range(1, len(result_train_loss) + 1)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(epochs, result_train_loss, 'b', label="train Loss")
        lns3 = ax1.plot(epochs, result_val_loss, 'k', label="test Loss")

        lns2 = ax2.plot(epochs, result_train_score, 'r', label="train score")
        lns7 = ax2.plot(epochs, result_val_score, 'y', label="final score")

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('f1 score')
        # 合并图例
        lns = lns1 + lns2 + lns3 + lns7
        labels = ["train Loss", "train score", "test Loss",  "final score"]
        plt.legend(lns, labels, loc=2)
        plt.savefig(self.path_figure1)
        plt.show()


if __name__ == '__main__':
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)



    train_dataset = MyDataset_multiframe(image_size, 'train', 'your_data.txt', num_frame=frame_num)
    test_dataset = MyDataset_multiframe(image_size, 'test', 'your_data.txt', num_frame=frame_num)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=False,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              drop_last=False,
                                              num_workers=8)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    model = DDLmodel()

    if Device == 'cuda':
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    if continue_train:
        model.load_state_dict(torch.load(model_path))


    criterion = nn.CrossEntropyLoss().cuda()
    Bloss = nn.BCELoss().cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    print(time, 'train:', train, 'test:', test, ' ==>', model_name, '\n', image_size, ' begin\n')
    for epoch in range(epoches):
        if train:

            print('Epoch {}/{}'.format(epoch + 1, epoches))
            print('-' * 10)
            model.train()
            train_loss = 0.0
            train_corrects = 0.0
            for (name, image, labels) in tqdm(train_loader):
                iter_loss = 0.0
                iter_corrects = 0.0
                image = image.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(image)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)



                loss.backward()
                optimizer.step()
                iter_loss = loss.data.item()
                train_loss += iter_loss
                iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                train_corrects += iter_corrects
                iteration += 1
                if not (iteration % 5000):
                    print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size,
                                                                               iter_corrects / batch_size))
                    # print(model.module.gate)

            epoch_loss = train_loss / train_dataset_size
            epoch_acc = train_corrects / train_dataset_size
            print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            result_train_loss.append(epoch_loss)
            result_train_score.append(epoch_acc)
            with open('summary/result/' + model_name + '_' + time + '.txt', 'a+') as file:
                file.write(
                    'epoch: ' + str(epoch+1+retrain_epoch) + ' train loss: ' + str(epoch_loss) + ' train_acc: ' + str(epoch_acc) + '\n')

        roc_pre = []
        roc_lab = []
        frame_f1_lab = []
        frame_f1_pre = []

        label_dict = {}
        count_dict = {}
        if test:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_corrects = 0.0
                for (name, image, labels) in tqdm(test_loader):
                    for i in range(len(name)):
                        label_dict[name[i]] = labels[i]
                    image = image.cuda()
                    labels = labels.cuda()
                    outputs = model(image)

                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)


                    test_loss += loss.data.item()
                    test_corrects += torch.sum(preds == labels.data).to(torch.float32)

                    frame_f1_lab.append(labels)
                    frame_f1_pre.append(preds)

                    softmax_out = torch.softmax(outputs, dim=-1)
                    roc_pre.append(softmax_out)
                    label_roc = labels.view(-1, 1)
                    label_roc = torch.cat((1 - label_roc, label_roc), dim=1)
                    roc_lab.append(label_roc)

                    #geshubijiao
                    # predstonp = preds.cpu().numpy()
                    # for i in range(len(name)):
                    #     if name[i] not in count_dict:
                    #         count_dict[name[i]] = torch.zeros(2)
                    #         count_dict[name[i]][predstonp[i]] += 1
                    #     else:
                    #         count_dict[name[i]][predstonp[i]] += 1

                    #预测分数求和
                    predstonp = softmax_out.cpu().numpy()
                    for i in range(len(name)):
                        if name[i] not in count_dict:
                            count_dict[name[i]] = torch.zeros(2)
                            count_dict[name[i]] += predstonp[i]
                        else:
                            count_dict[name[i]] += predstonp[i]


                epoch_loss = test_loss / test_dataset_size
                epoch_acc = test_corrects / test_dataset_size

                epoch_auc = roc_auc_score(torch.cat(roc_lab, dim=0).cpu().data, torch.cat(roc_pre, dim=0).cpu().data)
                epoch_f1 = f1_score(torch.cat(frame_f1_lab, dim=0).cpu().data, torch.cat(frame_f1_pre, dim=0).cpu().data)

                print('epoch test loss: {:.4f} Acc: {:.4f} AUC: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_auc, epoch_f1))
                result_val_loss.append(epoch_loss)
                result_val_score.append(epoch_acc)
                if epoch_acc >= best_acc:
                    if video_level:
                        pre = []
                        lab = []

                        length = len(label_dict)
                        for value in label_dict.values():
                            temp = value.unsqueeze(0)
                            lab.append(temp)
                            # video_f1_lab.append(value)

                        for value in count_dict.values():
                            temp = value.unsqueeze(0)
                            pre.append(temp)


                        lab = torch.cat(lab, dim=0)
                        pre = torch.cat(pre, dim=0)
                        _, index = torch.max(pre, 1)
                        iter_corrects = torch.sum(index == lab).to(torch.float32)
                        print('video_level_acc:', iter_corrects / length)

                        lab_roc = lab.view(-1, 1)
                        lab_roc = torch.cat((1 - lab_roc, lab_roc), dim=1)
                        pre_roc = torch.softmax(pre, dim=-1)
                        video_auc = roc_auc_score(lab_roc.cpu().data, pre_roc.cpu().data)
                        print('video_level_auc:', video_auc)

                        video_f1 = f1_score(lab.cpu().data, index.cpu().data)
                        print('video_level_f1:', video_f1)

                        with open('summary/result/' + model_name + '_' + time + '.txt', 'a+') as file:
                            file.write(
                                'epoch:' + str(epoch + 1 + retrain_epoch) + ' video_level_acc: ' + str(iter_corrects / length) +
                                ' video_level_auc: ' + str(video_auc)+ ' video_level_f1: ' + str(video_f1) + '\n')

                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, os.path.join(summary_path, str(epoch+1+retrain_epoch) + '.pth'))
                with open('summary/result/' + model_name + '_' + time + '.txt', 'a+') as file:
                    file.write('epoch:' + str(epoch+1+retrain_epoch) + ' test loss: ' + str(epoch_loss) + ' test_acc: ' + str(
                        epoch_acc) + ' test auc: ' + str(epoch_auc) + ' test f1: ' + str(epoch_f1)+ '\n' + '\n')

                scheduler.step()

            print('Best test Acc: {:.4f}'.format(best_acc))
            print('\n')

    Saver(path_save_figure1)