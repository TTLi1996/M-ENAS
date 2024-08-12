#coding=utf-8
from atriple_mbcnn import TripleBigNASDynamicModel, MDDDataset, define_sc, define_sc_2d, CELossSoft, CrossEntropyLoss_label_smoothed

import time
import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.utils import shuffle
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
def train_stkf(device, lr, num_epochs, batch_size, train_subset, val_subset, test_subset):

    train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

    train_pth = './ctpetfus_train_model.pth'
    val_pth = './ctpetfus_valed_model.pth'
    test_pth = './ctpetfus_test_model.pth'
    supernet_cfg = define_sc()
    net = TripleBigNASDynamicModel(supernet_cfg=supernet_cfg,n_classes=2, bn_param=(0.98, 1e-5))
    net.to(device)
    cels_criterion = CrossEntropyLoss_label_smoothed
    soft_ce = CELossSoft()
    net_params = [
        {"params": net.get_parameters(['bn', 'bias'], mode="exclude"), "weight_decay": 1e-5},
        {"params": net.get_parameters(['bn', 'bias'], mode="include"), "weight_decay": 0}, ]
    optimizer = optim.Adam(net_params, lr)
    w_sche = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2)
    #optimizer = optim.SGD(net_params, lr=lr, momentum=0.9, dampening=0.0, weight_decay=1e-5, nesterov=True)
    best_train = 0.0
    best_val = 0.0
    best_test = 0.0
    patience = 0
    patience1 = 0
    t1 = time.perf_counter()
    for cur_epoch in range(num_epochs):
        net.train()
        cur_step = cur_epoch * len(train_loader)
        running_loss_max = 0.0
        running_correct_max = 0
        running_loss_min = 0.0
        running_correct_min = 0
        train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        for cur_iter, (data1, data2, data3, true_target) in enumerate(train_loader):

            data1 = data1.type(torch.float).to(device)
            data2 = data2.type(torch.float).to(device)
            data3 = data3.type(torch.float).to(device)
            true_target = true_target.to(device)
            optimizer.zero_grad()
            net.sample_max_subnet()
            net.set_dropout_rate(0.2, 0.2)
            preds_max = net(data1, data2, data3)
            loss_max = cels_criterion(preds_max, true_target)

            running_loss_max += loss_max.item()
            predicted_max = torch.max(preds_max, dim=1)[1]
            running_correct_max += (predicted_max == true_target).sum().item()
            loss_max.backward(retain_graph=True)
            with torch.no_grad():
                soft_logits = preds_max.clone().detach()

            # optimizer.zero_grad()
            net.sample_min_subnet()
            net.set_dropout_rate(0, 0)
            preds_min = net(data1, data2, data3)
            loss_min = soft_ce(preds_min, soft_logits)
            running_loss_min += loss_min.item()
            predicted_min = torch.max(preds_min, dim=1)[1]
            running_correct_min += (predicted_min == true_target).sum().item()
            loss_min.backward(retain_graph=True)
            for i in range(2):
                subnet_seed = int("%d%.3d%.3d" % (cur_step, i, 0))
                net.sample_active_subnet(subnet_seed)
                net.set_dropout_rate(0, 0)
                preds_sam = net(data1, data2, data3)
                loss_sam = soft_ce(preds_sam, soft_logits)
                loss_sam.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(net.parameters(), 15)
            optimizer.step()

            rate = (cur_iter + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, (loss_max+loss_min) / 2.), end="")

        with torch.no_grad():
            net.sample_max_subnet()
            subnetmax = net.get_active_subnet()
            subnetmax.to(device)
            c = compare_model_parameters(net, subnetmax)
            assert c == True
            subnetmax.eval()
            val_loss_max = 0.0
            val_correct_max = 0
            test_loss_max = 0.0
            test_correct_max = 0

            net.sample_min_subnet()
            subnetmin = net.get_active_subnet()
            subnetmin.to(device)
            subnetmin.eval()
            val_loss_min = 0.0
            val_correct_min = 0
            test_loss_min = 0.0
            test_correct_min = 0
            for cur_iter, (data1, data2,data3,  true_target1) in enumerate(val_loader):
                data1 = data1.type(torch.float).to(device)
                data2 = data2.type(torch.float).to(device)
                data3 = data3.type(torch.float).to(device)
                true_target1 = true_target1.to(device)
                preds_submax = subnetmax(data1, data2, data3)
                sub_loss_max = cels_criterion(preds_submax, true_target1)
                predict_y = torch.max(preds_submax, dim=1)[1]
                val_correct_max += (predict_y == true_target1).sum().item()
                val_loss_max += sub_loss_max.item()

                soft_logits = preds_submax.clone().detach()

                preds_min = subnetmin(data1, data2, data3)
                sub_loss_min = soft_ce(preds_min, soft_logits)
                predict_y_min = torch.max(preds_min, dim=1)[1]
                val_correct_min += (predict_y_min == true_target1).sum().item()
                val_loss_min += sub_loss_min.item()

            for cur_iter, (data1, data2, data3, true_target1) in enumerate(test_loader):
                data1 = data1.type(torch.float).to(device)
                data2 = data2.type(torch.float).to(device)
                data3 = data3.type(torch.float).to(device)
                true_target1 = true_target1.to(device)
                preds_submax = subnetmax(data1, data2, data3)
                sub_loss_max = cels_criterion(preds_submax, true_target1)
                predict_y = torch.max(preds_submax, dim=1)[1]
                test_correct_max += (predict_y == true_target1).sum().item()
                test_loss_max += sub_loss_max.item()

                soft_logits = preds_submax.clone().detach()

                preds_min = subnetmin(data1, data2, data3)
                sub_loss_min = soft_ce(preds_min, soft_logits)
                predict_y_min = torch.max(preds_min, dim=1)[1]
                test_correct_min += (predict_y_min == true_target1).sum().item()
                test_loss_min += sub_loss_min.item()


        print()
        print((time.perf_counter() - t1), 's')
        train_acc_max = running_correct_max / len(train_subset)
        train_loss_max = running_loss_max / len(train_loader)

        train_acc_min = running_correct_min / len(train_subset)
        train_loss_min = running_loss_min / len(train_loader)

        valed_loss_max = val_loss_max / len(val_loader)
        valed_acc_max = val_correct_max / len(val_subset)

        valed_loss_min = val_loss_min / len(val_loader)
        valed_acc_min = val_correct_min / len(val_subset)

        test_loss_max = test_loss_max / len(test_loader)
        test_acc_max = test_correct_max / len(test_subset)

        test_loss_min = test_loss_min / len(test_loader)
        test_acc_min = test_correct_min / len(test_subset)

        if train_acc_max >= best_train or train_acc_min >= best_train:
            best_train = max(train_acc_max, train_acc_min)
            torch.save(net.state_dict(), train_pth)
        if valed_acc_max >= best_val or valed_acc_min >= best_val:
            best_val = max(valed_acc_max, valed_acc_min)
            patience = 0
            torch.save(net.state_dict(), val_pth)
        else:
            patience += 1
        if test_acc_max >= best_test or test_acc_min >= best_test:
            best_test = max(test_acc_max, test_acc_min)
            patience1 = 0
            torch.save(net.state_dict(), test_pth)
        else:
            patience1 += 1
        print(
            '[epoch %d] train_acc: [%.4f %.4f] loss: [%.4f %.4f] val_acc: [%.4f %.4f] %.1f loss: [%.4f %.4f] test_acc: [%.4f %.4f] %.1f loss: [%.4f %.4f] lr: %.6f' %
            (cur_epoch + 1, train_acc_max, train_acc_min, train_loss_max, train_loss_min, valed_acc_max, valed_acc_min,
             patience,
             valed_loss_max, valed_loss_min, test_acc_max, test_acc_min, patience1, test_loss_max, test_loss_min,
             optimizer.param_groups[0]['lr']))
        #w_sche.step()
    return 'ok'

def compare_model_parameters(model1, model2):
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    if len(params1) != len(params2):
        return False
    for p1, p2 in zip(params1, params2):
        if not torch.equal(p1.data, p2.data):
            return False
    return True

from monai.transforms import RandRotate
tf1_rotate5_10 = RandRotate(range_x=[0.0349066, 0.174533],prob=1.0)
tf2_rotate_5_10 = RandRotate(range_x=[6.10865, 6.24828], prob=1.0)
def tf3_tranx10(obj_image):
    return torch.roll(obj_image, shifts=10, dims=2)
def tf4_tranx_10(obj_image):
    return torch.roll(obj_image, shifts=-10, dims=2)
def tf5_trany10(obj_image):
    return torch.roll(obj_image, shifts=10, dims=3)
def tf6_trany10(obj_image):
    return torch.roll(obj_image, shifts=-10, dims=3)
def datactpetfus_augmentation(tra_data):
    data_augmented = []
    for i in range(len(tra_data)):
        obj_imagec, obj_imagep, obj_imaget, label = tra_data[i]
        if label == 3:
            obj_imagetf1c = tf1_rotate5_10(obj_imagec)
            obj_imagetf1p = tf1_rotate5_10(obj_imagep)
            obj_imagetf1t = tf1_rotate5_10(obj_imaget)
            obj_imagetf2c = tf2_rotate_5_10(obj_imagec)
            obj_imagetf2p = tf2_rotate_5_10(obj_imagep)
            obj_imagetf2t = tf2_rotate_5_10(obj_imaget)
            obj_imagetf3c = tf6_trany10(obj_imagec)
            obj_imagetf3p = tf6_trany10(obj_imagep)
            obj_imagetf3t = tf6_trany10(obj_imaget)

            data_augmented.append((obj_imagec, obj_imagep, obj_imaget, label))
            data_augmented.append((obj_imagetf1c, obj_imagetf1p, obj_imagetf1t, label))
            data_augmented.append((obj_imagetf2c, obj_imagetf2p, obj_imagetf2t, label))
            data_augmented.append((obj_imagetf3c, obj_imagetf3p, obj_imagetf3t, label))
        elif label != 3:
            obj_imagetfrec = tf3_tranx10(obj_imagec)
            obj_imagetfrep = tf3_tranx10(obj_imagep)
            obj_imagetfret = tf3_tranx10(obj_imaget)

            obj_imagetf1c = tf1_rotate5_10(obj_imagec)
            obj_imagetf1p = tf1_rotate5_10(obj_imagep)
            obj_imagetf1t = tf1_rotate5_10(obj_imaget)
            obj_imagetf2c = tf2_rotate_5_10(obj_imagec)
            obj_imagetf2p = tf2_rotate_5_10(obj_imagep)
            obj_imagetf2t = tf2_rotate_5_10(obj_imaget)
            obj_imagetf3c = tf5_trany10(obj_imagec)
            obj_imagetf3p = tf5_trany10(obj_imagep)
            obj_imagetf3t = tf5_trany10(obj_imaget)

            obj_imagetf4c = tf1_rotate5_10(obj_imagetfrec)
            obj_imagetf4p = tf1_rotate5_10(obj_imagetfrep)
            obj_imagetf4t = tf1_rotate5_10(obj_imagetfret)
            obj_imagetf5c = tf2_rotate_5_10(obj_imagetfrec)
            obj_imagetf5p = tf2_rotate_5_10(obj_imagetfrep)
            obj_imagetf5t = tf2_rotate_5_10(obj_imagetfret)
            obj_imagetf6c = tf5_trany10(obj_imagetfrec)
            obj_imagetf6p = tf5_trany10(obj_imagetfrep)
            obj_imagetf6t = tf5_trany10(obj_imagetfret)

            data_augmented.append((obj_imagetf1c, obj_imagetf1p, obj_imagetf1t, label))
            data_augmented.append((obj_imagetf2c, obj_imagetf2p, obj_imagetf2t, label))
            data_augmented.append((obj_imagetf3c, obj_imagetf3p, obj_imagetf3t, label))
            data_augmented.append((obj_imagetf4c, obj_imagetf4p, obj_imagetf4t, label))
            data_augmented.append((obj_imagetf5c, obj_imagetf5p, obj_imagetf5t, label))
            data_augmented.append((obj_imagetf6c, obj_imagetf6p, obj_imagetf6t, label))
        else:
            print("error")
    return data_augmented
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    #
    dti_path = '/mnt/data/litongtong/NAS-MDD/exp_data/dti'
    fmri_path = '/mnt/data/litongtong/NAS-MDD/exp_data/fmri'
    smri_path = '/mnt/data/litongtong/NAS-MDD/exp_data/smri'
    ldataset = MDDDataset(dti_path=dti_path, fmri_path=fmri_path, smri_path=smri_path)
    mdd_dataset = shuffle(ldataset, random_state=340)

    train_val_dataset, test_dataset = random_split(mdd_dataset, lengths=[92, 24],generator=torch.Generator().manual_seed(34))
    test_dataset = datactpetfus_augmentation(test_dataset)
    batch_size = 1

    num_epochs = 100
    lr = 0.00001
    kf = KFold(n_splits=4)
    split_num = 0

    for train_index, val_index in kf.split(train_val_dataset):
        split_num = split_num + 1
        print('running kf: {}'.format(split_num))
        train_subset = torch.utils.data.dataset.Subset(train_val_dataset, train_index)
        val_subset = torch.utils.data.dataset.Subset(train_val_dataset, val_index)
        train_subset = datactpetfus_augmentation(train_subset)
        val_subset = datactpetfus_augmentation(val_subset)
        train_labels_count = Counter([label for _,_,_, label in train_subset])
        val_labels_count = Counter([label for _,_,_,  label in val_subset])
        test_labels_count = Counter([label for _,_,_,  label in test_dataset])

        print("using {}:{} images for train, {}:{} images for val, {}:{} images for val.".format(len(train_subset),train_labels_count.most_common(),
               len(val_subset),val_labels_count.most_common(),len(test_dataset),test_labels_count.most_common()))
        if split_num == 2:
            break
    train_stkf(device=device,lr=lr,num_epochs=num_epochs,batch_size=batch_size,train_subset=train_subset, val_subset=val_subset,test_subset=test_dataset)