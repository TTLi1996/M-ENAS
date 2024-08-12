# -*- coding: utf-8 -*-
from atriple_mbcnn import TripleBigNASDynamicModel, MDDDataset, define_sc

import time
import math
import torch
import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.utils import shuffle
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.utils.data import random_split
from itertools import product
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
def get_all_subnets(supernet_cfg):
    # get all subnets
    all_subnets = []
    stage_names = ['mb1', 'mb2', 'mb3', 'mb4', 'mb5', 'mb6', 'mb7']

    first_conv = getattr(supernet_cfg, 'first_conv')
    last_conv = getattr(supernet_cfg, 'last_conv')

    mb_stage_subnets = []
    for mbstage in stage_names:
        mb_block_cfg = getattr(supernet_cfg, mbstage)
        mb_stage_subnets.append(list(product(
            mb_block_cfg.c,
            mb_block_cfg.d,
            mb_block_cfg.k
        )))
    all_mb_stage_subnets = list(product(*mb_stage_subnets))
    for fc in first_conv.c:
        for mb in all_mb_stage_subnets:
            np_mb_choice = np.array(mb)
            width = np_mb_choice[:, 0].tolist()  # c
            depth = np_mb_choice[:, 1].tolist()  # d
            kernel = np_mb_choice[:, 2].tolist() # k

            for lc in last_conv.c:
                all_subnets.append({
                        'width': [fc] + width + [lc],
                        'depth': depth,
                        'kernel_size': kernel,
                        'expand_ratio': [1, 1, 1, 1, 1, 1, 1]
                })
    return all_subnets
def data_test(model, test_loader, device):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    df = pd.DataFrame(columns=['pati_class_ID','pati_class','pati_val_ID','pati_val_benign_rate','pati_val_malignancy_rate'])

    model.eval()
    with torch.no_grad():
        for cur_iter, (data1, data2, data3, true_target1) in enumerate(test_loader):
            data1 = data1.type(torch.float).to(device)
            data2 = data2.type(torch.float).to(device)
            data3 = data3.type(torch.float).to(device)
            true_target1 = true_target1.to(device)
            preds_submax = model(data1, data2, data3)
            preds_submax1 = preds_submax.to('cpu').numpy()
            predict_y = torch.max(preds_submax, dim=1)[1]
            newres = {'pati_class_ID': true_target1[0].to("cpu").item(), 'pati_class': 'benign' if true_target1[0] == 0 else 'malignancy',
                      'pati_val_ID': predict_y[0].to("cpu").item(),
                      'pati_val_benign_rate': preds_submax1[0][0], 'pati_val_malignancy_rate': preds_submax1[0][1]}
            df.loc[len(df)] = newres
    test_acc = sum(df['pati_class_ID'] == df['pati_val_ID']) / len(df)
    return trainable_params, test_acc, df

def stkf_test(device,batch_size,test_dataset):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('_______________sample_min_subnet_______________')
    test_pth = './ctpetfus_test_model.pth'
    supernet_cfg = define_sc()
    supernet = TripleBigNASDynamicModel(supernet_cfg=supernet_cfg, n_classes=2, bn_param=(0.98, 1e-5))
    supernet.load_state_dict(torch.load(test_pth, map_location=device))
    supernet.to(device)

    supernet.eval()
    supernet.sample_min_subnet()
    subnet_trainable_params, subnet_test_acc, dfk = data_test(supernet, test_loader, device)
    print('min_subnet test:')
    print('[%8d] trainable_params: %d test_acc: %.6f' % (0, subnet_trainable_params, subnet_test_acc))


    # Envolutionary search
    print('_______________Envolutionary search_______________')
    for k in range(50):
        test_pth = './ctpetfus_test_model.pth'
        supernet_cfg = define_sc()
        supernet = TripleBigNASDynamicModel(supernet_cfg=supernet_cfg, n_classes=2, bn_param=(0.98, 1e-5))
        supernet.load_state_dict(torch.load(test_pth, map_location=device))
        all_subnets = get_all_subnets(supernet_cfg=supernet_cfg)
        selected_indexes = [7166249, 12898187, 6965854]
        subnet_cfg1 = all_subnets[selected_indexes[0]]
        subnet_cfg2 = all_subnets[selected_indexes[1]]
        subnet_cfg3 = all_subnets[selected_indexes[2]]
        supernet.set_active_subnet(width= subnet_cfg1['width'],depth=subnet_cfg1['depth'],
                                   kernel_size=subnet_cfg1['kernel_size'],expand_ratio=subnet_cfg1['expand_ratio'],
                                   width2=subnet_cfg2['width'], depth2=subnet_cfg2['depth'],
                                   kernel_size2=subnet_cfg2['kernel_size'], expand_ratio2=subnet_cfg2['expand_ratio'],
                                   width3=subnet_cfg3['width'], depth3=subnet_cfg3['depth'],
                                   kernel_size3=subnet_cfg3['kernel_size'], expand_ratio3=subnet_cfg3['expand_ratio'],)
        subnet = supernet.get_active_subnet().to(device)
        # Validate
        print('subnet test:',selected_indexes)
        subnet_trainable_params, subnet_test_acc, dfk = data_test(subnet, test_loader, device)
        print('[%8d] trainable_params: %d test_acc: %.6f' % (k + 1, subnet_trainable_params, subnet_test_acc))
        del subnet
        break

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    # #
    dti_path = '/mnt/data/litongtong/NAS-MDD/exp_data/dti'
    fmri_path = '/mnt/data/litongtong/NAS-MDD/exp_data/fmri'
    smri_path = '/mnt/data/litongtong/NAS-MDD/exp_data/smri'
    ldataset = MDDDataset(dti_path=dti_path, fmri_path=fmri_path, smri_path=smri_path)
    mdd_dataset = shuffle(ldataset, random_state=340)
    #
    train_val_dataset, test_dataset = random_split(mdd_dataset, lengths=[92, 24],generator=torch.Generator().manual_seed(34))
    batch_size = 1
    test_labels_count = Counter([label for _, _, _, label in test_dataset])
    print("using  {}:{} images for val.".format(len(test_dataset),test_labels_count.most_common()))

    stkf_test(device=device,batch_size=batch_size,test_dataset=test_dataset)
def get_model(pth_th):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    supernet_cfg = define_sc()
    supernet = TripleBigNASDynamicModel(supernet_cfg=supernet_cfg, n_classes=2, bn_param=(0.98, 1e-5))
    supernet.load_state_dict(torch.load(pth_th, map_location=device))
    all_subnets = get_all_subnets(supernet_cfg=supernet_cfg)
    selected_indexes = [7166249, 12898187, 6965854]
    subnet_cfg1 = all_subnets[selected_indexes[0]]
    subnet_cfg2 = all_subnets[selected_indexes[1]]
    subnet_cfg3 = all_subnets[selected_indexes[2]]
    supernet.set_active_subnet(width=subnet_cfg1['width'], depth=subnet_cfg1['depth'],
                               kernel_size=subnet_cfg1['kernel_size'], expand_ratio=subnet_cfg1['expand_ratio'],
                               width2=subnet_cfg2['width'], depth2=subnet_cfg2['depth'],
                               kernel_size2=subnet_cfg2['kernel_size'], expand_ratio2=subnet_cfg2['expand_ratio'],
                               width3=subnet_cfg3['width'], depth3=subnet_cfg3['depth'],
                               kernel_size3=subnet_cfg3['kernel_size'], expand_ratio3=subnet_cfg3['expand_ratio'], )

    subnet = supernet.get_active_subnet().to(device)
    return subnet