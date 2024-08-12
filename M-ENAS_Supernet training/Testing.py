from atriple_mbcnn import TripleBigNASDynamicModel, MDDDataset, define_sc, CELossSoft, CrossEntropyLoss_label_smoothed

from tqdm import tqdm
import torch
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import random_split
def get_cttest_df(test_data):
    class_indict = {
        "0": 'benign',
        "1": 'malignancy'
    }
    df = pd.DataFrame()
    df['pati_class_ID'] = [test_data[i][3] for i in range(len(test_data))]
    df['pati_class'] = [class_indict[str(test_data[i][3])] for i in range(len(test_data))]
    return df
def stkf_test(device, batch_size, test_subset):

    test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)
    val_info = get_cttest_df(test_subset)
    test_pth = './ctpetfus_test_model.pth'
    supernet_cfg = define_sc()
    net = TripleBigNASDynamicModel(supernet_cfg=supernet_cfg, n_classes=2, bn_param=(0.98, 1e-5))

    net.load_state_dict(torch.load(test_pth, map_location=device))
    net.to(device)

    net.eval()
    y_true_pred = []
    with torch.no_grad():
        for val_data in tqdm(test_loader):
            data1,data2,data3, target = val_data
            data1 = data1.type(torch.float).to(device)
            data2 = data2.type(torch.float).to(device)
            data3 = data3.type(torch.float).to(device)
            target = target.to(device)
            net.sample_min_subnet()
            outputs = net(data1, data2, data3)
            outputs = torch.softmax(outputs, dim=1)
            pmax = outputs
            outputs = torch.argmax(outputs, dim=1)
            y_pred = outputs.to('cpu').numpy()
            y_true = target.to("cpu").numpy()
            pmaxs = pmax.to("cpu").numpy()
            y_true_pred.append([y_true[0], y_pred[0], pmaxs[0][0], pmaxs[0][1]])
            if len(y_true) > 1:
                y_true_pred.append([y_true[1], y_pred[1], pmaxs[1][0], pmaxs[1][1]])
    val_info['pati_val_ID'] = [i[1] for i in y_true_pred]
    val_info['pati_val_benign_rate'] = [i[2] for i in y_true_pred]
    val_info['pati_val_malignancy_rate'] = [i[3] for i in y_true_pred]
    val_info.to_csv('valtriple_info.csv', index=False)
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

    dti_path = '/mnt/data/litongtong/NAS-MDD/exp_data/dti'
    fmri_path = '/mnt/data/litongtong/NAS-MDD/exp_data/fmri'
    smri_path = '/mnt/data/litongtong/NAS-MDD/exp_data/smri'
    ldataset = MDDDataset(dti_path=dti_path, fmri_path=fmri_path, smri_path=smri_path)
    mdd_dataset = shuffle(ldataset, random_state=340)

    train_val_dataset, test_dataset = random_split(mdd_dataset, lengths=[92, 24], generator=torch.Generator().manual_seed(34))
    test_dataset = datactpetfus_augmentation(test_dataset)
    batch_size = 2

    stkf_test(device=device,batch_size=batch_size,test_subset=test_dataset)