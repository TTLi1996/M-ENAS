import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.rc("font", family='Times New Roman')
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
val_info = pd.read_csv("./val1_info.csv")
#print(val_info.head())

acc = sum(val_info['pati_class_ID'] == val_info['pati_val_ID']) / len(val_info)

print("acc: ",acc)

print('rep:sensitivity，specificity \n',classification_report(val_info['pati_class_ID'], val_info['pati_val_ID'],digits=5))

cm = confusion_matrix(val_info['pati_class_ID'], val_info['pati_val_ID'])

import itertools
def cnf_matric_plotter(cm, classes, cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))

    plt.colorbar()
    plt.xlabel('Predicted Labels',fontsize=13)
    plt.ylabel('True Labels',fontsize=13)
    plt.title('Confusion matrix',fontsize=15)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for x, y in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            info = int(cm[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontsize=17,
                     color="white" if info > thresh else "black")
    plt.tight_layout()
    plt.show()

class_indict = {
        "0":'benign',
        "1":'malignancy'
    }

classes = list(class_indict.values())
cnf_matric_plotter(cm,classes,cmap='Blues')
random.seed(124)
colors = [ 'b', 'g','r','c' , 'm','y','k', 'tab:blue','tab:orange' ,'tab:green ','tab:red','tab:purple']
markers = [".",",","o","v", "^", "<",">","p","P",'*','h',"H"]
linestyle = ['--', '-', '-.']
def get_line_arg():
    line_arg = {}
    line_arg["color"] = random.choice(colors)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = 1
    return line_arg

def roc_curves(classes, df: pd.DataFrame):
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)',fontsize=13)
    plt.ylabel('True Positive Rate (Sensitivity)',fontsize=13)
    plt.grid(True)

    auc_list = []
    for each_class in classes:
        y_test = list((df["pati_class"] == each_class))
        y_score = list(df['pati_val_{}_rate'.format(each_class)])
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, **get_line_arg(), label = each_class)
        plt.legend()
        auc_list.append([each_class,auc(fpr, tpr)])
    plt.legend(loc='lower right')
    plt.show()
    return auc_list
auc_list = roc_curves(classes, val_info)
print("auc: ", auc_list)