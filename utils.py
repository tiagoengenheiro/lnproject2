import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
#from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

def save_acc_per_label(y_true,y_pred,model_name='sparse'):
    labels=np.unique(y_true)
    mcm=multilabel_confusion_matrix(y_true,y_pred,labels=labels)
    res={}
    for i,m in enumerate(mcm):
        print(m)
        label=labels[i]
        res[label]=[round((m[0][0]+m[1][1])/np.sum(m),3)]
    print(res)
    df=pd.DataFrame.from_dict(res)
    df.to_csv(f"class_acc/{model_name}",sep="\t")
    print("Done")

def save_confusion_matrices(y_true,y_pred,model_name='sparse'):
    labels=['DECEPTIVENEGATIVE','DECEPTIVEPOSITIVE','TRUTHFULNEGATIVE','TRUTHFULPOSITIVE']
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    plt.figure(figsize=(8, 6))
    label_abv=["DCN","DCP","TFN","TFP"]
    sns.set(font_scale=3.0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_abv, yticklabels=label_abv)
    fontsize=50
    if model_name=='dense':
        plt.xlabel('Predicted',fontsize=fontsize)
    plt.ylabel('Actual',fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'images/{model_name}/confusion_matrix_{model_name}.png')
    mcm=multilabel_confusion_matrix(y_true,y_pred,labels=labels)
    for i,label in enumerate(label_abv):
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=3.0)
        sns.heatmap(mcm[i], annot=True, fmt="d", cmap="Blues")
        if model_name=='dense':
            plt.xlabel('Predicted',fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f'images/{model_name}/confusion_matrix_{label}_{model_name}.png')


            
        
        


