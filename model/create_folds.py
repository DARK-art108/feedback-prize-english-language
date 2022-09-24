import numpy as np 
import pandas as pd
import os,gc,re,warnings
import sys
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
warnings.filterwarnings('ignore')

dftr = pd.read_csv('C:/Users/lionh/OneDrive/Desktop/roberta-train/data/train.csv')
dftr['src'] = 'train'
dfts = pd.read_csv('C:/Users/lionh/OneDrive/Desktop/roberta-train/data/test.csv')
dfts['src'] = 'test'

df = pd.concat([dftr,dfts],ignore_index=True)

target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

FOLDS = 25
skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
for i, (train_index, val_index) in enumerate(skf.split(dftr, dftr[target_cols])):
    dftr.loc[val_index, 'FOLD'] = i


#saving folds
dftr.to_csv("C:/Users/lionh/OneDrive/Desktop/roberta-train/data/train_folds.csv", index=False)
print("Saved Folds..")