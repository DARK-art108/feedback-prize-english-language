from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np 
import pandas as pd
import os,gc,re,warnings
import sys

from cuml.svm import SVR
import cuml


#Fetching data from folds
dftr = pd.read_csv('C:/Users/lionh/OneDrive/Desktop/roberta-train/data/train_folds.csv')
dfts = pd.read_csv('C:/Users/lionh/OneDrive/Desktop/roberta-train/data/test.csv')

#for ease I have defined the models here, but youb need to take it from config in actual process
tokenizer = None
MAX_LEN = 640
BATCH_SIZE = 4
model = {
    'MODEL_LM_debertabase' : 'microsoft/deberta-base',
    'MODEL_LM_debertalargev3' : 'microsoft/deberta-v3-large',
    'MODEL_LM_debertalarge' : 'microsoft/deberta-large',
    'MODEL_LM_debertalargemnli' : 'microsoft/deberta-large-mnli',
    'MODEL_LM_debertaxlarge' : 'microsoft/deberta-xlarge'
}

#Code for mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]
        tokens = tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens

ds_tr = EmbedDataset(dftr)
embed_dataloader_tr = torch.utils.data.DataLoader(ds_tr,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)
ds_te = EmbedDataset(dfts)
embed_dataloader_te = torch.utils.data.DataLoader(ds_te,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)



#Extracting Embeddings from various deberta models



def get_embeddings(MODEL_NM='', MAX=640, BATCH_SIZE=4, verbose=True):
    global tokenizer, MAX_LEN
    DEVICE="cuda"
    model = AutoModel.from_pretrained( MODEL_NM )
    tokenizer = AutoTokenizer.from_pretrained( MODEL_NM )
    MAX_LEN = MAX
    
    model = model.to(DEVICE)
    model.eval()
    all_train_text_feats = []
    for batch in tqdm(embed_dataloader_tr,total=len(embed_dataloader_tr)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        all_train_text_feats.extend(sentence_embeddings)
    all_train_text_feats = np.array(all_train_text_feats)
    if verbose:
        print('Train embeddings shape',all_train_text_feats.shape)
        
    te_text_feats = []
    for batch in tqdm(embed_dataloader_te,total=len(embed_dataloader_te)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        te_text_feats.extend(sentence_embeddings)
    te_text_feats = np.array(te_text_feats)
    if verbose:
        print('Test embeddings shape',te_text_feats.shape)
        
    return all_train_text_feats, te_text_feats


#taking deberta variants one by one
MODEL_NM = model['MODEL_LM_debertabase']
all_train_text_feats, te_text_feats = get_embeddings(MODEL_NM)
print('Got debertabase embeddings')

MODEL_NM = model['MODEL_LM_debertalargev3']
all_train_text_feats2, te_text_feats2 = get_embeddings(MODEL_NM)

MODEL_NM = model['MODEL_LM_debertalarge']
all_train_text_feats2, te_text_feats2 = get_embeddings(MODEL_NM)

MODEL_NM = model['MODEL_LM_debertalargemnli']
all_train_text_feats2, te_text_feats2 = get_embeddings(MODEL_NM)

MODEL_NM = model['MODEL_LM_debertaxlarge']
all_train_text_feats2, te_text_feats2 = get_embeddings(MODEL_NM)



#combinig all the embeddings
all_train_text_feats = np.concatenate([all_train_text_feats,all_train_text_feats2,
                                       all_train_text_feats3,all_train_text_feats4,
                                       all_train_text_feats5],axis=1)

te_text_feats = np.concatenate([te_text_feats,te_text_feats2,
                                te_text_feats3,te_text_feats4,
                                te_text_feats5],axis=1)


#deleting all the variables to free up memory
del all_train_text_feats2, te_text_feats2
del all_train_text_feats3, te_text_feats3
del all_train_text_feats4, te_text_feats4
del all_train_text_feats5, te_text_feats5
gc.collect()

print('concatenated embeddings have shape', all_train_text_feats.shape)



#Now defining a Rapid SVR model to predict the target variable
preds = []
scores = []
def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(target_cols)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)

#for fold in tqdm(range(FOLDS),total=FOLDS):
for fold in range(FOLDS):
    print('#'*25)
    print('### Fold',fold+1)
    print('#'*25)
    
    dftr_ = dftr[dftr["FOLD"]!=fold]
    dfev_ = dftr[dftr["FOLD"]==fold]
    
    tr_text_feats = all_train_text_feats[list(dftr_.index),:]
    ev_text_feats = all_train_text_feats[list(dfev_.index),:]
    
    ev_preds = np.zeros((len(ev_text_feats),6))
    test_preds = np.zeros((len(te_text_feats),6))
    for i,t in enumerate(target_cols):
        print(t,', ',end='')
        clf = SVR(C=1)
        clf.fit(tr_text_feats, dftr_[t].values)
        ev_preds[:,i] = clf.predict(ev_text_feats)
        test_preds[:,i] = clf.predict(te_text_feats)
    print()
    score = comp_score(dfev_[target_cols].values,ev_preds)
    scores.append(score)
    print("Fold : {} RSME score: {}".format(fold,score))
    preds.append(test_preds)
    
print('#'*25)
print('Overall CV RSME =',np.mean(scores))


#preds: output of the model
#  array([[2.94900918, 2.80713797, 3.1332984 , 2.95393324, 2.66409802,
#          2.67784142],
#         [2.72251678, 2.46881819, 2.70102072, 2.32998419, 2.05252576,
#          2.68055749],
#         [3.63700128, 3.44540453, 3.56421232, 3.65185833, 3.40497899,
#          3.35660815]])]
# np.array(preds)
# [[2.91515398 2.82003951 3.16167831 2.97052073 2.67681551 2.68956351]
#   [2.70253325 2.45898795 2.71900558 2.3118124  2.01470613 2.63774276]
#   [3.6399312  3.4656961  3.58689547 3.64418197 3.41387224 3.35341716]]


#taking average of the predictions
sub = dfte.copy()
sub.loc[:,target_cols] = np.average(np.array(preds),axis=0)
sub_columns = pd.read_csv("C:/Users/lionh/OneDrive/Desktop/roberta-train/data/sample_submission.csv").columns
sub = sub[sub_columns]