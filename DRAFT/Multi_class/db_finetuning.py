import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
                        RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
# from tqdm import tqdm
import torch
from typing import List, Dict, Tuple, Type, Union
from torch.utils.data import Dataset, DataLoader
from numpy import ndarray
from torch import Tensor, device
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from evaluate import load
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
import logging
import os
from datasets import load_dataset




simcse_ckpt = 'princeton-nlp/sup-simcse-bert-base-uncased'

f1_metric = load('f1')
acc_metric = load('accuracy')
rec_metric = load('recall')
prec_metric = load('precision')

class TestDataset(Dataset):
    def __init__(self, dataset, tok, max_len, pad_index=None):
        super().__init__()
        self.tok =tok
        self.max_len = max_len
        self.content = dataset
        self.len = self.content.shape[0]
        self.pad_index = self.tok.pad_token
    
    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([0] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
            return inputs
        else:
            inputs = inputs[:max_len]
            return inputs
    
    def __getitem__(self,idx):
        instance = self.content.iloc[idx]
        text = instance['text']
        input_ids = self.tok.encode(text)
        
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)
        return {"encoder_input_ids" : np.array(input_ids, dtype=np.int_)}
                
    def __len__(self):
        return self.len

class HateEncoder(object):

    def __init__(self, ckpt, dataset, existed):
        if existed:
            self.tokenizer = RobertaTokenizer.from_pretrained(ckpt)
            self.model = RobertaForSequenceClassification.from_pretrained(ckpt)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(simcse_ckpt)
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=14)
        
        self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

    def detect(self):

        device = self.device
        test_setup = TestDataset(dataset= self.dataset, tok=self.tokenizer, max_len=128)
        test_dataloader = DataLoader(test_setup, batch_size = 4, shuffle=False)
        pred = []
        logit_socre = []
        self.model.to(device)
        self.model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
            with torch.no_grad():
                outputs = self.model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask)
            
            logits = outputs.logits
            logit_socre.append(logits.cpu())
            
            predictions = torch.argmax(logits, dim=-1)
            pred.append(predictions)

        logit_socre = torch.cat(logit_socre,0)
        logit_socre = np.array(logit_socre.cpu())

        pred = torch.cat(pred, 0)
        pred = np.array(pred.cpu())

        return pred, logit_socre

class contentDataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index=None):
        super().__init__()
        self.tok =tok
        self.max_len = max_len
        self.content = pd.read_csv(file)
        self.len = self.content.shape[0]
        self.pad_index = self.tok.pad_token
    
    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            # pad = np.array([self.pad_index] * (max_len - len(inputs)))
            pad = np.array([0] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
            return inputs
        else:
            inputs = inputs[:max_len]
            return inputs
    
    def __getitem__(self,idx):
        instance = self.content.iloc[idx]
        # text = "[CLS]" + instance['content'] + "[SEP]"
        text = instance['text']
        input_ids = self.tok.encode(text)
        
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)
        label_ids = instance['label']
        # encoder_attention_mask = input_ids.ne(0).float()
        return {"encoder_input_ids" : np.array(input_ids, dtype=np.int_),
                "label" : np.array(label_ids,dtype=np.int_)}
        
    def __len__(self):
        return self.len

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=50, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def evaluate_result(ckpt,dataset, existed):
    pretrained_hate = HateEncoder(ckpt=ckpt, dataset=dataset, existed=existed)

   
    result, logit_socre = pretrained_hate.detect()

    f1_score = f1_metric.compute(predictions=result, references=dataset['label'], average="weighted")
    acc_score = acc_metric.compute(predictions=result, references=dataset['label'])
    rec_score = rec_metric.compute(predictions=result, references=dataset['label'], average="weighted")
    prec_score = prec_metric.compute(predictions=result, references=dataset['label'], average="weighted")
    
    return result, logit_socre, f1_score, acc_score, rec_score, prec_score

def draw_precision_recall_curve(pred, dataset):
    p,r,_ = precision_recall_curve(pred, dataset['label'])
    plt.plot(p, r, marker='.', label='BERT')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    auc_score = auc(r, p)
    print('auc score: ',auc_score)

def draw_roc_curve(prob, dataset,content_name):
    fpr, tpr, _ = metrics.roc_curve(dataset['label'].astype(int),  prob)
    auc = metrics.roc_auc_score(dataset['label'].astype(int),  prob)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    plt.savefig(f'../../finetuning/facts_net/roc_curve_png/ROC_{content_name}.png')

def main(TRAIN_MODE, random_state):

    class_0 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_0_th_label.csv")
    class_1 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_1_th_label.csv")
    class_2 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_2_th_label.csv")
    class_3 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_3_th_label.csv")
    class_4 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_4_th_label.csv")
    class_5 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_5_th_label.csv")
    class_6 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_6_th_label.csv")
    class_7 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_7_th_label.csv")
    class_8 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_8_th_label.csv")
    class_9 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_9_th_label.csv")
    class_10 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_10_th_label.csv")
    class_11 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_11_th_label.csv")
    class_12 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_12_th_label.csv")
    class_13 = pd.read_csv(f"../../building_dataset/dbpedia/{random_state}_13_th_label.csv")

    class_0.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_1.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_2.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_3.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_4.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_5.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_6.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_7.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_8.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_9.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_10.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_11.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_12.drop(["Unnamed: 0"],axis=1,inplace=True)
    class_13.drop(["Unnamed: 0"],axis=1,inplace=True)


    def construct_queries(data, num, seed):
        label_queries = []
        for label in np.unique(data['label']):
            label_query = []
            cond = data['label'] == label
            query = data[cond].sample(n=num, random_state=seed)
            label_query.append(query)
            label_queries.append(label_query)
        label_queries = np.array(label_queries)
        label_queries = label_queries.squeeze(1)
        return label_queries
    
    dataset = load_dataset("dbpedia_14")
    dbpedia_train = pd.DataFrame(dataset['train'])
    dbpedia_train.drop(['title'],axis=1,inplace=True)
    dbpedia_train.columns = ['label','text']
    query_result = construct_queries(dbpedia_train, 8, random_state)           

    # total_queries = pd.read_csv("../../building_dataset/dbpedia/query/query.csv")

    class_0 = pd.concat([class_0, pd.DataFrame(query_result[0])[0]],axis=0)
    class_1 = pd.concat([class_1, pd.DataFrame(query_result[1])[0]],axis=0)
    class_2 = pd.concat([class_2, pd.DataFrame(query_result[2])[0]],axis=0)
    class_3 = pd.concat([class_3, pd.DataFrame(query_result[3])[0]],axis=0)
    class_4 = pd.concat([class_4, pd.DataFrame(query_result[4])[0]],axis=0)
    class_5 = pd.concat([class_5, pd.DataFrame(query_result[5])[0]],axis=0)
    class_6 = pd.concat([class_6, pd.DataFrame(query_result[6])[0]],axis=0)
    class_7 = pd.concat([class_7, pd.DataFrame(query_result[7])[0]],axis=0)
    class_8 = pd.concat([class_8, pd.DataFrame(query_result[8])[0]],axis=0)
    class_9 = pd.concat([class_9, pd.DataFrame(query_result[9])[0]],axis=0)
    class_10 = pd.concat([class_10, pd.DataFrame(query_result[10])[0]],axis=0)
    class_11 = pd.concat([class_11, pd.DataFrame(query_result[11])[0]],axis=0)
    class_12 = pd.concat([class_12, pd.DataFrame(query_result[12])[0]],axis=0)
    class_13 = pd.concat([class_13, pd.DataFrame(query_result[13])[0]],axis=0)


    class_0['label'] = 0
    class_1['label'] = 1
    class_2['label'] = 2
    class_3['label'] = 3
    class_4['label'] = 4
    class_5['label'] = 5
    class_6['label'] = 6
    class_7['label'] = 7
    class_8['label'] = 8
    class_9['label'] = 9
    class_10['label'] = 10
    class_11['label'] = 11
    class_12['label'] = 12
    class_13['label'] = 13

    class_0 = class_0.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_1 = class_1.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_2 = class_2.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_3 = class_3.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_4 = class_4.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_5 = class_5.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_6 = class_6.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_7 = class_7.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_8 = class_8.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_9 = class_9.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_10 = class_10.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_11 = class_11.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_12 = class_12.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    class_13 = class_13.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

    dataset = pd.concat([class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10, class_11, class_12, class_13],axis=0)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset.drop([0],axis=1)
    dataset = dataset.dropna()
    model = AutoModelForSequenceClassification.from_pretrained(simcse_ckpt, num_labels=14)
    tokenizer = AutoTokenizer.from_pretrained(simcse_ckpt)

    train_num = int(len(dataset)*0.8)
    trainset = dataset.iloc[:train_num]
    validset = dataset.iloc[train_num:]

    trainset.to_csv('simcse_enc_trainset.csv')
    validset.to_csv('simcse_enc_validset.csv')

    train_setup = contentDataset(file = "simcse_enc_trainset.csv",tok = tokenizer, max_len = 128)
    valid_setup = contentDataset(file = "simcse_enc_validset.csv",tok = tokenizer, max_len = 128)

    train_dataloader = DataLoader(train_setup, batch_size=256, shuffle=True)
    valid_dataloader = DataLoader(valid_setup, batch_size=256, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)


    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    progress_bar = tqdm(range(num_training_steps))


    if TRAIN_MODE:
        def train_one_epoch(epoch_index):
            running_loss = 0.
            last_loss = 0.

            for i, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
                optimizer.zero_grad()
                
                outputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])

                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                
                progress_bar.update(1)

                running_loss += loss.item()
                if i % 10 == 9:
                    last_loss = running_loss / 9
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.

            return last_loss

        progress_bar = tqdm(range(num_training_steps))
        loss_list = []
        best_vloss = 99999999
        vloss_list = []
        es = EarlyStopping(patience=2)

        model.train()
        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            model.train(True)
            avg_loss = train_one_epoch(epoch)
            loss_list.append(avg_loss)

            model.train(False)
            running_vloss = 0.0
            for i, batch in enumerate(valid_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
                with torch.no_grad():
                    voutputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])
                vloss = voutputs.loss
                running_vloss += vloss


            avg_vloss = running_vloss / (i + 1)
            vloss_list.append(avg_vloss)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                print('best updated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('bets vloss: ',best_vloss)
                model.save_pretrained(f"../../finetuning/facts_net/ckpt/{random_state}_DBPedia")

            if es.step(best_vloss.item()):
                print('EARLY STOPPING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break

        # validation
        model = AutoModelForSequenceClassification.from_pretrained(f"../../finetuning/facts_net/ckpt/{random_state}_DBPedia", num_labels=14)
        model.to(device)
        pred = []
        ref = []

        model.eval()
        for batch in valid_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
            with torch.no_grad():
                outputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])
                
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            pred.append(predictions)
            ref.append(batch['label'])

        pred = torch.cat(pred, 0)
        ref = torch.cat(ref, 0)

        v_f1_score = f1_metric.compute(predictions=pred, references=ref, average="weighted")
        v_acc_score = acc_metric.compute(predictions=pred, references=ref)
        v_rec_score = rec_metric.compute(predictions=pred, references=ref, average="weighted")
        v_prec_score = prec_metric.compute(predictions=pred, references=ref, average="weighted")

        print('validation f1: ',v_f1_score)
        print('validation acc: ',v_acc_score)
        print('validation recall: ',v_rec_score)
        print('validation precision: ',v_prec_score)


    dbpedia = load_dataset("dbpedia_14")
    Test_data = pd.DataFrame(dbpedia['test'])
    Test_data.drop(['title'],axis=1,inplace=True)
    Test_data.columns = ['label','text']

    pred, logit, f1_score, acc_score, rec_score, prec_score = evaluate_result(ckpt=f'../../finetuning/facts_net/ckpt/{random_state}_DBPedia', dataset=Test_data, existed=False)
    prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1)
    prob = prob[::,1]
    print('='*30)
    # draw_roc_curve(np.array(prob), Test_data, content_name.split('_')[1])

    return f1_score, acc_score, rec_score, prec_score

if __name__ == "__main__":
    

    logger = logging.getLogger(__name__)

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('./DBPedia_finetuned_test_results.log')
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    # Only Training

    print(f"DBPedia content detection classifier is training start!!!!!!!!!!!!!!!!!")

    for random_state in [1234, 5678, 7777, 9999]:
        f1, acc, rec, prec = main(True, random_state) 

        print(f"DBPedia content detection classifier Evaluation start!!!!!!!!!!!!!!!!!")
        logger.setLevel(level=logging.DEBUG)
        logger.debug(f"content_name: DBPedia___random state: {random_state}")
        logger.debug(f"f1-score: {round(f1['f1'],4)}")
        logger.debug(f"accuracy: {round(acc['accuracy'],4)}")
        logger.debug(f"recall: {round(rec['recall'],4)}")
        logger.debug(f"precision: {round(prec['precision'],4)}")
        logger.debug("="*100)
