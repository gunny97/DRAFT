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
import nltk
import pickle
from nltk.corpus import stopwords
import re
# nltk.download('all')

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
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    # 0: normal, 1: hate
    result, logit_socre = pretrained_hate.detect()

    f1_score = f1_metric.compute(predictions=result, references=dataset['label'])
    acc_score = acc_metric.compute(predictions=result, references=dataset['label'])
    rec_score = rec_metric.compute(predictions=result, references=dataset['label'])
    prec_score = prec_metric.compute(predictions=result, references=dataset['label'])
    
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
    plt.savefig(f'/DPR/finetuning/facts_net/baseline_roc_curve_png/ROC_{content_name}.png')

def main(baseline_type, class_num, test_data_path, query_list):

    Test_data = pd.read_csv(test_data_path)
    Test_data.drop(['Unnamed: 0'],axis=1,inplace=True)
    Test_data.columns = ['text', 'label']
    for i in range(Test_data.shape[0]):
        if Test_data.iloc[i]['label'] == 0:
            Test_data.label[i] = 1
        elif Test_data.iloc[i]['label'] == 1 or Test_data.iloc[i]['label'] == 2:
            Test_data.label[i] = 0

    Test_data.drop(index = len(Test_data)-1, inplace=True)

    # random
    if baseline_type == 'random_':

        random_pred = np.random.randint(0, class_num, size=Test_data.shape[0])

        f1_score = f1_metric.compute(predictions=random_pred, references=Test_data['label'])
        acc_score = acc_metric.compute(predictions=random_pred, references=Test_data['label'])
        rec_score = rec_metric.compute(predictions=random_pred, references=Test_data['label'])
        prec_score = prec_metric.compute(predictions=random_pred, references=Test_data['label'])
    
    elif baseline_type == 'token_level_freq_':


        def cleaning(content):
            result = re.sub(r'[^\.\?\!\w\d\s]','',content) 
            result = result.lower()
            tokenized_list = nltk.word_tokenize(result)
            token_pos = nltk.pos_tag(tokenized_list)
            NN_words = []
            for word, pos in token_pos:
                if 'NN' in pos:
                    NN_words.append(word)
            return NN_words

        query_nouns = [cleaning(q) for q in query_list]
        query_nouns = sum(query_nouns, [])

        pred = []
        for i in range(Test_data.shape[0]):
            for j, noun in enumerate(query_nouns):
                if noun in Test_data.iloc[i]['text']:
                    pred.append(1)
                    break
                if j == len(query_nouns)-1:
                    pred.append(0)

        f1_score = f1_metric.compute(predictions=pred, references=Test_data['label'])
        acc_score = acc_metric.compute(predictions=pred, references=Test_data['label'])
        rec_score = rec_metric.compute(predictions=pred, references=Test_data['label'])
        prec_score = prec_metric.compute(predictions=pred, references=Test_data['label'])

    elif baseline_type == 'only_query_based_':

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        tokenizer = AutoTokenizer.from_pretrained(simcse_ckpt)
        model = AutoModel.from_pretrained(simcse_ckpt)

        q_tensor = []
        for q in query_list:

            inputs = tokenizer(q, return_tensors="pt")
            outputs = model(**inputs)

            # cls token
            cls_emb = outputs.last_hidden_state[:,0,:]
            q_tensor.append(cls_emb)

        dist_list = []
        for i, text_a in enumerate(q_tensor):
            for j, text_b in enumerate(q_tensor):
                if i != j and i < j:
                    dist = cos(torch.tensor(text_a[0]),torch.tensor(text_b[0]))
                    dist_list.append(dist)


        threshold = np.average(dist_list)

        query_based_pred = []
        for i in range(Test_data.shape[0]):
            test_input = tokenizer(Test_data.iloc[i]['text'], return_tensors="pt")
            test_outputs = model(**test_input)
            test_cls_emb = test_outputs.last_hidden_state[:,0,:]

            for j, text in enumerate(q_tensor):
                score = cos(torch.tensor(test_cls_emb[0]),torch.tensor(text[0]))
                if score.item() >= threshold:
                    query_based_pred.append(1)
                    break
                if j == len(q_tensor)-1:
                    query_based_pred.append(0)

        f1_score = f1_metric.compute(predictions=query_based_pred, references=Test_data['label'])
        acc_score = acc_metric.compute(predictions=query_based_pred, references=Test_data['label'])
        rec_score = rec_metric.compute(predictions=query_based_pred, references=Test_data['label'])
        prec_score = prec_metric.compute(predictions=query_based_pred, references=Test_data['label'])

    elif baseline_type == 'test':
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        tokenizer = AutoTokenizer.from_pretrained(simcse_ckpt)
        model = AutoModel.from_pretrained(simcse_ckpt)

        total_length = 0
        for i in range(Test_data.shape[0]):
            test_input = tokenizer(Test_data.iloc[i]['text'], return_tensors="pt")

            total_length += len(test_input['input_ids'][0])

        print('='*100)
        print(total_length)
        print('='*100)

    return f1_score, acc_score, rec_score, prec_score


if __name__ == "__main__":

    import logging

    logger = logging.getLogger(__name__)

    streamHandler = logging.StreamHandler()

    # 'random_', "token_level_freq_", "only_query_based_"
    baseline_type = 'only_query_based_'
    fileHandler = logging.FileHandler(f'./factsNet_baselines_{baseline_type}results_NEW.log')
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    query_path = "/crawl/crawler/query_output"
    query_path_list = []


    for cat in os.listdir(query_path):
        if cat == 'general':
            continue
        else:
            sub_path = os.path.join(query_path,cat)

            for cat_2 in os.listdir(sub_path):
                edge_path = os.path.join(sub_path, cat_2)
                for cat_3 in os.listdir(edge_path):
                    file_path = os.path.join(edge_path, cat_3)
                    query_path_list.append(file_path)

    test_data_path = "/crawl/crawler/test_data"
    test_data_path_list = []

    for cat in os.listdir(test_data_path):
        sub_path = os.path.join(test_data_path,cat)

        for cat_2 in os.listdir(sub_path):
            edge_path = os.path.join(sub_path, cat_2)
            test_data_path_list.append(edge_path)

    # finetuning train data path
    path = "/DPR/building_dataset/factsnet" 
    file_list = []
    for (root, directories, files) in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    # file_list[151], file_list[152] = file_list[152], file_list[151]
    # file_list.insert(153, file_list[169])
    # del file_list[170]


    for i,x in enumerate(file_list[0::2]):
        
        # if i <= 4:
        #     continue

        content_name = x.split('/')[-1].split('.csv')[0]
        print('='*30,content_name,'='*30)
        pos_train_path = x
        neg_train_path = x.replace('positive','negative')
        for q in query_path_list:
            if content_name in q:
                content_query_path = q
                print(content_query_path)
                break

        query_list = list(pd.read_csv(content_query_path)['query'])

        for tdp in test_data_path_list:
            if content_name.split('_')[1] in tdp:
                test_data_path = tdp
                break

        ckpt_exist = True

        class_num = 2

        if pos_train_path == "/DPR/building_dataset/factsnet/history/query_culture-facts.csv_positive.csv":
            test_data_path = "/crawl/crawler/test_data/history/culture-facts.csv"
        elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_cat-facts.csv_positive.csv":
            test_data_path ="/crawl/crawler/test_data/nature/cat-facts.csv"
        elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_otter-facts.csv_positive.csv":
            test_data_path ="/crawl/crawler/test_data/nature/otter-facts.csv"
        elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_bear-facts.csv_positive.csv":
            test_data_path = "/crawl/crawler/test_data/nature/bear-facts.csv"
        elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_whale-facts.csv_positive.csv":
            test_data_path = "/crawl/crawler/test_data/nature/whale-facts.csv"
        elif pos_train_path == "/DPR/building_dataset/factsnet/world/query_egypt-facts.csv_positive.csv":
            test_data_path = "/crawl/crawler/test_data/world/egypt-facts.csv"

        if ckpt_exist:
            f1, acc, rec, prec = main(baseline_type, class_num, test_data_path, query_list)
        else:
            continue
        
        logger.setLevel(level=logging.DEBUG)
        logger.debug(f"content_name: {content_name}")
        logger.debug(f"f1-score: {round(f1['f1'],4)}")
        logger.debug(f"accuracy: {round(acc['accuracy'],4)}")
        logger.debug(f"recall: {round(rec['recall'],4)}")
        logger.debug(f"precision: {round(prec['precision'],4)}")
        logger.debug("="*100)
    


