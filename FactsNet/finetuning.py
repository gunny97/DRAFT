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
    plt.savefig(f'/DPR/finetuning/facts_net/roc_curve_png/ROC_{content_name}.png')

def main(pos_train_path, neg_train_path, query_list, content_name, test_data_path ,TRAIN_MODE = True):


    if TRAIN_MODE:
        passages = pd.read_csv(pos_train_path)
        negative = pd.read_csv(neg_train_path)

        passages.drop(['Unnamed: 0'],axis=1, inplace=True)
        negative.drop(['Unnamed: 0'],axis=1, inplace=True)

        query = query_list

                            
        passages = passages.append(pd.DataFrame(query,columns=['text']))

        total_passages = list(passages.text)
        for q in query:
            total_passages.append(q)

        positive_df = passages.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

        if positive_df.shape[0] > negative.shape[0]:
            positive_df = positive_df.sample(n=negative.shape[0],random_state=1234)
            negative_df = negative
        else:
            negative_df = negative.sample(n=positive_df.shape[0], random_state=1234)


        def build_dataset(pos_df, neg_df):
            # --> in this example
            # pos: hate text 
            # neg: normal text
            pos_df['label'] = 1
            neg_df['label'] = 0
            data = pd.concat([pos_df, neg_df],axis=0)
            data = data.sample(frac=1).reset_index(drop=True)
            return data

        dataset = build_dataset(positive_df, negative_df)
        dataset = dataset.dropna(axis=0)

        dataset = dataset.drop_duplicates(['text'], keep = False)

        
        model = AutoModelForSequenceClassification.from_pretrained(simcse_ckpt, num_labels=2)
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

        from torch.nn.parallel import DistributedDataParallel

        device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)


        progress_bar = tqdm(range(num_training_steps))

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
                model.save_pretrained(f"/DPR/finetuning/facts_net/error_6_ckpt/{content_name}")

            if es.step(best_vloss.item()):
                print('EARLY STOPPING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break

        # validation
        model = AutoModelForSequenceClassification.from_pretrained(f"/DPR/finetuning/facts_net/error_6_ckpt/{content_name}", num_labels=2)
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

        v_f1_score = f1_metric.compute(predictions=pred, references=ref)
        v_acc_score = acc_metric.compute(predictions=pred, references=ref)
        v_rec_score = rec_metric.compute(predictions=pred, references=ref)
        v_prec_score = prec_metric.compute(predictions=pred, references=ref)

        print('validation f1: ',v_f1_score)
        print('validation acc: ',v_acc_score)
        print('validation recall: ',v_rec_score)
        print('validation precision: ',v_prec_score)


    Test_data = pd.read_csv(test_data_path)
    Test_data.drop(['Unnamed: 0'],axis=1,inplace=True)
    Test_data.columns = ['text', 'label']
    for i in range(Test_data.shape[0]):
        if Test_data.iloc[i]['label'] == 0:
            Test_data.label[i] = 1
        elif Test_data.iloc[i]['label'] == 1 or Test_data.iloc[i]['label'] == 2:
            Test_data.label[i] = 0

    Test_data.drop(index = len(Test_data)-1, inplace=True)

    pred, logit, f1_score, acc_score, rec_score, prec_score = evaluate_result(ckpt=f'/DPR/finetuning/facts_net/error_6_ckpt/{content_name}', dataset=Test_data, existed=False)
    prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1)
    prob = prob[::,1]
    print('='*30)
    #  draw_roc_curve(np.array(prob), Test_data, content_name.split('_')[1])

    return f1_score, acc_score, rec_score, prec_score

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    streamHandler = logging.StreamHandler()
    # fileHandler = logging.FileHandler('./factsNet_finetuned_DRAFT_results.log')
    fileHandler = logging.FileHandler('./ss_500_factsnet_DRAFT.log')
    
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
                break

        query_list = list(pd.read_csv(content_query_path)['query'])

        for tdp in test_data_path_list:
            if content_name.split('_')[1] in tdp:
                test_data_path = tdp
                break
        

        # Only Evaluation
        # ckpt_exist = os.path.exists(f"/DPR/finetuning/facts_net/ckpt/{content_name}")
        
        # if ckpt_exist:
        #     f1, acc, rec, prec = main(pos_train_path, neg_train_path, query_list, content_name, test_data_path, False)
        # else:
        #     continue


        # Only Training
        ckpt_exist = os.path.exists(f"/DPR/finetuning/facts_net/error_6_ckpt/{content_name}")
        if ckpt_exist:

            logger.setLevel(level=logging.DEBUG)
            logger.debug(f"testdata: {test_data_path}")
            logger.debug(f"contentname: {content_name}")
            logger.debug(f"pos_train_path: {pos_train_path}")
            logger.debug(f"neg_train_path: {neg_train_path}")
            logger.debug(f"query: {query_list}")
            logger.debug("="*100)
            continue
        else:
            print(f"{content_name} content detection classifier is training start!!!!!!!!!!!!!!!!!")

            if pos_train_path == "/DPR/building_dataset/factsnet/history/query_culture-facts.csv_positive.csv":
                test_data_path = "/crawl/crawler/test_data/history/culture-facts.csv"
                pos_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_culture-facts_positive.csv"
                neg_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_culture-facts_negative.csv"
                query_list_path = "/crawl/crawler/query_output/history/culture/query_culture-facts.csv"

            elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_cat-facts.csv_positive.csv":
                test_data_path ="/crawl/crawler/test_data/nature/cat-facts.csv"
                pos_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_cat-facts_positive.csv"
                neg_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_cat-facts_negative.csv"
                query_list_path = "/crawl/crawler/query_output/nature/animals/query_cat-facts.csv"

            elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_otter-facts.csv_positive.csv":
                test_data_path ="/crawl/crawler/test_data/nature/otter-facts.csv"
                pos_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_otter-facts_positive.csv"
                neg_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_otter-facts_negative.csv"
                query_list_path = "/crawl/crawler/query_output/nature/animals/query_otter-facts.csv"

            elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_bear-facts.csv_positive.csv":
                test_data_path = "/crawl/crawler/test_data/nature/bear-facts.csv"
                pos_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_bear-facts_positive.csv"
                neg_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_bear-facts_negative.csv"
                query_list_path = "/crawl/crawler/query_output/nature/animals/query_bear-facts.csv"

            elif pos_train_path == "/DPR/building_dataset/factsnet/nature/query_whale-facts.csv_positive.csv":
                test_data_path = "/crawl/crawler/test_data/nature/whale-facts.csv"
                pos_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_whale-facts_positive.csv"
                neg_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_whale-facts_negative.csv"
                query_list_path = "/crawl/crawler/query_output/nature/animals/query_whale-facts.csv"
                
            elif pos_train_path == "/DPR/building_dataset/factsnet/world/query_egypt-facts.csv_positive.csv":
                test_data_path = "/crawl/crawler/test_data/world/egypt-facts.csv"
                pos_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_egypt-facts_positive.csv"
                neg_train_path = "/DPR/building_dataset/factsnet_error_update/error_updatequery_egypt-facts_negative.csv"
                query_list_path = "/crawl/crawler/query_output/world/countries/query_egypt-facts.csv"
            else:
                continue
            
            query_list = list(pd.read_csv(query_list_path)['query'])
            
            import pdb; pdb.set_trace()

            f1, acc, rec, prec = main(pos_train_path, neg_train_path, query_list, content_name, test_data_path, True)

            print(f"{content_name} content detection classifier Evaluation start!!!!!!!!!!!!!!!!!")
            logger.setLevel(level=logging.DEBUG)
            logger.debug(f"content_name: {content_name}")
            logger.debug(f"f1-score: {round(f1['f1'],4)}")
            logger.debug(f"accuracy: {round(acc['accuracy'],4)}")
            logger.debug(f"recall: {round(rec['recall'],4)}")
            logger.debug(f"precision: {round(prec['precision'],4)}")
            logger.debug("="*100)
