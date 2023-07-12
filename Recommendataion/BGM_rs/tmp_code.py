from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from matplotlib import pyplot as plt
from datasets import load_metric
import argparse

parser = argparse.ArgumentParser(description='BGM Recommendation given Text')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--ckpt_path',
                        type=str,
                        default="monologg/kobigbird-bert-base",
                        help='need existed pretrained model ckpt')

        parser.add_argument('--finetuned_ckpt_path',
                        type=str,
                        default="tmp_finetune",
                        help='naming finetuned ckpt file')

        parser.add_argument('--data_path',
                        type=str,
                        default="/home/keonwoo/anaconda3/envs/bgmRS/data/labeled_data_0706.csv",
                        help='input dataset')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='batch size')

        parser.add_argument('--num_epochs',
                            type=int,
                            default=5,
                            help='batch size')

        parser.add_argument('--device',
                type=str,
                default="cuda:0",
                help='enter device type')

        parser.add_argument('--lang',
                type=str,
                default="kor",
                help='enter language type')

        return parser


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


def load_dataset(path, batch_size, tokenizer):
    dataset = pd.read_csv(path)
    dataset.drop(['Unnamed: 0'],axis=1,inplace=True)

    dataset['label'] = pd.factorize(dataset['label'])[0]
    # dataset.columns = ['label','text']
    dataset = dataset.sample(frac=1).reset_index(drop=True)


    train_num = int(len(dataset)*0.9)
    trainset = dataset.iloc[:train_num]
    validset = dataset.iloc[train_num:]

    trainset.to_csv('/home/keonwoo/anaconda3/envs/bgmRS/data/trainset.csv')
    validset.to_csv('/home/keonwoo/anaconda3/envs/bgmRS/data/validset.csv')

    train_setup = contentDataset(file = "/home/keonwoo/anaconda3/envs/bgmRS/data/trainset.csv",tok = tokenizer, max_len = 512)
    valid_setup = contentDataset(file = "/home/keonwoo/anaconda3/envs/bgmRS/data/validset.csv",tok = tokenizer, max_len = 512)


    train_dataloader = DataLoader(train_setup, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_setup, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader



def train(model, train_dataloader, num_epochs, device, ckpt_path):
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    # device = torch.device(f"cuda:{device_num}") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))
    loss_list = []

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
            outputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])
            loss = outputs.loss
            loss_list.append(loss.item())
            
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.save_pretrained(f"/home/keonwoo/anaconda3/envs/bgmRS/ckpt/{ckpt_path}")
    return model, loss_list

def evaluate(ckpt_path, device, valid_dataloader):
    # device = torch.device(f"cuda:{device_num}") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(f"/home/keonwoo/anaconda3/envs/bgmRS/ckpt/{ckpt_path}", num_labels=9)
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


    acc = load_metric("accuracy")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    prec = load_metric("precision")

    acc_result = acc.compute(predictions=pred, references=ref)
    recall_result = recall.compute(predictions=pred, references=ref, average="weighted")
    prec_result = prec.compute(predictions=pred, references=ref, average="weighted")
    f1_result = f1.compute(predictions=pred, references=ref, average="weighted")

    print("Accuracy: ", acc_result)
    print("Recall: ", recall_result)
    print("Precision: ", prec_result)
    print("F1-score: ", f1_result)

    return acc_result, recall_result, prec_result, f1_result

def plot_train_loss(loss_list):
    plt.plot(loss_list)
    plt.title('train loss')
    plt.show()

def main(ckpt_path, data_path, batch_size, num_epochs, device, finetuned_ckpt_path):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=9)
    torch.cuda.empty_cache()
    train_dataloader, valid_dataloader = load_dataset(data_path, batch_size, tokenizer)
    fintuend_model, train_loss = train(model, train_dataloader, num_epochs, device, finetuned_ckpt_path)
    plot_train_loss(train_loss)
    acc_result, recall_result, prec_result, f1_result = evaluate(finetuned_ckpt_path, device, valid_dataloader)

    print('Accuracy: ', acc_result)
    print('Recall: ', recall_result)
    print('Precision: ', prec_result)
    print('F1-score: ', f1_result)


if __name__ == '__main__':
    
    # ko-bert : "monologg/kobert"
    # ko-sbert: "jhgan/ko-sbert-sts"
    # snu-sbert: 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
    # ko-bigbird: "monologg/kobigbird-bert-base"
    # ko-electra: "monologg/koelectra-base-v3-discriminator"
    # ko-simcse: 'BM-K/KoSimCSE-roberta'
    # en-sbert: "nlptown/bert-base-multilingual-uncased-sentiment"
    # multi-sbert: "sentence-transformers/paraphrase-xlm-r-multilingual-v1"


    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args.ckpt_path, args.data_path, args.batch_size, args.num_epochs, \
         args.device, args.finetuned_ckpt_path, args.lang)


    


    