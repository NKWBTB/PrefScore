# import model as M
# import evaluate as E
import config as CFG
import os
import json
import csv
import html
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BatchEncoding, BertModel
from tqdm import tqdm

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

DATASET_ROOT= 'exp/data/'
RESULT_ROOT = "exp/result_bert_base_uncased_summarly"
METHOD = 'summar_ly'

DATASETS = ["cnn_dailymail"]

class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()
        self.model = AutoModel.from_pretrained(CFG.BERT_MODEL)
        self.score_head = nn.Linear(self.model.config.hidden_size, 1)
        self.classify_head = nn.Linear(self.model.config.hidden_size, 3)
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        score = self.score_head(outputs.pooler_output)
        logits = self.classify_head(outputs.last_hidden_state)

        return score, logits

# test(model, test_set)

from utils import sent_tokenizer

def evaluate(docs, sums, model, score_output, detail_output=None):
    docs = sent_tokenizer(docs)
    sums = sent_tokenizer(sums)

    inputs = []

    for _doc, _sum in tqdm(zip(docs, sums)):
        doc_sents = ["[CLS] " + sent for sent in _doc]
        sum_sents = ["[CLS] " + sent for sent in _sum]
        doc_text = " ".join(doc_sents)
        sum_text = " ".join(sum_sents)
        input = tokenizer(doc_text, sum_text, 
            padding='max_length', truncation="longest_first", return_tensors="pt")
        input = {k: v.squeeze() for k, v in input.items()}
        inputs.append(input) 
    
    eval_loader = DataLoader(inputs, batch_size=CFG.BATCH_SIZE)
    
    scores = []
    preds = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch = BatchEncoding(batch)
            batch = batch.to(CFG.DEVICE)
            
            score, logits = model(batch)
            scores.append(score.detach().cpu().numpy())
            preds.append(logits.detach().cpu().numpy())

    scores = np.vstack(scores)
    preds = np.vstack(preds)
    
    print(scores.shape, preds.shape)
    with open(score_output, "w", encoding = "UTF-8") as f:
        f.write("\n".join([str(score[0]) for score in scores])+"\n")

def load_newsroom(csv_file):
    docs = []
    sums = []
    with open(csv_file, "r", encoding="utf-8") as csvfile: 
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"") 
        reader.__next__()
        for row in reader: 
            [_doc, _sum] = row[2:4]
            _doc = _doc.replace("</p><p>", "")
            _sum = _sum.replace("</p><p>", "")
            _doc=html.unescape(_doc) 
            _sum=html.unescape(_sum) 

            # label = scorer([_doc], [_sum]).detach().cpu().numpy()[0][0]
            docs.append(_doc)
            sums.append(_sum)

    return docs, sums

if __name__ == '__main__':
    DATASET = DATASETS[0]
    # test_set = TokenizedTestset(os.path.join(DATASET_ROOT, DATASET, METHOD, 'test.jsonl'), 10)
    CKPT_PATH = os.path.join(RESULT_ROOT, DATASET, METHOD, "model.pth")
    model = Scorer()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=CFG.DEVICE))
    model.to(CFG.DEVICE)
    docs, sums = load_newsroom("human/newsroom/newsroom-human-eval.csv")
    model.eval()
    evaluate(docs, sums, model, os.path.join(RESULT_ROOT, DATASET, METHOD, "test_results_newsroom.tsv"))
