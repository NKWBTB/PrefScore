import csv
import json
import html
import os
import config as CFG
import torch
from model import Scorer

# Evaluate for TAC2010
def evaluate_tac(json_file, output_path, scorer):
    with open(output_path, "w") as f:
        tac = json.load(open(json_file, 'r', encoding="utf-8"))
        for docset in tac.keys():
            for article in tac[docset]["articles"]: 
                # each of 10 articles is a list of strings 
                article = " ".join(article)
                article = article.replace("\n", " ")
                article = article.replace("\t", " ")
                if len(article) == 0:
                    article = " ." 

                for summarizer in tac[docset]["summaries"].keys():
                    summary = " ".join(tac[docset]['summaries'][summarizer]['sentences']) 
                    # no need for [0] since we changed the format of jsonfile
                    summary = summary.replace("\n", " ")
                    summary = summary.replace("\t", " ")
                    if len(summary) == 0:
                        summary = " ."
                    
                    label = scorer([article], [summary]).detach().cpu().numpy()[0][0]
                    f.write(str(label) + "\n")

def evaluate_newsroom(csv_file, output_path, scorer):
    with open(output_path, "w") as f:
        with open(csv_file, "r", encoding="utf-8") as csvfile: 
            reader = csv.reader(csvfile, delimiter=",", quotechar="\"") 
            counter = 0 
            for row in reader: 
                if counter > 0:
                    [_doc, _sum] = row[2:4]
                    _doc = _doc.replace("</p><p>", "")
                    _sum = _sum.replace("</p><p>", "")
                    _doc=html.unescape(_doc) 
                    _sum=html.unescape(_sum) 

                    label = scorer([_doc], [_sum]).detach().cpu().numpy()[0][0]
                    f.write(str(label) + "\n")
                counter += 1

def evaluate_realsumm(tsv_file, output_path, scorer):
    with open(output_path, "w") as f:
        with open(tsv_file, "r", encoding="utf-8") as tsv:
            for line in tsv:
                line = line.split('\t')
                _doc = ' '.join(line[0].split())
                for j in range(1, len(line)) :
                    _sum = ' '.join(line[j].split())
                    
                    label = scorer([_doc], [_sum]).detach().cpu().numpy()[0][0]
                    f.write(str(label) + "\n")

if __name__ == "__main__":
    DATASET='billsum'
    CKPT_PATH = os.path.join(CFG.RESULT_ROOT, DATASET, CFG.METHOD, "model.pth")

    scorer = Scorer()
    scorer = scorer.load_state_dict(torch.load(CKPT_PATH))
    scorer.to(CFG.DEVICE)
    
    print("Evaluating...")
    evaluate_tac("TAC2010_all.json", os.path.join(CFG.RESULT_ROOT, DATASET, CFG.METHOD, "test_results_tac.tsv"), scorer)
    evaluate_newsroom("newsroom-human-eval.csv", os.path.join(CFG.RESULT_ROOT, DATASET, CFG.METHOD, "test_results_newsroom.tsv"), scorer)
    evaluate_realsumm("realsumm_100.tsv", os.path.join(CFG.RESULT_ROOT, DATASET, CFG.METHOD, "test_results_realsumm.tsv"), scorer)
   