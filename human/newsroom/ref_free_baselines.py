import json, os, re, csv, html
from summ_eval.supert_metric import SupertMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.blanc_metric import BlancMetric

def main():
    # Fix SummaQA summa_qa_utils.py:20 code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering
    scorers = [SupertMetric(), SummaQAMetric(), BlancMetric(inference_batch_size=32)]

    in_file = 'newsroom-human-eval.csv'

    docs = []
    sums = []
    counter = 0 
    with open(in_file, 'r') as csvfile: 
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        for row in reader: 
            if counter > 0:
                [_doc, _sum] = row[2:4]
                _doc = _doc.replace("</p><p>", "")
                _sum = _sum.replace("</p><p>", "")
                _doc=html.unescape(_doc) 
                _sum=html.unescape(_sum) 
                docs.append(_doc.strip())
                sums.append(_sum.strip())
            counter += 1
    
    for scorer in scorers:
        scores = scorer.evaluate_batch(sums, docs, aggregate=False)
        scorer_names = list(scores[0].keys())
        for scorer_name in scorer_names:
            with open(os.path.join("predictions", "metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
                for score in scores:
                    f.write(str(score[scorer_name])+"\n")
            
if __name__ == '__main__':
    main()
