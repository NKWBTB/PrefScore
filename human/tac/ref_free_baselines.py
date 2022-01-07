import json, os, re
from summ_eval.supert_metric import SupertMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.blanc_metric import BlancMetric

def main():
    # Fix SummaQA summa_qa_utils.py:20 code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering
    scorers = [SupertMetric(), SummaQAMetric(), BlancMetric(inference_batch_size=32)]

    in_file = '../../bert/TAC2010_all.json'

    tac = {}
    with open(in_file, "r", encoding="utf-8") as f:
        tac = json.load(f)

    docs = []
    sums = []
    for doc_id in tac:
        article = " ".join([" ".join(text) for text in tac[doc_id]["articles"]])
        article = article.replace("\t", " ")
        article = article.replace("\n", " ")
        article = re.sub(" +", " ", article)

        for sum_id, content in tac[doc_id]["summaries"].items():
            summary = " ".join(content["sentences"])
            summary = summary.replace("\t", " ")
            summary = summary.replace("\n", " ")
            summary = re.sub(" +", " ", summary)
            
            sums.append(summary)
            docs.append(article)

    results = {}
    
    for scorer in scorers:
        scores = scorer.evaluate_batch(sums, docs, aggregate=False)
        cid = 0
        for doc_id in tac:
            for sum_id in tac[doc_id]["summaries"]:
                results.setdefault(".".join([doc_id, sum_id]), {}).update(scores[cid])
                cid += 1
    
    with open('baselines_ref_free.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)
            
if __name__ == '__main__':
    main()
