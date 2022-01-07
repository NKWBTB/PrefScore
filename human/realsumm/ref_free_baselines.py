import json, os
from summ_eval.supert_metric import SupertMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.blanc_metric import BlancMetric

def main():
    # Fix SummaQA summa_qa_utils.py:20 code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering
    scorers = [SupertMetric(), SummaQAMetric(), BlancMetric(inference_batch_size=32)]

    in_file = 'realsumm_100.tsv'

    docs = []
    sums = []
    with open(os.path.join(in_file), 'r', encoding='utf-8') as f:
        for line in f:
            texts = line.strip().split('\t')
            article = texts[0]
            summaries = texts[1:]
            
            docs.extend([article] * len(summaries))
            sums.extend(summaries)
    
    for scorer in scorers:
        scores = scorer.evaluate_batch(sums, docs, aggregate=False)
        scorer_names = list(scores[0].keys())
        for scorer_name in scorer_names:
            with open(os.path.join("predictions", "metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
                for score in scores:
                    f.write(str(score[scorer_name])+"\n")
            
if __name__ == '__main__':
    main()
