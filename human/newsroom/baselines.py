import csv
import html
import json
import difflib
import os
from anyascii import anyascii
from tqdm import tqdm
from summ_eval.bleu_metric import BleuMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.s3_metric import S3Metric    # Use sklearn 0.21.X
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.rouge_metric import RougeMetric

def pair_title(id2sample, test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            sample = json.loads(line)
            target_str = sample["title"]
            for k in id2sample:
                if "title" in id2sample[k]: continue
                source_str = id2sample[k]["source_title"]
                if difflib.SequenceMatcher(a=target_str, b=source_str).quick_ratio() > 0.9:
                    id2sample[k].update(sample)

    for k in id2sample:
        id2sample[k]["summary"] = anyascii(id2sample[k]["summary"])
        if not "title" in id2sample[k]:
            print(k, id2sample[k]["source_title"])

def main(dump=False, calc=True):
    WORKERS = 6
    scorers = [RougeMetric(), CiderMetric(), BleuMetric(n_workers=WORKERS), S3Metric(n_workers=WORKERS), MeteorMetric(), BertScoreMetric(), MoverScoreMetric(version=2)]

    in_file = 'newsroom-human-eval.csv'
    test_file = "test-stats.jsonl"

    ids = []
    id2sample = {}
    sums = []
    counter = 0 
    with open(in_file, 'r', encoding="utf-8") as csvfile: 
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        for row in reader: 
            if counter > 0:
                id, sysname, article, summary, title, coh, flu, inf, rel = row
                summary = summary.replace("</p><p>", "")
                summary= html.unescape(summary) 
                sums.append(summary.strip())
                ids.append(id)
                id2sample[id] = {"source_title": html.unescape(title)}
            counter += 1
    
    pair_title(id2sample, test_file)

    if dump:
        with open("test.json", "w") as f:
            json.dump(id2sample, f, indent=2)

    if calc:
        Refs = []
        Hyps = []

        for i in range(len(ids)):
            Hyps.append(sums[i])
            Refs.append([id2sample[ids[i]]["summary"]])
        
        for scorer in scorers:
            print(type(scorer))
            scores = scorer.evaluate_batch(Hyps, Refs, aggregate=False)
            scorer_names = list(scores[0].keys())
            for scorer_name in scorer_names:
                with open(os.path.join("predictions", "metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
                    for score in scores:
                        f.write(str(score[scorer_name])+"\n")


if __name__ == "__main__":
    main()
