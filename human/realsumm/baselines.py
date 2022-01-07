import json, os, copy, re
import pickle
from summ_eval.bleu_metric import BleuMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.s3_metric import S3Metric    # Use sklearn 0.21.X
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.mover_score_metric import MoverScoreMetric
# from summ_eval.rouge_metric import RougeMetric
# rouge = RougeMetric(rouge_args="-n 4 -w 1.2 -m  -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a")
# rouge = RougeMetric(rouge_args="-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a")
# rouge = RougeMetric()
# rouge_dict = bertscore.evaluate_example(hyp, refs)

def main():
    # Fix multi reference for BertScoreMetric & S3Metric
    # Fix model repeatly loading for BertScoreMetric
    WORKERS = 6
    scorers = [CiderMetric(), BleuMetric(n_workers=WORKERS), S3Metric(n_workers=WORKERS), MeteorMetric(), BertScoreMetric(), MoverScoreMetric(version=2)]
    
    sd_abs_path = "abs.pkl"
    sd_ext_path = "ext.pkl"
    sd_abs = pickle.load(open(sd_abs_path, "rb"))
    sd_ext = pickle.load(open(sd_ext_path, "rb"))
    sd = copy.deepcopy(sd_abs)
    for doc_id in sd:
        isd_sota_ext = sd_ext[doc_id]
        isd_sota_ext['system_summaries']['bart_out_ext.txt'] = isd_sota_ext['system_summaries']['bart_out.txt']
        sd[doc_id]['system_summaries'].update(isd_sota_ext['system_summaries'])

    Refs = []
    Hyps = []
    for doc_id in sd:
        refs = []
        ref_summ = sd[doc_id]["ref_summ"]
        ref_summ = ref_summ.replace("<t>", "")
        ref_summ = ref_summ.replace("</t>", "")
        ref_summ = re.sub(" +", " ", ref_summ)
        refs.append(ref_summ.strip())

        for sys_name, system in sd[doc_id]["system_summaries"].items():
            sys_sum = system["system_summary"]
            sys_sum = sys_sum.replace("<t>", "")
            sys_sum = sys_sum.replace("</t>", "")
            sys_sum = re.sub(" +", " ", sys_sum)
            
            Hyps.append(sys_sum)
            Refs.append(refs.copy())
        
    for scorer in scorers:
        print(type(scorer))
        scores = scorer.evaluate_batch(Hyps, Refs, aggregate=False)
        scorer_names = list(scores[0].keys())
        for scorer_name in scorer_names:
            with open(os.path.join("predictions", "metric_"+scorer_name+".tsv"), "w", encoding="utf-8") as f:
                for score in scores:
                    f.write(str(score[scorer_name])+"\n")

if __name__ == '__main__':
    main()
