import xml2dict, json, os
from summ_eval.bleu_metric import BleuMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.s3_metric import S3Metric    # Use sklearn 0.21.X
from summ_eval.meteor_metric import MeteorMetric, enc
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

    eval_path = '/home/gluo/Dataset/TAC2010/GuidedSumm2010_eval/ROUGE/'
    in_file = 'rouge_A.in'

    xml = ''
    with open(os.path.join(eval_path, in_file), 'r', encoding='utf-8') as f:
        xml = f.read()

    eval = xml2dict.parse(xml)
    eval = eval["ROUGE_EVAL"]["EVAL"]

    results = {}
    Refs = []
    Hyps = []
    for exp in eval:
        print(exp["@ID"])
        MODELS = exp["MODELS"]["M"]
        PEERS = exp["PEERS"]["P"]
        
        refs = []
        for model in MODELS:
            model_file = model["#text"]
            with open(os.path.join(eval_path, 'models', model_file), 'r', encoding='cp1252') as f:
                refs.append(f.read())
        
        for peer in PEERS:
            peer_file = peer["#text"]
            print(peer_file)
            peer_text = ''
            with open(os.path.join(eval_path, 'peers', peer_file), 'r', encoding='cp1252') as f:
                peer_text = f.read()

            # Evaluate one at a time
            # results[peer_file] = calc_one(peer_text, refs, scorers)
            # results[peer_file] = {}
            
            Hyps.append(peer_text)
            Refs.append(refs.copy())

    for scorer in scorers:
        print(type(scorer))
        scores = scorer.evaluate_batch(Hyps, Refs, aggregate=False)
        cid = 0
        for exp in eval:
            PEERS = exp["PEERS"]["P"]
            for peer in PEERS:
                peer_file = peer["#text"]
                results.setdefault(peer_file, {}).update(scores[cid])
                # print(scores[cid])
                cid += 1
    
    with open('baselines.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
