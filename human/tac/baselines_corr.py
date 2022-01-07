import math
from tac import *
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
import json

def calc_corr(metric_score, scores, rscore_type, hscore_type, method=pearsonr, level="pool"):
    """ Calculate correlation between metrics and human evaluation score
        
        metric_score: dict, scores for metrics
        scores: dict, scores for human evaluation   
        rscore_type: int, the number of metrics
        hscore_type: int, the number of categories in human evaluation
        methods: function, the correlation methods
        level: str, the level used for calculation, can be "pool" or "summary" or "system"

        return: numpy array of shape (rscore_type, hscore_type)
    """
    corr = np.zeros((rscore_type, hscore_type), dtype=np.float32)
    if level == 'pool':
        x = [ [metric_score[key][i] for key in metric_score.keys()] for i in range(rscore_type)]
        y = [ [scores[doc][summarizer][i] for doc, summarizer in metric_score.keys()] for i in range(hscore_type)]
        
        for i in range(rscore_type):
            for j in range(hscore_type):
                corr[i, j] = method(x[i], y[j])[0]
    elif level == 'summary':
        for i in range(rscore_type):
            for j in range(hscore_type):
                for doc in scores:
                    x = [metric_score[(doc, summarizer)][i] for summarizer in scores[doc] if summarizer.isnumeric()]
                    y = [scores[doc][summarizer][j] for summarizer in scores[doc] if summarizer.isnumeric()]
                    # Caculate Correlation for each of the document
                    corr[i, j] += method(x, y)[0]
        
        # Mean value across docs
        corr /= len(scores)
    
    elif level == 'system':
        for i in range(rscore_type):
            for j in range(hscore_type):
                x_all = []
                y_all = []
                for doc in scores:
                    x_all.append([metric_score[(doc, summarizer)][i] for summarizer in scores[doc] if summarizer.isnumeric()])
                    y_all.append([scores[doc][summarizer][j] for summarizer in scores[doc] if summarizer.isnumeric()])
                x_all = np.array(x_all)
                y_all = np.array(y_all)
                
                # Aggregate score over documents by mean
                assert(x_all.shape == y_all.shape)
                x = np.mean(x_all, axis=0)
                y = np.mean(y_all, axis=0)
                corr[i, j] += method(x, y)[0]
    else:
        print("???")
        assert(False)

    return corr

def main():
    TAC_result_root = "/home/gluo/Dataset/TAC2010"
    score_path = TAC_result_root + "/GuidedSumm2010_eval/manual"
    rouge_score_path = TAC_result_root + "/GuidedSumm2010_eval/ROUGE/rouge_A.m.out"

    output_file = "rouge_score.tsv"

    setIDs = ["A"]
    summary_types = ["peers", "models"]

    hscore_type = 3
    rscore_type = 21
    
    scores = get_scores(score_path, summary_types, setIDs)
    rouge_scores = get_rouge(rouge_score_path)

    method = kendalltau
    level = 'summary'
    
    output = calc_corr(rouge_scores, scores, rscore_type, hscore_type, method, level)

    with open(output_file, "w", encoding="UTF-8") as f:
        for line in output:
            strline = "\t".join([str(val) for val in line]) + "\n"
            f.write(strline)

    # Correlation for other baselines
    baselines_score_path = "baselines.json" # or "baselines_ref_free.json"
    baselines_score = {}
    with open(baselines_score_path, "r", encoding="utf-8") as f:
        baselines_score = json.load(f)

    akey = list(baselines_score.keys())[0]
    score_type = list(baselines_score[akey].keys())
    rscore_type = len(score_type)

    converted_score = {}
    for doc_file in baselines_score:
        score_list = []
        for score in score_type:
            v = baselines_score[doc_file][score]
            if v is None or math.isinf(v) or math.isnan(v):
                v = 0.0
            score_list.append(v)

        doc_string = doc_file.split('.')
        doc = doc_string[0]
        summarizer = doc_string[-1]
        converted_score[(doc, summarizer)] = score_list
    
    output = calc_corr(converted_score, scores, rscore_type, hscore_type, method, level)
    with open("baselines_score.tsv", "w", encoding="UTF-8") as f:
        lc = 0
        for line in output:
            strline = score_type[lc] + "\t" + "\t".join([str(val) for val in line]) + "\n"
            f.write(strline)
            lc += 1

if __name__ == "__main__":  
    main()