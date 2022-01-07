import numpy as np
import utils
import pickle
import copy
import os
from scipy.stats.stats import pearsonr, spearmanr, kendalltau

def merge_results(result_root, training_sets, include_old=True):
    """
    Merge the results of baselines and ours into the same dictionary
    """
    sd_abs_path = "abs.pkl"
    sd_ext_path = "ext.pkl"
    sd_abs = pickle.load(open(sd_abs_path, "rb"))
    sd_ext = pickle.load(open(sd_ext_path, "rb"))
    sd = copy.deepcopy(sd_abs)
    for doc_id in sd:
        isd_sota_ext = sd_ext[doc_id]
        isd_sota_ext['system_summaries']['bart_out_ext.txt'] = isd_sota_ext['system_summaries']['bart_out.txt']
        sd[doc_id]['system_summaries'].update(isd_sota_ext['system_summaries'])
        del isd_sota_ext['system_summaries']['bart_out.txt']

    abs_systems = sd_abs[1]['system_summaries'].keys()
    ext_systems = sd_ext[1]['system_summaries'].keys()

    def merge_one(filepath, metric_name):
        with open(filepath, "r", encoding="utf-8") as f:
            for doc_id in sd:
                for sys_name, system in sd[doc_id]["system_summaries"].items():
                    score = float(f.readline())
                    
                    if sys_name in abs_systems:
                        sd_abs[doc_id]["system_summaries"][sys_name]["scores"][metric_name] = score
                    elif sys_name in ext_systems:
                        sd_ext[doc_id]["system_summaries"][sys_name]["scores"][metric_name] = score
                    else:
                        print("???")
                        assert(False)
    
    # Merge results in the predictions folder
    if include_old:
        pred_path = 'predictions'
        preds = os.listdir(pred_path)
        preds = [tsvfile for tsvfile in preds if tsvfile.endswith('.tsv')]
        for tsvfile in preds:
            metric_name = "A_" + tsvfile.split('.')[0] if not tsvfile.startswith("metric") else tsvfile.split('.')[0]
            merge_one(os.path.join(pred_path, tsvfile), metric_name)
    
    # Merge results in the exp folder
    for training_set in training_sets:
        methods = os.listdir(os.path.join(result_root, training_set))
        for method in methods:
            prediction_tsv = os.path.join(result_root, training_set, method, "test_results_realsumm.tsv")
            if os.path.exists(prediction_tsv):
                metric_name = "B_{}_{}".format(training_set, method)
                merge_one(prediction_tsv, metric_name)
    
    return sd_abs, sd_ext

def calc_corr(level, method, pair, sd, systems):
    
    corr = 0
    if level == "pool":
        x = []
        y = []
        for doc_id in sd:
            for sys_name, sys in sd[doc_id]["system_summaries"].items():
                x.append(sys["scores"][pair[0]])
                y.append(sys["scores"][pair[1]])
        corr = method(x, y)[0]
    elif level == "summary":
        for doc_id in sd:
            x = [sys["scores"][pair[0]] for sys_name, sys in sd[doc_id]["system_summaries"].items()]
            y = [sys["scores"][pair[1]] for sys_name, sys in sd[doc_id]["system_summaries"].items()]
            corr += method(x, y)[0]
        corr /= len(sd)
    elif level == 'system':
        x = [scores[pair[0]] for sys_name, scores in systems.items()]
        y = [scores[pair[1]] for sys_name, scores in systems.items()]
        corr = method(x, y)[0]
    else:
        print("???")
    return corr

def main():
    # Configurations 
    result_root = "../../exp/result_bert_base_uncased"
    training_sets = os.listdir(result_root)
    level="system"

    sd_abs, sd_ext = merge_results(result_root, training_sets, False)

    for dataset, name in [(sd_abs, "abs"), (sd_ext, "ext")]:
        sd = dataset
        mlist = utils.get_metrics_list(sd)
        all_pairs = [('litepyramid_recall', m) for m in mlist if m != 'litepyramid_recall']
        systems = utils.get_system_level_scores(sd, mlist, agg='mean')
        print("---------------{}--------------".format(name))
        for pair in all_pairs: print(pair[1])
        for pair in all_pairs:
            for method in [pearsonr, spearmanr, kendalltau]:
                corr = calc_corr(level, method, pair, sd, systems)
                print("%.4f" % corr, end=" ")
            
            print("")



if __name__ == '__main__':
    main()