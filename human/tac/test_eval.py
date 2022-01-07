import numpy as np
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
import os, json, csv

np.set_printoptions(precision=4)

article_per_set = 10
num_summarizer = 47

def read_tac_test_result(BERT_result_file, tac_json, summarizer_type):
    """Load BERT result of using TAC2010 as test set and average over 10 articles

    By BERT convention, the file name is test_result.tsv 
    46 document sets, 47 summarizers (2 baselines, 41 machine ones, and 4 humans)

    one number per line 

    summarizer_type: str, "human", "machine", or "both"

    Due to how samples are generated, this is the correspondence between lines and doc-sum pairs
    docset_1: 
        article_1:
            summarizer_1, prediction 
            summarizer_2, prediction 
            ...
            summarizer_47, prediction 
        article_2: 
            summarizer_1, prediction 
            summarizer_2, prediction 
            ...
            summarizer_47, prediction 
        ...
        article_10: 
            ...
    docset_2: 
    ...
    docset_46: 

    what we want is:
    docset 1:
        summarizer_1, average of 10 predictions for article_{1,2,...,10}
        summarizer_2, average of 10 predictions for article_{1,2,...,10}
        ...
        summarizer_47, average of 10 predictions for article_{1,2,...,10}
    docset 2:
    ...
    docset_46: ...
    """

    with open(BERT_result_file, "r") as f:
        all_lines = f.read() 

    lines = all_lines.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]  # get rid of last empty line 

    tac = json.load(open(tac_json, 'r'))
    score_dict = {}  # keys as (docset, summarizer), values as list of 10 floats
    docset_counter, article_counter, summarizer_counter = 0,0,0
    # Note that docset_counter, article_counter, nor summarizer_counters is actual docset, article or summarizer IDs. It's just counters to know whether we loop to next article. 
    
    for line in lines:
        docset = list(tac.keys())[docset_counter]
        summarizer = list(tac[docset]["summaries"].keys())[summarizer_counter]
        key = (docset, summarizer)
        if (summarizer_type=="machine" and summarizer.isnumeric()) or \
               (summarizer_type=="human" and summarizer.isalpha()) or \
               (summarizer_type=="both") :
            score_dict.setdefault(key, []).append(float(line))
        
        if summarizer_counter == 47 - 1:
            summarizer_counter = 0
            if article_counter == 10 - 1: 
                article_counter = 0
                docset_counter += 1 
            else:
                article_counter += 1 
        else:
            summarizer_counter += 1 


    # Now, convert to the order in tac and get average 
    score_sorted = [] 
    for docset in tac.keys():
        for summarizer in tac[docset]["summaries"].keys():
            if (summarizer_type=="machine" and summarizer.isnumeric()) or \
               (summarizer_type=="human" and summarizer.isalpha()) or \
               (summarizer_type=="both") : # this condition is redundant but to be safe
                ten_scores = score_dict[(docset, summarizer)]
                avg_score = sum(ten_scores)/len(ten_scores)
                score_sorted.append(avg_score)

    return score_sorted

def load_tac_json(task_json, summarizer_type):
    """Load the human scores from TAC from the JSON file compiled and dumped by our tac.py script 

    task_json: the JSON file containing all TAC samples and their human scores
    summarizer_type: str, "human", "machine", or "both"

    The order of extracting scores from task_json needs to match that in _pop_tac_samples() in run_classifier.py

    order:
    docset_1, summarizer_1, scores[0:2]
    docset_1, summarizer_2, scores[0:2]
    ...
    docset_1, summarizer_47, scores[0:2]

    docset_2, summarizer_1, scores[0:2]
    docset_2, summarizer_2, scores[0:2]
    ...
    docset_2, summarizer_47, scores[0:2]
    ...
    ...
    ...
    docset_46 


    """

    tac_scores = [] # 46 x 47 rows, 3 columns

    tac = json.load(open(task_json, 'r'))
    for docset in tac.keys():
        if summarizer_type == "machine": 
            tac_scores += [ tac[docset]["summaries"][summarizer]["scores"] for summarizer in tac[docset]["summaries"].keys() if summarizer.isnumeric() ]
        elif summarizer_type == "human": 
            tac_scores += [ tac[docset]["summaries"][summarizer]["scores"] for summarizer in tac[docset]["summaries"].keys() if summarizer.isalpha() ]
        elif summarizer_type == "both":
            tac_scores += [ tac[docset]["summaries"][summarizer]["scores"] for summarizer in tac[docset]["summaries"].keys()]

    return tac_scores 

def calc_cc(tac_results, tac_scores, method=pearsonr, level="pool"):
    """Compute the correlation coefficients between BERT results on TAC test set and human evaluated scores on TAC test set

    tac_results: 1-D list of floats, 46(docset)x47(summarizers) elements
    tac_scores: 2-D list of floats, 46(docset)x47(summarizers) rows, and 3 columns
    """
    tac_scores = np.array(tac_scores)
    docs = 46
    summarizers = int(tac_scores.shape[0] / docs)
    
    corr = None
    if level == "pool":
        corr = [method(tac_results, tac_scores[:, i])[0] for i in range(3)]
    elif level == "summary":
        tac_scores = tac_scores.reshape(docs, summarizers, 3)
        tac_results = np.array(tac_results).reshape(docs, summarizers)
        corr = np.zeros((3, ), dtype=np.float32)
        for doc in range(docs):
            corr += np.array([method(tac_results[doc, :], tac_scores[doc, :, i])[0] for i in range(3)])
        corr /= doc
    elif level == 'system':
        tac_scores = tac_scores.reshape(docs, summarizers, 3)
        tac_results = np.array(tac_results).reshape(docs, summarizers)
        
        tac_scores_system = np.mean(tac_scores, axis=0)
        tac_results_system = np.mean(tac_results, axis=0)
        # print(tac_scores.shape, tac_scores_system.shape)
        corr = [method(tac_results_system, tac_scores_system[:, i])[0] for i in range(3)]
    else:
        print("???")
        assert(False)

    line = ["%.5f"%i for i in corr]
    line = "\t".join(line)

    print (line, end = ' ')
    # for i in range(3):
    #     corr_pearson = pearsonr(tac_results, tac_scores[:, i])
    #     corr_spearman = spearmanr(tac_results, tac_scores[:, i])

    #     print(corr_pearson[0], corr_spearman[0])
    # print("---------------------------")


def cc_all(plot = False):
    BERT_result_prefix = "../../exp/result_bert_base_uncased/"
    datasets = os.listdir(BERT_result_prefix)
    tac_json_file = "../../TAC2010_all.json"
    human_only="machine"
    level = "system"

    tac_scores = load_tac_json(tac_json_file, human_only)
    result_dict = {}
    for dataset in datasets:
        methods = os.listdir(os.path.join(BERT_result_prefix, dataset))
        for method in methods:
            BERT_result_file = os.path.join(BERT_result_prefix, dataset, method, "test_results_tac.tsv")
            if not os.path.exists(BERT_result_file):
                continue
            print (dataset, method, end='\t')
            tac_results = read_tac_test_result(BERT_result_file, tac_json_file, human_only)
            result_dict[dataset] = tac_results
            for corr_method in [pearsonr, spearmanr, kendalltau]:
                calc_cc(tac_results, tac_scores, method=corr_method, level=level)
            print("")

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        def plot_results():
            for dataset in datasets:
                sns.kdeplot(result_dict[dataset], label=dataset)
        
        tac_scores = np.array(tac_scores)
        plt.figure(figsize=(12, 3))
        ax = plt.subplot(1, 3, 1)
        plot_results()
        sns.kdeplot(tac_scores[ :, 0], label='Modified')
        ax.legend()
        ax = plt.subplot(1, 3, 2)
        sns.histplot((tac_scores[ :, 1]-1)/4, stat='density', bins=5, label='Linguistic', color='tab:purple')
        plot_results()
        ax.legend()
        ax = plt.subplot(1, 3, 3)
        sns.histplot((tac_scores[ :, 2]-1)/4, stat='density', bins=5, label='Overall', color='tab:purple')
        plot_results()
        ax.legend()
        plt.show()
        
if __name__ == "__main__":
    cc_all()
