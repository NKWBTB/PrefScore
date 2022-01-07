# Collect and analyze results on newsroom dataset 

# Newsroom raw CSV foramt: 
# ArticleID,System,ArticleText,SystemSummary,ArticleTitle,CoherenceRating,FluencyRating,InformativenessRating,RelevanceRating

#%% 
import csv 
import scipy.stats
import os
# import numpy 

def mean(a_list):
    return sum(a_list)/len(a_list)
    # return numpy.mean(a_list)

def median(a_list):
    a_list.sort()
    return a_list[len(a_list)//2]
    # return numpy.median(a_list)

def load_newsroom_and_ours(newsroom_csv, prediction_tsv):
    """Load newsroom's human evaluation CSV file and run_classifier.py's prediction using our models. 
    The correspondence between documents and summaries in the prediction file is determined by to_tsv.py file. 

    Newsroom-csv is downloaded from
    https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv

    Return data structure: 
    {
        docID: {
            system1:{
                "Coherence":       float, 
                "Fluency":         float,
                "Informativeness": float,
                "Relevance":       float,
                "Ours":            float
            }
            system2: { ... }
            ...
            system7: {... }
        }
    }
    """

    predictions = [float(x.strip()) for x in open(prediction_tsv).readlines()]

    counter = -1 
    scores = {}  
    with open(newsroom_csv) as csvfile: 
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        for row in reader: 
            if counter > -1:
                docID, system = row[0:2]
                Coherence, Fluency, Informativeness, Relevance = list(map(float, row[5:9]))

                score_local = {
                            "Coherence" : [Coherence],
                            "Informativeness": [Informativeness],
                            "Fluency"  : [Fluency],
                            "Relevance": [Relevance]
                        }

                if docID not in scores: 
                    scores[docID] = {system: score_local 
                    }
                elif system not in scores[docID]:
                    scores[docID][system] = score_local 
                else: 
                    for aspect, score_list in score_local.items():
                        scores[docID][system][aspect] += score_list

                scores[docID][system]["Ours"] = [predictions[counter]]
            counter += 1

    return scores 

# Test 
# scores = load_newsroom_and_ours("./newsroom-human-eval.csv", "/mnt/12T/data/NLP/anti-rogue/result_base_sent/billsum/cross_sent_delete/test_results_newsroom.tsv")

# %% 
def summary_judge(scores, metrics_newsroom, metrics_other, concensus, correlation_types):
    """Summary-level evaluation between newsroom metrics and metrics from another group 

    metrics_newsroom: e.g., ["Coherence", "Informativeness", "Fluency", "Relevance"]
    metrics_other: e.g., ["Ours"]    # only ours for now. we can add others, e.g., ours from different training sets and/or negative sampling methods 
    correlation_type: str, "pearsonr", "spearmanr", or "kendalltau"
    concensus: string, "median", "mean", or "all" 

    """
    def get_correlation_per_doc_given_metrics(scores_of_one_document, metric_newsroom, metric_other, concensus, correlation_type):
        """

        scores of one document: 
        {
            system1:{
                "Coherence":       float, 
                "Fluency":         float,
                "Informativeness": float,
                "Relevance":       float,
                "Ours":            float
            }
            system2: { ... }
            ...
            system7: {... }
        }

        """
        vector_newsroom = [] # scores from a newsroom metric
        vector_other = []    # scores from a non-newsroom metric 
        for system, score_local in scores_of_one_document.items():
            score_newsroom = score_local[metric_newsroom] # list of 3 floats
            score_other = score_local[metric_other] # list of ONE float (TODO: need to expand to allow more than one float for multi-score metrics)
            if concensus in ["median", "mean", "max", "min"]:
                score_newsroom = eval(f"{concensus}(score_newsroom)")
                # if concensus == "median":
                #     score_newsroom.sort() 
                #     score_newsroom = score_newsroom[1] # type change 
                # elif concensus == "mean":
                #     score_newsroom = mean(score_newsroom) # type change 

                vector_newsroom.append(score_newsroom)
                vector_other   .append(score_other[0])
            elif concensus == "all":
                vector_newsroom += score_newsroom
                vector_other    += score_other*len(score_newsroom) # just duplicate 

        if (max(vector_newsroom) == min(vector_newsroom)):
            # A workaround to avoid constant vector issue for pearsonr
            vector_newsroom[0] += 0.001 
        return  eval(f"scipy.stats.{correlation_type}(vector_newsroom, vector_other)")[0]

    # now begins the summary-level judge 
    correlations  = {}
    for correlation_type in correlation_types:
        # print (correlation_type)
        correlations[correlation_type] = {}
        for metric_newsroom in metrics_newsroom: # one metric from newsroom
            for metric_other in metrics_other:  # one metric to evaluate against newsroom 
                correlation_across_documents = [get_correlation_per_doc_given_metrics(scores_of_one_document, metric_newsroom, metric_other, concensus, correlation_type) for scores_of_one_document in scores.values()]
                # a list of floats

                correlations[correlation_type]\
                            [(metric_newsroom, metric_other)] = \
                            mean(correlation_across_documents)
    
    return correlations

# Test 
# correlations = summary_judge(scores, ["Coherence", "Informativeness", "Fluency", "Relevance"], ["Ours"], "mean")

def print_beautiful(correlation, correlation_types, metrics_newsroom):
    """
    correlation looks like this:
    {'kendalltau': {('Coherence', 'Ours'): 0.49600166520307726, ('Informativeness', 'Ours'): 0.5808293864832368, ('Fluency', 'Ours'): 0.45790855761575266, ('Relevance', 'Ours'): 0.4740934359726846}, 'spearmanr': {('Coherence', 'Ours'): 0.5906983676040027, ('Informativeness', 'Ours'): 0.6694288674637746, ('Fluency', 'Ours'): 0.5434449050990235, ('Relevance', 'Ours'): 0.5830224096341955}}
    """
    for correlation_type in correlation_types:
        for metric_newsroom in metrics_newsroom: 
            for metric_other in ["Ours"]:
                print ("{:>1.4}".format(correlation[correlation_type][(metric_newsroom, metric_other)]), end="\t")

    print ("")

def system_judge(scores, metrics_newsroom, metrics_other, concensus, correlation_types):
    """Compute the correlation between two (groups of) metrics at system-level 

    This function is highly similar to summary_judge

    """
    all_system_names = list(scores[list(scores.keys())[0]].keys()) 
    # a safe approach to prevent system name mismatching. 

    def get_correlation_two_metrics(scores, metric_newsroom, metric_other, concensus, correlation_type):

        mean_score_vector_newsroom = []
        mean_score_vector_other = []
        
        for system in all_system_names:
            vector_newsroom = [] # scores from a newsroom metric 
            vector_other = []    # scores from a non-newsroom metric 
            for docID in scores.keys():
                score_local = scores[docID][system]
                score_newsroom = score_local[metric_newsroom] # list of 3 floats
                score_other = score_local[metric_other] # list of ONE Float
                if concensus in ["median", "mean", "max", "min"]:
                    score_newsroom = eval(f"{concensus}(score_newsroom)")
                    vector_newsroom.append(score_newsroom)
                    vector_other   .append(score_other[0])
                elif concensus == "all":
                    vector_newsroom += score_newsroom
                    vector_other    += score_other*len(score_newsroom) # just duplicate

            mean_score_vector_newsroom.append(mean(vector_newsroom))
            mean_score_vector_other   .append(mean(vector_other))
        return  eval(f"scipy.stats.{correlation_type}(vector_newsroom, vector_other)")[0]

    # now begins the system-level judge 
    correlations  = {}
    for correlation_type in correlation_types:
        # print (correlation_type)
        correlations[correlation_type] = {}
        for metric_newsroom in metrics_newsroom: # one metric from newsroom
            for metric_other in metrics_other:  # one metric to evaluate against newsroom 
                correlations[correlation_type]\
                            [(metric_newsroom, metric_other)] = \
                            get_correlation_two_metrics(scores, metric_newsroom, metric_other, concensus, correlation_type)
    
    return correlations

def pooled_judge(scores, metrics_newsroom, metrics_other, concensus, correlation_types):
    """Judge metrics purely based on how they correlation over a pair of document and summary
    """
    def get_correlation_given_two_metrics(scores, metric_newsroom, metric_other, concensus, correlation_type):
        vector_newsroom = [] # scores from a newsroom metric 
        vector_other = []    # scores from a non-newsroom metric 
        for docID in scores.keys():
            for system in scores[docID].keys():
                score_newsroom = scores[docID][system][metric_newsroom] # list of 3 floats
                score_other = scores[docID][system][metric_other] # list of ONE Float
                if concensus in ["median", "mean", "max", "min"]:
                    score_newsroom = eval(f"{concensus}(score_newsroom)")
                    vector_newsroom.append(score_newsroom)
                    vector_other   .append(score_other[0])
                elif concensus == "all":
                    vector_newsroom += score_newsroom
                    vector_other    += score_other*len(score_newsroom) # just duplicate
        return  eval(f"scipy.stats.{correlation_type}(vector_newsroom, vector_other)")[0]

    # now begins the pooled- judge 
    correlations  = {}
    for correlation_type in correlation_types:
        # print (correlation_type)
        correlations[correlation_type] = {}
        for metric_newsroom in metrics_newsroom: # one metric from newsroom
            for metric_other in metrics_other:  # one metric to evaluate against newsroom 
                correlations[correlation_type]\
                            [(metric_newsroom, metric_other)] = \
                            get_correlation_given_two_metrics(scores, metric_newsroom, metric_other, concensus, correlation_type)
    
    return correlations


#%%
def main():
    """

    Assumptions on files and paths: 
    * Results are organized into folders as $DATA_ROOT/training_set/negative_sampling_method/.
    * Newsroom human evaluation groundtruth is ./newsroom-human-eval.csv
    * Prediction from BERT's 

    """

    # Configurations 
    result_root = "../../exp/result_bert_base_uncased"

    # training_sets = ["billsum"] #, "scientific_papers", "big_patent", "cnn_dailymail"]
    training_sets = os.listdir(result_root)
    # methods = ["sent_delete", "sent_replace"]

    concensus_based_on="mean" # "median", "max", "min"
    level = "summary" # "system", "summary", or "pooled"

    metrics_newsroom = ["Coherence", "Informativeness", "Fluency", "Relevance"]
    correlation_types = ["pearsonr", "spearmanr", "kendalltau"]
    # End of configurations

    
    judge_function = {"summary":summary_judge, "system":system_judge, "pooled":pooled_judge}.get(level)

    # header to print 
    short_metrics_newsroom = [x[:3] for x in metrics_newsroom]
    short_correlation_types = [x[0] for x in correlation_types]
    cross_header = ["_".join([x,y]) for x in short_correlation_types for y in short_metrics_newsroom]
    print ("\t".join(["{:<17}".format("training_set"), " neg_sample"]+cross_header))
    
    for training_set in training_sets:
        methods = os.listdir(os.path.join(result_root, training_set))
        for method in methods:
            print (f'{training_set:<17}\t', method, end="\t")
            prediction_tsv = os.path.join(result_root, training_set, method, "test_results_newsroom.tsv")
            if not os.path.exists(prediction_tsv):
                continue
            scores = load_newsroom_and_ours("./newsroom-human-eval.csv", prediction_tsv)

            correlations = judge_function(
                scores, 
                metrics_newsroom, 
                ["Ours"], 
                concensus_based_on, 
                correlation_types
                )
            print_beautiful(correlations, correlation_types, metrics_newsroom)

    # Load scores from baselines
    tsvs = os.listdir('.') 
    tsvs = [tsv for tsv in tsvs if tsv.startswith("metric")]
    for tsv in tsvs:
        print (tsv)
        prediction_tsv = tsv

        scores = load_newsroom_and_ours("./newsroom-human-eval.csv", prediction_tsv)

        correlations = judge_function(
            scores, 
            metrics_newsroom, 
            ["Ours"], 
            concensus_based_on, 
            correlation_types
            )
        print_beautiful(correlations, correlation_types, metrics_newsroom)


# %%

if __name__ == "__main__":
    main()