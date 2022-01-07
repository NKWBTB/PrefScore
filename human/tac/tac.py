# This script processes TAC results 

# Where are the files (using 2010 as example)
# 1. Documents: TAC2010_Summarizatioon_Documents.tgz 
#               -> GuidedSumm10_test_docs_files.tar.gz
#               -> 46 (NIST doc sometimes says 44) folders (e.g., D1022D)
#               -> two folders one for A and one for B (e.g., D1022D-B, D1022D-A)
# 2. Summaries: 
# 
#NIST assessors wrote 4 model summaries for each docset.
# The NIST human summarizer IDs are A-H.

#NIST received 41 runs from 23 participants for the guided
#summarization task.  The participants each submitted up to two runs,
#and their summarizer IDs are 3-43.

# Two baseline summarizer: Leadword (ID=1) and MEAD (ID=2)
# 3. Scores: 

import os, statistics, glob, os.path
import bs4 # beautifulsoup
import numpy as np
import json
import re

# Human summarizer ID

def parse_tac_article(filename, sentence_delimiter):
    """Turn one TAC article in HTML as a list of string
    
    In the source format, each <p> tag is one sentence. 

    The strings may contain special characters such as end-of-line. 

    """
#    print (filename)
    #print(filename)
    article = []
    with open(filename, encoding="UTF-8", errors='ignore') as f:
        s = bs4.BeautifulSoup(f, "html.parser")
#        article = sentence_delimiter.join([p.get_text() for p in s.find_all("p")])
        
        label_p = [p.get_text() for p in s.find_all("p")]
        
        label_text = [] if len(label_p) != 0 else [text.get_text() for text in s.find_all("text")]
         
        #print(label_text)
        article.extend(label_p)
        article.extend(label_text)

#        article = article.replace("\n", " ")

    return article 

def get_articles(dataset_path, setIDs, sentence_delimiter):
    """Extract articles from 2008-2010 updated/guided summary tasks 

    dataset_path: str, path to the parent directory of all docsets.
       Under the path, there are many folders, like D1022D, D1042H
        
    setIDs: list of str, consisting of "A" or "B"
        Most TAC summarization tasks uses two sets,
        each set consisting of 10 news articles

    sentence_delimiter: str, e.g.,"\n\n"
    NOT IN USE


    return:
        dict, keys as document set (10 articles), values as list of 10 lists of strings 

    {"docset1": [[sent1, sent2, ...], # 1st article
                [sent1, sent2, ...], # 2nd article
                ...
                [sent1, sent2, ...]]# 10th article , 
    "docset2": [[sent1, sent2, ...], # 1st article
                [sent1, sent2, ...], # 2nd article
                ...
                [sent1, sent2, ...]]# 10th article,
    ...
    "docset n": ...
    }


    File structure: TAC20{08,09,10}_Summarization_Documents{.tgz}
                    |___> GuidedSumm{}_test_docs_files{.tar.gz}  (==dataset_path)
                          |___> D1001A
                                |___> D1001A-A (docset name is D1001-A)
                                      |___> 10 HTML files
                                |___> D1001A-B  (docset name is D1001-B, where the A is the NIST staff who picked this news)
                                      |___> 10 HTML files
                          |___> D1002A
                          ...
                          |___> D1046H 

    Todo: TAC2011 articles are released as indexes in Gigiword datasets. So a different function is needed. 

    """
    articles = {} 
    for docset in os.listdir(dataset_path):
        for set_suffix in setIDs:
            docset_folder = docset + "-" + set_suffix
            docset_name = docset[:-1] + "-" + set_suffix  # drop the human picker's name, e.g., D1001A -> D1001 
            docset_path = os.path.join(dataset_path, docset, docset_folder)
            for doc in os.listdir(docset_path):
                article = parse_tac_article(os.path.join(docset_path, doc), sentence_delimiter)
                articles.setdefault(docset_name, []).append(article)
    return articles 

def get_statistics(articles):
    """

    articles: dict, keys as document set (10 articles form one docset), 
                    values as list of strings

    """
    c, w, s = [], [], [] # number of characters, words, and sentences
    for docset, docs in articles.items():
        for doc in docs:
            # if type(doc) == str: # each doc is a length string
            #     c.append(len(doc)) 
            #     w.append(len(doc.split(" ")))
            #     s.append(len(doc.split(". ")))
            # elif type(doc) == list: #each doc is a list of strings
                c.append(sum(map(len, doc))) 
                w.append(sum([len(sent.split(" ")) for sent in doc]))
                s.append(len(doc))

            
#    dist = [round(q, 1) for q in statistics.quantiles(lengths, n=10)] # need python3.8
    
    for name, quantity in zip(["char", "word", "sent"], [c,w,s]):
        print (name, " quantile:",[int(np.percentile(quantity, x)) for x in range(1, 101, 10)])
    return c,w,s

def get_summaries(dataset_path, setIDs, sentence_delimiter, summary_types):
    """Extract summaries from 2008-2011 updated/guided summary tasks 

    dataset_path: str, path to a ROUGE (instead of peers or BE) directory, 
                 under which there are two folders: peers and models. 

    summary_types: list of strs, subsets of ["peers", "models"]

    setIDs: list of str, consisting of "A" or "B"
        Most TAC summarization tasks uses two sets,
        each set consisting of 10 news articles

    sentence_delimiter: str, e.g., " ---- " or "\n\n"
    NOT IN USE

    return:
        dict, keys as document set (str, e.g., "D1001-B"), 
              values as dict, whose
                              keys as summarizer ID (str, e.g., "E" ), 
                              values as a list of strs, (summaries from 4 humans and 43 summarizers)
                              Each summary is a list of strings.

            Return may contain escape sequences 

        "docset 1": {"summarizer1": [sent1, sent2, ...], 
                    "summarizer2": [sent1, sent2, ...],
                    ...
                    "summarizer43": [sent1, sent2, ...],
                    "summarizerA": [sent1, sent2, ...],
                    ... # only 4 humans, could be any 4 between A and H 
                    "summarizerH": [sent1, sent2, ...], 
                    }, 
        "docset 2": {... },

        ...

        "docset n": {...}, 

 
    File structure: GuidedSumm20{08,09,10,11}_eval{.tgz}
                    |___> manual
                    |___> ROUGE  (==dataset_path)
                          |___> peers (summaries by machine summarizers)
                                |___> D1001-A.M.100.A.1 (leadword)
                                |___> D1001-A.M.100.A.2 (MEAD)
                                ...
                                |___> D1001-A.M.100.A.43 (3 to 43 are TAC participating summarizers)
                          |___> models (summaries by humans, 4 NIST staffers out of 8 total, A-H)
                                |___> D1001-A.M.100.A.{A-H} 
                    |___> BE    

    TODO: 
    Forrest: Please check whether it is one sentence per line. 

   """
    summaries = {} 
    for summary_type in summary_types:
        for summary_file in glob.glob(os.path.join(dataset_path,summary_type, "*")):
#                    print (summary_file)
                    [docset_name, _, _, _, summarizer] = os.path.basename(summary_file).split(".")
                    setID = docset_name.split("-")[1]
                    if setID in setIDs:
                        if docset_name not in summaries:
                            summaries[docset_name] = {}
                        summary = parse_tac_article(os.path.join(dataset_path, summary_type, summary_file), None)
                        if len(summary) == 0:
                            with open(os.path.join(dataset_path, summary_type, summary_file), 
                                    encoding="utf8", errors='ignore') as f:
                                summary = f.readlines()
                        # summaries[docset_name].setdefault(summarizer, []).append(summary)
                        summaries[docset_name][summarizer] = summary # each summarizer writes only one summary per document set
    return summaries 
    

def get_scores(score_path, summary_types, setIDs):
    """Extract scores for each summarizers, both machine and human, for each document set 

    Peer file format:
        D1001-A	1	0.286	5	0	A	A	0.276	4	2
        D1001-A	2	0.179	3	0	A	A	0.172	4	2
    MOdel file format:
        ['D1001-A', 'A', '16', 'A', 'A', '0.905', '5', '5']

    See README_EVAL.txt to confirm 

    score_path: str, path to a manual evaluation results folder, e.g., 
                  TAC2010/GuidedSumm2010_eval/manual

    summary_types: list of strs, subsets of ["peers", "models"]
 
    setIDs: list of str, consisting of "A" or "B"
        Most TAC summarization tasks uses two sets,
        each set consisting of 10 news articles


    """
    scores = {}
    for summary_type in summary_types:
        summary_type = summary_type[:-1] # drops the plural s 
        for setID in setIDs:
            scorefile = os.path.join(score_path, ".".join(["manual", summary_type, setID]))
            with open(scorefile) as f:
                for line in f:
                    l = line.split()
                    setID = l[0]
                    summarizer = l[1]
                    if setID not in scores:
                            scores[setID] = {}
                    if summary_type == "peer":
                        #pyramid_score = float(l[2])
                        modified_score = float(l[7])
                        #modified_score = pyramid_score
                        linguistic_quality = int(l[8])
                        overall_score = int(l[9])
                    elif summary_type == "model":
                        modified_score = float(l[5])
                        # modified_score = "0"
                        linguistic_quality = int(l[6])
                        overall_score = int(l[7])
                    else:
                        print ("wrong summarizer")
                        exit()
                    scores[setID][summarizer] = [modified_score, linguistic_quality, overall_score]

    return scores

def dump_data(articles, summaries, scores, dump_to):
    """combine articles, summaries, and scores into one dictionary and dump as JSON

    final structure:

        {"docset 1":
                    {"articles": [[sent1, sent2, ...]
                                  ...
                                  # 1 to 10 articles in this docset 
                                 ] # end of articles in this docset 

                     "summaries": {"summarizer 1": 
                                                  {"sentences": [sent1, sent2, ...]
                                                   "scores": [score1, score2, ...]
                                                  }
                                   "summarizer 2": { "sentences": [...]
                                                     "scores": [...]
                                                    }
                                    ....
                                   # 43 + 4 machine and human summarizers
                                  } # end of summaries and scores for this docset
                    } # end of the 1st docset 
         "docset 2": 
                    ...

        }


    """
    combined  = {}
    for docID, summary_dict in summaries.items():
        combined[docID] = {}
        combined[docID]["articles"] = articles[docID]
        combined[docID]["summaries"]={}
        for summarizer, summary_sentences in summary_dict.items():
            combined[docID]["summaries"][summarizer] = {}
            combined[docID]["summaries"][summarizer]["sentences"] = summary_sentences
            combined[docID]["summaries"][summarizer]["scores"] = scores[docID][summarizer]

    parsed = json.dumps(combined, indent=4, sort_keys=True, separators=(',', ': '))
    with open(dump_to, 'w') as f:
        f.write(parsed)

    return combined 

def get_rouge(filepath, dump_to=None):
    """Load TAC2010 ROUGE score results 


    filepath: str, GuidedSumm2010_eval/ROUGE/rouge_A.m.out


    Structure of scores:
             ('D1035-A', '28'): {'1': (0.32374, 0.34439, 0.33375),
                                  '2': (0.05569, 0.05928, 0.05743),
                                  '3': (0.01467, 0.01562, 0.01513),
                                  '4': (0.00741, 0.00789, 0.00764),
                                  'L': (0.26619, 0.28316, 0.27441),
                                  'SU4': (0.09885, 0.10533, 0.10199),
                                  'W-1.2': (0.08912, 0.17609, 0.11835)},

    Structure of ordered_scores 
        ('D1035-A', '28'): [0.32374,
          0.34439,
          0.33375,
          0.05569,
          0.05928,
          0.05743,
          0.01467,
          0.01562,
          0.01513,
          0.00741,
          0.00789,
          0.00764,
          0.26619,
          0.28316,
          0.27441,
          0.09885,
          0.10533,
          0.10199,
          0.08912,
          0.17609,
          0.11835],


    return: dict, keys as a tuple (docset, summarizer), 
                  values as 21 ROUGE scores, 
                  which are Recall, precision, F1 for Rouge 1, 
                  then  Recall, precision, F1 for Rouge 2, 
                  then ... for                    Rouge 3,
                  then   ........             for Rouge 4,
                  then   .......              for Rouge L
                  then   .......              for Rouge SU4
                  then   .......              for Rouge W-1.2




    """
    scores = {}
    ROUGE_types = ["1", "2", "3", "4", "L", "SU4", "W-1.2"]
    with open(filepath) as f:
        for line in f:
            m=re.match(r'(\d+) ROUGE-([\d\w\-\.]+) Eval D(\d+)-A.M.100.[A-H].(\d{1,2}) R:0.(\d{5}) P:0.(\d{5}) F:0.(\d{5})\n', line)
            if m!=None:
                (summarizer, ROUGE_type, docset_number, _, R, P, F) = m.groups()
                docsetID = "D"+docset_number+"-A" # only deal with set A
                if (docsetID, summarizer) not in scores:
                    scores[(docsetID, summarizer)] = {}
                scores[(docsetID, summarizer)][ROUGE_type] = (float("."+R), float("."+P), float("."+F))

    ordered_scores = {}
    for key in scores.keys():
        for ROUGE_type in ROUGE_types:
            ordered_scores.setdefault(key, []).extend(scores[key][ROUGE_type])

    if dump_to != None:
        parsed = json.dumps(ordered_scores, indent=4, sort_keys=True, separators=(',', ': '))
        with open(dump_to, 'w') as f:
            f.write(parsed)

    return ordered_scores

if __name__ == "__main__":
    # article_set_path = "F:/Dataset/TAC2010/TAC2010/TAC2010_Summarization_Documents/GuidedSumm10_test_docs_files/"
    # summary_set_path = "F:/Dataset/TAC2010/TAC2010/GuidedSumm2010_eval/ROUGE"
    # score_path = "F:/Dataset/TAC2010/TAC2010/GuidedSumm2010_eval/manual"

    

    article_set_path = "/mnt/insecure/data/TAC/TAC2010/TAC2010_Summarization_Documents/GuidedSumm10_test_docs_files/"
    summary_set_path = "/mnt/insecure/data/TAC/TAC2010/GuidedSumm2010_eval/ROUGE"
    score_path = "/mnt/insecure/data/TAC/TAC2010/GuidedSumm2010_eval/manual"

    dump_to = "TAC2010_all.json"

    rouge_score_path = "/mnt/insecure/data/TAC/TAC2010/GuidedSumm2010_eval/ROUGE/rouge_A.m.out"
    dump_to_rouge = "rouge2010.json"

    setIDs = ["A"]  # we only use set A because set B is not applicable 
    sentence_delimiter = "  "
    summary_types = ["peers", "models"]
    
    
    articles = get_articles(article_set_path, setIDs, sentence_delimiter)
    _,_,_ = get_statistics(articles)

    summaries = get_summaries(summary_set_path, setIDs, sentence_delimiter, summary_types)
                                                # sentence_delimiter,  NOT IN USE 

    scores = get_scores(score_path, summary_types, setIDs)

    combined = dump_data(articles, summaries, scores, dump_to=dump_to)
    

    #rouge_scores = get_rouge(rouge_score_path, dump_to_rouge)
#    print (rouge_scores)
