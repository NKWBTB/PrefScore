# Ordered negative sampling for Siamese model
from unittest import result
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import random
import os, errno
import copy
import string
import numpy
import json
import re
import tensorflow_datasets as tfds
from scipy.spatial.distance import cosine
# from summ_eval.bert_score_metric import BertScoreMetric
# from summ_eval.rouge_metric import RougeMetric
# import embed as E
import evaluate as E
import time
from tqdm import tqdm

# export TOKENIZERS_PARALLELISM=true
metric = E.load("bertscore")
# metric = BertScoreMetric()
# metric = RougeMetric()
# mname = "bert_score_recall"
# embedder = E.SentenceEmbedder()

def combine_shuffle(methods, data_root, dataset_name, split, mode):
    """Combine dumped sample files into one file and shuffle

    cat train_*.tsv > train.tsv
    rm  train_*.tsv 
    cat test_*.tsv > test.tsv
    rm test_*.tsv 
    shuf train.tsv  > train_shuffled.tsv
    shuf test.tsv  > test_shuffled.tsv
    head train_shuffled.tsv -n 120722 > train_shuffled_10_percent.tsv
    head test_shuffled.tsv -n 6707 > test_shuffled_10_percent.tsv
    """
    for method in methods:
        dump_root = os.path.join(data_root, dataset_name.split(':')[0], method + "_" +  mode)
        print ("Combining and shuffling at ", dump_root)
        chops = os.path.join(dump_root, F"{split}_*.jsonl")
        concrete = os.path.join(dump_root, F"{split}.jsonl")
        tmp = os.path.join(dump_root, "tmp.jsonl")

        os.system(F"cat {chops} > {concrete}")
        os.system(F"rm {chops}")
        os.system(F"shuf {concrete} > {tmp}")
        os.system(F"mv {tmp} {concrete}")

def addPeriod(text):
    text = text.strip()
    if text[-1] != '.':
        text += " ."
    return text

def split_pairs(pairs, tokenizer_name="spacy", spacy_batch_size=2**10, n_jobs= 4):
    """For each pair, return the summary as a list of strings 

    tokenizer_name: str, "spacy", "stanza", or "nltk"
    Spacy is about 4 times faster than Stanza --- not fully saturated CPU

    Process-level parallelism impossible with Stanza
    # AttributeError: Can't pickle local object 'CNNClassifier.__init__.<locals>.<lambda>'

    FIXME: interestingly, spacy_batch_size and n_jobs have little effect on Spacy's speed

    Examples
    -------------
    >>> split_pairs([("iam ahjppy asf.","fsd. sdf.fsd. fsd. f")])
    [('iam ahjppy asf.', ['fsd.', 'sdf.fsd.', 'fsd.', 'f'])]
    >>> split_pairs([("i am ahjppy.", "today is monday. food is yummy."), ("the code is hard. the water is cold.", "the number is low. the government sucks. ")], )
    [('i am ahjppy.', ['today is monday.', 'food is yummy.']),
     ('the code is hard. the water is cold.',
      ['the number is low.', 'the government sucks.'])]
    """

    print ("Splitting summaries...", end= " ")
    if tokenizer_name == 'spacy':
        import spacy
        nlp=spacy.load("en_core_web_sm", exclude=["tok2vec",'tagger','parser','ner', 'attribute_ruler', 'lemmatizer'])
        nlp.add_pipe("sentencizer")
        nlp.max_length = 2000000 # default is 1,000,000

        list_summaries = [
            [x.text for x in doc.sents] # sentences in each summary
            for doc in nlp.pipe( list(zip(*pairs)) [1], n_process= n_jobs, batch_size=spacy_batch_size)]

        list_documents = [
            [x.text for x in doc.sents] # sentences in each document
            for doc in nlp.pipe( list(zip(*pairs)) [0], n_process= n_jobs, batch_size=spacy_batch_size)]
        
    
    elif tokenizer_name == 'nltk':
        from nltk.tokenize import sent_tokenize
        list_summaries = [sent_tokenize(_sum) for (_doc, _sum) in pairs]
        list_documents = [sent_tokenize(_doc) for (_doc, _sum) in pairs]

    def postprocessing(text_list):
        # Merge short sentences in to previous sentences
        texts = []
        for sents in text_list:
            new_list = []
            text = ""
            for sent in sents:
                text = text + sent + " "
                
                def isSent(test_text):
                    test_text = test_text.strip()
                    tokens = test_text.split()
                    count = 0
                    for token in tokens:
                        if any(c.isalpha() for c in token):
                            count += 1
                    return count >= 3

                if isSent(text):
                    new_list.append(addPeriod(text))
                    text = ""
            
            if text.strip() != "": new_list.append(addPeriod(text))
            texts.append(new_list)
        return texts
                                    

    new_pairs = list(zip(
                    postprocessing(list_documents), # list of list of str, segmented documents
                    postprocessing(list_summaries) # list of list of str, segmented summaries
                ))

    return new_pairs

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def greedy_selection(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    doc_length = len(doc_sent_list)
    sum_length = len(abstract_sent_list)

    sum_sents = [_rouge_clean(s).split() for s in abstract_sent_list]
    doc_sents = [_rouge_clean(s).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in doc_sents]
    reference_1grams = [_get_word_ngrams(1, [sent]) for sent in sum_sents]
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in doc_sents]
    reference_2grams = [_get_word_ngrams(2, [sent]) for sent in sum_sents]

    coverage = []
    for j in range(sum_length):
        max_rouge = 0.0
        selected = []
        done = False
        for _ in range(doc_length):
            if done: break
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(doc_length):
                if (i in selected):
                    continue
                c = selected + [i]
                candidates_1 = [evaluated_1grams[idx] for idx in c]
                candidates_1 = set.union(*map(set, candidates_1))
                candidates_2 = [evaluated_2grams[idx] for idx in c]
                candidates_2 = set.union(*map(set, candidates_2))
                rouge_1 = cal_rouge(candidates_1, reference_1grams[j])['r']
                rouge_2 = cal_rouge(candidates_2, reference_2grams[j])['r']
                rouge_score = rouge_1 + rouge_2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                done = True
                break
            selected.append(cur_id)
            max_rouge = cur_max_rouge
        coverage.append(selected)
    
    return coverage

def get_coverage(pairs):
    print("coverage processing...")

    for _doc, _sum in tqdm(pairs):
        sum_length = len(_sum)
        cov = greedy_selection(_doc, _sum)
        for i in range(sum_length):
            _sum[i] = {"text": _sum[i], "cov": cov[i], "ling": 0, "fact": 0}

def replace_special_character(s, L):
    """replaces all special characters in _L_ within _s_ by spaces
    """
    for l in L:
        s= s.replace(l, " ")
    return s

def normalize_sentence(s, special_chars):
    """Normalizing a sentence
    """
    # s = s.lower()
    s = replace_special_character(s, special_chars)
    s = s[:4000] # up to 500 characters per article or summary, useful when article is very long.

    return s 

def mutate_delete(words, ratio, sent_end):
    """delete _ratio_% of words in random locations of _words_, 
    while preserving sentence separators

    words: list of strings, e.g., ["I", "am", "lucky"]
    ratio: float, 0 to 1 
    sent_end: list of strings, the end of a sentence, e.g., [".", "?", "!"]

    example: 
        >>> mutate_delete("i am a happy guy now".split(" "), 0.2, string.punctuation)
            'i am a guy'
    """
    words =  copy.deepcopy(words)
    length = len(words)
    indices = random.sample(range(length), int( ratio * length))

    try : 
        res=  ' '.join([words[i] for i in range(length) \
                    if i not in indices or words[i][-1] in sent_end])
    except IndexError:
        print (words )

    return res 

def generate_one(dataset_name, split, features, neg_pos_ratio, load_start, load_end, special_chars, data_root, tokenizer_name, n_jobs, spacy_batch_size, batch_id): 
    """Generate one batch of data for one split (test or train) on one dataset, 
    given the start and end indexes of samples in the dataset
    """
    
    available_mutate = ["sent_replace", "sent_delete", "word_delete"]#, #"word_reorder"]

    # 1. Load data 
    dataset = tfds.load(name=dataset_name, download=False, 
                        split=split+ '[{}:{}]'.format(load_start, load_end)
                       )

    pairs = [(normalize_sentence(piece[features[0]].numpy().decode("utf-8"), special_chars), 
              normalize_sentence(piece[features[1]].numpy().decode("utf-8"), special_chars) )
                for piece in dataset]

    pairs = [(_doc, _sum) for (_doc, _sum) in pairs if len(_sum) > 0]

    # 2. Split summary sentences 
    pairs = split_pairs(pairs, tokenizer_name=tokenizer_name, spacy_batch_size=spacy_batch_size, n_jobs=n_jobs)

    # 3. Get Converage labels using ROUGE
    get_coverage(pairs)

    # 3. Mutate & cross pair the summaries
    num_samples = len(pairs)
    lines = []
    for sample_id, (_doc, _sum) in enumerate(pairs):
        line = {"doc": _doc, "sums": []}
        makeup_sums = []
        sum_mutate = copy.deepcopy(_sum)
        sum_flag = [False] * len(sum_mutate)
        
        # Track doc sent coverage infos
        coverage = numpy.zeros(len(_doc), dtype=int)
        for sum_sent in _sum: coverage[sum_sent["cov"]] += 1
    
        line["sums"].append({"sample": _sum, "coverage": (coverage > 0).tolist()})
        
        # Sentence-level mutation
        while len(sum_mutate) > 1 and not all(sum_flag):
            
            def select_sentence():
                rest_ids = [i for i in range(len(sum_flag)) if not sum_flag[i]]
                return random.choice(rest_ids)
            
            def replace_sentence():
                pair_id = sample_id # pick to new doc-sum pair
                while pair_id == sample_id:
                    pair_id = random.randint(0, len(pairs)-1)
                return random.choice(pairs[pair_id][1])
            
            # Select one sentence to mutate
            mutation = random.choice(available_mutate)
            sentence_id = select_sentence()
            
            if mutation == 'sent_replace':
                sum_mutate[sentence_id]["text"] = replace_sentence()["text"]
                sum_flag[sentence_id] = True
                # Change flag
                sum_mutate[sentence_id]["fact"] = 1
                coverage[sum_mutate[sentence_id]["cov"]] -= 1
                sum_mutate[sentence_id]["cov"] = []
            elif mutation == 'sent_delete':
                coverage[sum_mutate[sentence_id]["cov"]] -= 1
                del sum_mutate[sentence_id]
                del sum_flag[sentence_id]
            elif mutation == 'word_delete':
                words = sum_mutate[sentence_id]["text"].split()
                if len(words) == 0: continue
                # At least remove one word
                ratio = max(1.0/len(words), random.uniform(0, 1)) 
                sum_mutate[sentence_id]["text"] = addPeriod(mutate_delete(words, ratio, string.punctuation))
                # Change flag
                sum_mutate[sentence_id]["ling"] = 1
                sum_flag[sentence_id] = True
            elif mutation == 'word_reorder':
                words = sum_mutate[sentence_id]["text"].split()
                length = len(words) - 1 # Avoid the period(".") being reversed
                if length == 0: continue
                revwords = list(reversed(words))[1:]
                lr = random.choices(range(length), k=2)
                l, r = min(lr), max(lr)
                revl, revr = length-l, length-r
                words[l:r+1] = revwords[revr-1:revl]
                # Change flag
                sum_mutate[sentence_id]["text"] = " ".join(words)
                sum_mutate[sentence_id]["ling"] = 1
                sum_flag[sentence_id] = True
            else:
                print("???")
                assert(False)
            
            coverage_label = (coverage > 0).tolist()
            makeup_sums.append({"sample": copy.deepcopy(sum_mutate), "coverage": coverage_label})
        
        # Sample the mutated samples
        makeup_len = len(makeup_sums)
        sample_size = numpy.min([makeup_len, neg_pos_ratio])
        randIndex = random.sample(range(makeup_len), sample_size)
        randIndex.sort()
        makeup_sums = [makeup_sums[i] for i in randIndex]
        line["sums"].extend(makeup_sums)

        # import sys
        # print(json.dumps(line, indent=2))
        # sys.exit(-1)

        # Add a cross pairing sample in the end
        # cross_id = numpy.random.randint(num_samples)
        # while cross_id == sample_id:
        #    cross_id = numpy.random.randint(num_samples)
        # _, cross_sum = pairs[cross_id]
        # line.append(" ".join(cross_sum))
        lines.append(line)
        
    dumpfile = os.path.join(data_root, dataset_name.split(':')[0], "summar_ly", "{}_{}.jsonl".format(split, batch_id))
    if not os.path.exists(os.path.dirname(dumpfile)):
        try:
            os.makedirs(os.path.dirname(dumpfile))
        except OSError as exc: # Guard against rare conditions
            if exc.errno != errno.EEXIST:
                raise
    
    print ("Dumping into", dumpfile, end="...")
    with open(dumpfile, 'w') as f:
        for line in lines:
            f.write(json.dumps(line)+"\n")

    return lines


def sample_generation(conf):
    """main function to generate samples 
    """

    cfg = __import__(conf)

    for dataset_name in cfg.dataset_names:
        print ("From dataset:", dataset_name)
        features = cfg.dataset_features[dataset_name.split(':')[0]]

        for split in cfg.splits:
            print ("Data split:", split)
            total_samples = min(cfg.dataset_sizes_w_split[dataset_name.split(':')[0]][split], cfg.used_sample_num)

            boundaries = list(range(0, total_samples, cfg.my_batch_size))
            boundaries.append(total_samples)
            boundaries = list(zip(boundaries[:-1], boundaries[1:]))

            # for batch_id, (load_start, load_end) in enumerate(boundaries):
            for batch_id, (load_start, load_end) in enumerate(boundaries):
                # load_end = load_start + 2
                print ("\t batch {0}/{1}".format(batch_id+1, len(boundaries)), end="...")
                start_time = time.time()
                generate_one(dataset_name, split, features, cfg.neg_pos_ratio, load_start, load_end, cfg.special_characters_to_clean, cfg.data_root, cfg.tokenizer_name, cfg.n_jobs, cfg.spacy_batch_size, batch_id)

                elapse = time.time() - start_time
                print ("  Took {:.3f} seconds".format(elapse))
            
            combine_shuffle(["summar"], cfg.data_root, dataset_name, split, "ly")


if __name__ == "__main__":
    sample_generation("sentence_conf")