# Ordered negative sampling for Siamese model
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import random
import os, errno
import copy
import random
import string
import numpy
import tensorflow_datasets as tfds

import time

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
        dump_root = os.path.join(data_root, dataset_name, method + "_" +  mode)
        print ("Combining and shuffling at ", dump_root)
        chops = os.path.join(dump_root, F"{split}_*.tsv")
        concrete = os.path.join(dump_root, F"{split}.tsv")
        tmp = os.path.join(dump_root, "tmp.tsv")

        os.system(F"cat {chops} > {concrete}")
        os.system(F"rm {chops}")
        os.system(F"shuf {concrete} > {tmp}")
        os.system(F"mv {tmp} {concrete}")

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
    
    elif tokenizer_name == 'nltk':
        from nltk.tokenize import sent_tokenize
        list_summaries = [sent_tokenize(_sum) for (_doc, _sum) in pairs]

    new_pairs = list(zip(
                    list(zip(*pairs))[0], # docs 
                    list_summaries # list of list of str, segmented summaries
                ))
    return new_pairs

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
    
    available_mutate = ["sent_replace", "sent_delete", "word_delete"]

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

    # 3. Mutate & cross pair the summaries
    num_samples = len(pairs)
    lines = []
    for sample_id, (_doc, _sum) in enumerate(pairs):
        line = [_doc, " ".join(_sum)]
        makeup_sums = []
        sum_mutate = _sum.copy()
        sum_flag = [False] * len(sum_mutate)
        
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
                sum_mutate[sentence_id] = replace_sentence()
                sum_flag[sentence_id] = True
            elif mutation == 'sent_delete':
                del sum_mutate[sentence_id]
                del sum_flag[sentence_id]
            elif mutation == 'word_delete':
                words = sum_mutate[sentence_id].split()
                if len(words) == 0: continue
                # At least remove one word
                ratio = max(1.0/len(words), random.uniform(0, 1)) 
                sum_mutate[sentence_id] = mutate_delete(words, ratio, string.punctuation)
                sum_flag[sentence_id] = True
            else:
                print("???")
                assert(False)
                
            makeup_sums.append(" ".join(sum_mutate))
        
        # Sample the mutated samples
        makeup_len = len(makeup_sums)
        sample_size = numpy.min([makeup_len, neg_pos_ratio])
        randIndex = random.sample(range(makeup_len), sample_size)
        randIndex.sort()
        makeup_sums = [makeup_sums[i] for i in randIndex]
        line.extend(makeup_sums)

        # Add a cross pairing sample in the end
        # cross_id = numpy.random.randint(num_samples)
        # while cross_id == sample_id:
        #    cross_id = numpy.random.randint(num_samples)
        # _, cross_sum = pairs[cross_id]
        # line.append(" ".join(cross_sum))
        lines.append(line)

    dumpfile = os.path.join(data_root, dataset_name, "pref_ordered", "{}_{}.tsv".format(split, batch_id))
    if not os.path.exists(os.path.dirname(dumpfile)):
        try:
            os.makedirs(os.path.dirname(dumpfile))
        except OSError as exc: # Guard against rare conditions
            if exc.errno != errno.EEXIST:
                raise
    
    print ("Dumping into", dumpfile, end="...")
    with open(dumpfile, 'w') as f:
        for line in lines:
            line = map(str, line)
            f.write("\t".join(line)+"\n")

    return lines


def sample_generation(conf):
    """main function to generate samples 
    """

    cfg = __import__(conf)

    for dataset_name in cfg.dataset_names:
        print ("From dataset:", dataset_name)
        features = cfg.dataset_features[dataset_name]

        for split in cfg.splits:
            print ("Data split:", split)
            total_samples = cfg.dataset_sizes_w_split[dataset_name][split]

            boundaries = list(range(0, total_samples, cfg.my_batch_size))
            boundaries.append(total_samples)
            boundaries = list(zip(boundaries[:-1], boundaries[1:]))

            for batch_id, (load_start, load_end) in enumerate(boundaries):
                print ("\t batch {0}/{1}".format(batch_id+1, len(boundaries)), end="...")
                start_time = time.time()
                generate_one(dataset_name, split, features, cfg.neg_pos_ratio, load_start, load_end, cfg.special_characters_to_clean, cfg.data_root, cfg.tokenizer_name, cfg.n_jobs, cfg.spacy_batch_size, batch_id)

                elapse = time.time() - start_time
                print ("  Took {:.3f} seconds".format(elapse))
            
            combine_shuffle(["pref"], cfg.data_root, dataset_name, split, "ordered")


if __name__ == "__main__":
    sample_generation("sentence_conf")