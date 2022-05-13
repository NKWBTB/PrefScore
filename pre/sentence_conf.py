# configuration file for senentece-level negative sampling 

import string

# ==== constants about datasets 
dataset_features = {"cnn_dailymail": ['article', 'highlights'],
    "big_patent": ['description', 'abstract'],
    "newsroom": ['text', 'summary'],
    "scientific_papers": ['article', 'abstract'],
    "billsum":['text','summary'],
    "dryrun":[]
    }

dataset_sizes = {"billsum":23455, "scientific_papers":215913, "newsroom":1212740, 
                 "cnn_dailymail":311971, "big_patent":1341362}

dataset_sizes_w_split = {# new for sentence-level mutation
    "billsum":{'train':18949, 'test':3269},   
    "cnn_dailymail":{'train':287113, 'test':11490},
    "big_patent":{'train':1207222, 'test':67072},
    "scientific_papers":{'train':203037, 'test':6440},
    "newsroom": {'train': 995041, 'test': 108862}
}

#======== data loading parameters 

# Must match their names in TFDS 
# dataset_name = "dryrun" 
dataset_names = ["newsroom", "big_patent:2.0.0", "billsum", "scientific_papers"] # "cnn_dailymail", ] 

splits = ['train', 'test'] # We only need the train split. We skip validation and test.
# note that billsum has no validation set

#========= data output/dumping parameters 

data_root = "../exp/data"  # new for sentence-level mutation

n_jobs = 35

# compact or plain 
# plain is 3-column, doc, summary, target
# but plain may contain repeated docs, 
# which will cause extra time in sentence embedding (not applicable for BERT) 
# compact: small. easy for inspecting dump. Format per line: 
# doc, sum1, label1, sum2, label2, sum3, label3, ...

dump_format = "compact"

my_batch_size = 2**8*64
# how many doc-sum pairs to process each time
# When using stanza, too large or too small reduces GPU utility rate. 2**8 is a good number.
# The speed is about 10 seconds per 2**8 doc-sum pairs on 3090
# Doesn't matter when using Spacy. Set it to 2**8*64 on CPU. Adjust based on your RAM.

#========= NLP parameters

special_characters_to_clean = ['\n', '\t'] # replace such strings in raw data 

sent_end = [".", "!", "?"]  # symbols that represent the end of a sentence 
sent_end = string.punctuation

tokenizer_name = 'spacy' # or "nltk" 
spacy_batch_size = 8000 # doesn't seem to have much effect though

#========= negative sampling parameters 

# ratio between negative and positive samples
# minimal: 1 
neg_pos_ratio = 5
