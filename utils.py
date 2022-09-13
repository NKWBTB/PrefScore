from pre import sentence_conf as CFG
from pre import summarly_generation_rouge as P

class SentenceTokenizer:
    def __init__(self, do_postprocessing=True):
        self.do_postprocessing = do_postprocessing
        self.tokenizer_name = CFG.tokenizer_name
        self.n_jobs = CFG.n_jobs
        self.spacy_batch_size = CFG.spacy_batch_size
        if self.tokenizer_name == 'spacy':
            import spacy
            self.nlp=spacy.load("en_core_web_sm", exclude=["tok2vec",'tagger','parser','ner', 'attribute_ruler', 'lemmatizer'])
            self.nlp.add_pipe("sentencizer")
            self.nlp.max_length = 2000000 # default is 1,000,000
        elif self.tokenizer_name == 'nltk':
            from nltk.tokenize import sent_tokenize

    def __call__(self, texts):
        if self.tokenizer_name == 'spacy':
            list_sents = [
                [x.text for x in doc.sents] # sentences in each summary
                for doc in self.nlp.pipe( texts, n_process= self.n_jobs, batch_size=self.spacy_batch_size)]
        elif self.tokenizer_name == 'nltk':
            list_sents = [sent_tokenize(text) for text in texts]
        if self.do_postprocessing:
            list_sents = P.postprocessing(list_sents)
        return list_sents

sent_tokenizer = SentenceTokenizer()