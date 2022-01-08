# PrefScore
PrefScore for summarization evaluation

## Requirements

- [Tensorflow](https://www.tensorflow.org/) and [Tensorflow datasets](https://www.tensorflow.org/datasets/)
- [Pytorch](https://pytorch.org/)
- [Hugginface transformers](https://huggingface.co/docs/transformers/index)
- [Spacy](https://spacy.io/)
- [Scipy](https://scipy.org/)
- [SummEval](https://github.com/Yale-LILY/SummEval) (Optional)
- [NLTK](https://www.nltk.org/) (Optional)

## Files
```
pre/            Codes for negative sampling
human/          Codes for human evaluation
config.py       Config file for folder and training settings
model.py        Script for training the models
evaluate.py     Evaluate the trained models on target datasets 
```

## Negative sampling (Preprocess)
Code for generating negative samples are in `pre` folder. 

```bash
cd pre
python3 ordered_generation.py  
```
Edit ``pre/sentence_conf.py`` to change negative sampling settings. 

## Training
Run ``python3 model.py -h`` for full command line arguments. 

Example (Training on the preprocessed **billsumm** dataset):
```
python3 model.py --dataset billsum
```

## Evaluating 
To evaluate the trained model on **newsroom**, **realsumm** or **tac2010**, go to ``human/`` folder for detailed instructions to get the processed files:
- ``human/newsroom/newsroom-human-eval.csv``
- ``human/realsumm/realsumm_100.tsv``
- ``human/tac/TAC2010_all.json``

Run ``python3 evaluate.py -h`` for full command line arguments. 

Example (Evaluate the model trained from **billsumm** on **newsroom**):
```
python3 evaluate.py --dataset billsum --target newsroom
```

## Alignment with human evaluations 

Code for computing the correlation between our models' predictions and human ratings from the three datasets is in the `human` folder. 

## Reproduction Hint
The pretrained model and results are release in [exp.zip](https://drive.google.com/file/d/1IFXiH7di9pBM74dLCexNAD_ycOhjfIyD/view?usp=sharing), download and extract in the repo folder for reproducing the results in the paper.

https://drive.google.com/file/d/1IFXiH7di9pBM74dLCexNAD_ycOhjfIyD/view?usp=sharing

## Misc
1. To evaluate on a custom dataset, format the dataset as a tsv file where each line starts with a document and followed by serveral summaries of the document separated by ``'\t'``. See ``example.tsv`` for example.

2. Example for use the metric in a script:
```python
import torch
import config as CFG
from model import Scorer
from evaluate import evaluate

scorer = Scorer()
# CKPT_PATH is the path of a pretrained pth model file
scorer.load_state_dict(torch.load(CKPT_PATH, map_location=CFG.DEVICE))
scorer.to(CFG.DEVICE)
scorer.eval() 

docs = ["This is a document.", "This is another document."]
sums = ["This is summary1", "This is summary2."]

results = evaluate(docs, sums, scorer)
```