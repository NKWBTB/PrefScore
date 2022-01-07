# Alignment with human evaluation in [RealSumm dataset](https://github.com/neulab/REALSumm)

1. Download the CNNDM dataset and human evaluation results from RealSumm
```
wget -O src.txt "https://drive.google.com/uc?export=download&id=1z1_i3cCQOd-1PWfaoFwO34YgCvdJemH7"

wget -O abs.pkl "https://github.com/neulab/REALSumm/blob/master/scores_dicts/abs.pkl?raw=true"

wget -O ext.pkl "https://github.com/neulab/REALSumm/blob/master/scores_dicts/ext.pkl?raw=true"
```

2. Run ``generate_test.py`` to generate test file (``realsumm_100.tsv``).

3. Run ``baselines.py`` to evaluate reference-based upperbounds.

4. Run ``ref_free_baselines.py`` to evaluate reference-free baselines.

5. Run ``test_eval.py`` to compute the correlation between human evaluation scores and those from our model and baselines. Scores from our model are by default under `exp/result*/`. Scores from baselines are produced in Step 3, 4 above. 
