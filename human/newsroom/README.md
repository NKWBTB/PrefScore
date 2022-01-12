# Alignment with human evaluation in [Newsroom dataset](https://github.com/lil-lab/newsroom)

1. Download the human evaluation results from Newsroom

    ```bash
    wget https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv?raw=true -c -O newsroom-human-eval.csv
    ```
2. Run ``ref_free_baselines.py`` to produce summary quality scores in baselines.
3. Run ``baselines.py`` to evaluate reference-based upperbounds, request the [dataset access](https://lil.nlp.cornell.edu/newsroom/download/index.html) and put **test-stats.jsonl** in the folder.
4. Run ``test_eval.py`` to compute the correlation between human evaluation scores and those from our model and baselines. Scores from our model are by default under `exp/result*/`. Scores from baselines are produced in Step 2, 3 above (**predictions/metric_*.tsv**). 
