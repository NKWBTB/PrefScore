import os

TAC_ROOT = "/home/gluo/Dataset/TAC2010"
ARTICLE_SET_PATH = os.path.join(TAC_ROOT, "TAC2010_Summarization_Documents", "GuidedSumm10_test_docs_files")
SUMMARY_SET_PATH = os.path.join(TAC_ROOT, "GuidedSumm2010_eval", "ROUGE")
SCORE_PATH = os.path.join(TAC_ROOT, "GuidedSumm2010_eval", "manual")
ROUGE_SCORE_PATH = os.path.join(SUMMARY_SET_PATH, "rouge_A.m.out")