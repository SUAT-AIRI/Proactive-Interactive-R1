# sentence_bleu_metric.py
from typing import Any, Dict, List, Optional
from nltk.translate.bleu_score import sentence_bleu


class BLEUMetric():
    """
    Corpus-free sentence-level BLEU:
        reference = `groundtruth`
        hypothesis = `completion`
    """

    def score(
        self,
        groundtruth: str,
        completion:str,
    ) -> float:
        if completion is None:
            raise ValueError("`completion` must not be None for BLEU scoring.")

        bleu = sentence_bleu([groundtruth], completion)
        return float(bleu)

