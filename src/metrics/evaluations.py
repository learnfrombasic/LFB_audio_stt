import re
import string
from typing import Dict, List

import jiwer


def preprocess_text(text: str) -> str:
    """
    Preprocesses text for ASR evaluation.
    Converts to lowercase, removes punctuation, and normalizes whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Normalize whitespace (replace multiple spaces with a single space)
    text = " ".join(text.split())

    return text


def compute_wer(
    references: List[str], hypotheses: List[str], preprocess: bool = True
) -> float:
    """
    Computes the Word Error Rate (WER) across a batch of sentences.

    Args:
        references: List of ground truth transcripts.
        hypotheses: List of predicted transcripts from the ASR model.
        preprocess: Whether to lowercase and remove punctuation before evaluation.

    Returns:
        float: The Word Error Rate (lower is better, 0.0 is perfect).
    """
    if preprocess:
        references = [preprocess_text(ref) for ref in references]
        hypotheses = [preprocess_text(hyp) for hyp in hypotheses]

    return jiwer.wer(references, hypotheses)


def compute_cer(
    references: List[str], hypotheses: List[str], preprocess: bool = True
) -> float:
    """
    Computes the Character Error Rate (CER) across a batch of sentences.
    Useful for agglutinative languages or when sub-word accuracy is important.

    Args:
        references: List of ground truth transcripts.
        hypotheses: List of predicted transcripts from the ASR model.
        preprocess: Whether to lowercase and remove punctuation before evaluation.

    Returns:
        float: The Character Error Rate (lower is better, 0.0 is perfect).
    """
    if preprocess:
        references = [preprocess_text(ref) for ref in references]
        hypotheses = [preprocess_text(hyp) for hyp in hypotheses]

    return jiwer.cer(references, hypotheses)


def evaluate_asr(
    references: List[str], hypotheses: List[str], preprocess: bool = True
) -> Dict[str, float]:
    """
    Computes all standard ASR metrics at once.
    """
    if preprocess:
        references = [preprocess_text(ref) for ref in references]
        hypotheses = [preprocess_text(hyp) for hyp in hypotheses]

    # jiwer.process_words gives a detailed output including hits, substitutions, deletions, and insertions
    word_output = jiwer.process_words(references, hypotheses)

    return {
        "wer": word_output.wer,
        "cer": jiwer.cer(references, hypotheses),
        "mer": word_output.mer,  # Match Error Rate
        "wil": word_output.wil,  # Word Information Lost
        "substitutions": word_output.substitutions,
        "deletions": word_output.deletions,
        "insertions": word_output.insertions,
        "hits": word_output.hits,
    }
