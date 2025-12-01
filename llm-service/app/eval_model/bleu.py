from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score for translation evaluation

    Args:
        reference: List of reference sentences (tokenized)
        candidate: Generated sentence (tokenized)

    Returns:
        BLEU score between 0 and 1
    """
    # Apply smoothing to handle zero n-gram matches
    smoothing = SmoothingFunction().method4

    # Calculate BLEU with 1-4 gram precision
    bleu_score = sentence_bleu(
        [reference],
        candidate,
        smoothing_function=smoothing
    )

    return bleu_score
