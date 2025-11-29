from bert_score import score


def calculate_bertscore(references, candidates, model_type='distilbert-base-uncased'):
    """
    Calculate BERTScore for semantic similarity evaluation

    Args:
        references: List of reference texts
        candidates: List of generated texts
        model_type: BERT model for embedding computation

    Returns:
        Precision, Recall, and F1 BERTScores
    """

    P, R, F1 = score(
        candidates,
        references,
        model_type=model_type,
        verbose=False
    )

    results = {
        'precision': P.tolist(),
        'recall': R.tolist(),
        'f1': F1.tolist()
    }

    return results
