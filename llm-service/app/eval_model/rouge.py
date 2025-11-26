from rouge_score import rouge_scorer


def calculate_rouge_scores(reference, candidate):
    """
    Calculate ROUGE scores for summarization evaluation

    Args:
        reference: Reference summary text
        candidate: Generated summary text

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2'],
        use_stemmer=True
    )

    # Calculate scores
    scores = scorer.score(reference, candidate)

    # Extract F1 scores for each metric
    results = {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
    }

    return results
