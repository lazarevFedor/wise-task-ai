from rouge import calculate_rouge_scores
from bleu_score import calculate_bleu_score
from definitions_dataset import def_dataset
from itertools import product
from generate_answer import generate_answer
from logging import debug, basicConfig, DEBUG


basicConfig(level=DEBUG)


def evaluate_model_responses(data,
                             temperature,
                             top_p,
                             top_k,
                             repeat_penalty):
    """
    Evaluate model responses for given generation parameters
    and compute evaluation metrics.

    This function generates answers for each question in the dataset using specified
    generation parameters, then computes ROUGE and BLEU scores against reference answers.

    Args:
        data (list): Dataset containing questions, contexts, and reference answers
        temperature (float): Sampling temperature for text generation (0.0-1.0)
        top_p (float): Nucleus sampling parameter (0.0-1.0)
        top_k (int): Top-k sampling parameter
        repeat_penalty (float): Penalty for repeated tokens (1.0 = no penalty)

    Returns:
        dict: Evaluation results containing:
            - generation parameters (temperature, top_p, top_k, repeat_penalty)
            - average ROUGE-1, ROUGE-2, and BLEU scores
            - weighted composite score (25% ROUGE-1 + 25% ROUGE-2 + 50% BLEU)
            - list of generated answers
    """
    generated_answers = []
    references = []

    debug(f'Generate answers with params: '
          f'temp={temperature}, '
          f'top_p={top_p}, '
          f'top_k={top_k}, '
          f'repeat_penalty={repeat_penalty}')

    for i, item in enumerate(data):
        generated_answer = generate_answer(
            item['question'],
            item['context'],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty
        )
        generated_answers.append(generated_answer)
        references.append(item['answer'])

    rouge1_scores = []
    rouge2_scores = []

    for ref, cand in zip(references, generated_answers):
        rouge_scores = calculate_rouge_scores(ref, cand)
        rouge1_scores.append(rouge_scores['rouge1_f1'])
        rouge2_scores.append(rouge_scores['rouge2_f1'])

    bleu_scores = []

    for ref, cand in zip(references, generated_answers):
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        bleu_score = calculate_bleu_score(ref_tokens, cand_tokens)
        bleu_scores.append(bleu_score)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    weighted_score = 0.25 * avg_rouge1 + 0.25 * avg_rouge2 + 0.5 * avg_bleu

    return {
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'repeat_penalty': repeat_penalty,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'bleu': avg_bleu,
        'answers': generated_answers,
        'weighted_score': weighted_score
    }


def grid_search(data):
    """
    Perform grid search over generation parameters to optimize model performance.

    Evaluates all combinations of specified generation parameters and computes
    evaluation metrics for each combination. Returns results sorted by weighted
    composite score (25% ROUGE-1 + 25% ROUGE-2 + 50% BLEU).

    Args:
        data (list): Dataset containing questions, contexts, and reference answers

    Returns:
        list: Sorted list of evaluation results for all parameter combinations,
              descending by weighted score. Each result contains:
              - generation parameters
              - evaluation metrics (ROUGE-1, ROUGE-2, BLEU, weighted_score)
              - generated answers
    """

    param_grid = {
        'temperature': [0.1, 0.3, 0.5],
        'top_p': [0.3, 0.6, 0.9],
        'top_k': [20, 40],
        'repeat_penalty': [1, 1.05, 1.1, 1.2]
    }

    param_combinations = list(product(
        param_grid['temperature'],
        param_grid['top_p'],
        param_grid['top_k'],
        param_grid['repeat_penalty']
    ))

    results = []

    for i, (temp, top_p, top_k, repeat_penalty) in enumerate(param_combinations):
        debug(f'\nProgress: {i + 1}/{len(param_combinations)}')
        result = evaluate_model_responses(
            data,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty
        )

        results.append(result)

        debug(f'Current params: temp={temp}, '
              f'top_p={top_p}, top_k={top_k}, '
              f'repeat_penalty={repeat_penalty}')
        debug(f'Metrics: ROUGE-1={result['rouge1']:.4f}, '
              f'ROUGE-2={result['rouge2']:.4f}, '
              f'BLEU={result['bleu']:.4f}')

    best_rouge1 = max(results, key=lambda x: x['rouge1'])
    best_rouge2 = max(results, key=lambda x: x['rouge2'])
    best_bleu = max(results, key=lambda x: x['bleu'])
    best_weighted = max(results, key=lambda x: x['weighted_score'])

    debug(f'Best ROUGE-1: {best_rouge1['rouge1']:.4f}\n'
          f'temp={best_rouge1['temperature']}, '
          f'top_p={best_rouge1['top_p']}, '
          f'top_k={best_rouge1['top_k']}, '
          f'repeat_penalty={best_rouge1['repeat_penalty']}')

    debug(f'Best ROUGE-2: {best_rouge2['rouge2']:.4f}\n'
          f'temp={best_rouge2['temperature']}, '
          f'top_p={best_rouge2['top_p']}, '
          f'top_k={best_rouge2['top_k']}, '
          f'repeat_penalty={best_rouge2['repeat_penalty']}')

    debug(f'Best BLEU: {best_bleu['bleu']:.4f}\n'
          f'temp={best_bleu['temperature']}, '
          f'top_p={best_bleu['top_p']}, '
          f'top_k={best_bleu['top_k']}, '
          f'repeat_penalty={best_bleu['repeat_penalty']}')

    debug(f'Best WEIGHTED: {best_weighted['weighted_score']:.4f}\n'
          f'temp={best_weighted['temperature']}, '
          f'top_p={best_weighted['top_p']}, '
          f'top_k={best_weighted['top_k']}, '
          f'repeat_penalty={best_weighted['repeat_penalty']}')


if __name__ == '__main__':
    grid_search(def_dataset)
