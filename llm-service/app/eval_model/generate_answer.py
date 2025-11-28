from requests import post
from logging import error, basicConfig, DEBUG


basicConfig(level=DEBUG)


def generate_answer(question, context, temperature, top_p, top_k, repeat_penalty, n_predict):
    """
    Generate an answer for a given question using context
    by querying a local language model API.

    This function constructs a prompt from the provided question
    and context, then sends a request
    to a locally hosted language model inference server to generate a contextual answer.

    Args:
        question (str): The question to be answered
        context (str): Contextual information used to inform the answer
        temperature (float): Sampling temperature for text generation (0.0-1.0)
        top_p (float): Nucleus sampling parameter (0.0-1.0)
        top_k (int): Top-k sampling parameter
        repeat_penalty (float): Penalty for repeated tokens (1.0 = no penalty)
        n_predict (int): Number of predictions to return

    Returns:
        str: The generated answer text, or empty string if generation fails

    Raises:
        Logs errors but does not raise exceptions - returns empty string on failure

    Note:
        The function expects a locally running inference server at http://localhost:11343.
        Generation is limited to 64 tokens and
        stops at 'Ответ:' marker.
    """
    prompt = (f'Контекст: {context} '
              f'Вопрос: {question} '
              f'Ответь на вопрос, используя контекст.')

    try:
        response = post(
            'http://localhost:11343/v1/completions',
            json={
                'prompt': prompt,
                'stream': False,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'repeat_penalty': repeat_penalty,
                'num_predict': n_predict,
                'seed': 42,
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0,
                'stop': ['Ответ:']
            },
            timeout=90
        )
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '').strip()
        return result
    except Exception as e:
        error(f'Error generating answer: {str(e)}')
        return ''
