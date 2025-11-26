from requests import post
from logging import error, basicConfig, DEBUG


basicConfig(level=DEBUG)


def generate_answer(question, context, temperature, top_p, top_k, repeat_penalty):
    """TODO: add docstring"""
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
                'num_predict': 64,
                'seed': 42,
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0,
                'stop': ['\n\n', 'Ответ:']
            },
            timeout=90
        )
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '').strip()
        return result
    except Exception as e:
        error(f'Error generating answer: {str(e)}')
        return ''
