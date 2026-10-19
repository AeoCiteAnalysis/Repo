"""Small wrapper around the OpenAI API with a disk cache.

Needs OPENAI_API_KEY (and optionally OPENAI_MODEL) in the project's .env file.
Every answer is stored in data/tables/llm_cache.jsonl keyed by (model, instructions, text), so
re-running a script never pays twice for the same question and results are reproducible.
"""
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from common import TABLES

CACHE = TABLES / 'llm_cache.jsonl'
_lock = threading.Lock()
_cache = None


def model_name():
    return os.environ.get('OPENAI_MODEL', 'gpt-5.4-mini')


def available():
    return bool(os.environ.get('OPENAI_API_KEY'))


def _key(model, instructions, text):
    return hashlib.sha1(f'{model}\x00{instructions}\x00{text}'.encode('utf-8')).hexdigest()


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            with open(CACHE, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    _cache[r['key']] = r['answer']
    return _cache


def ask(instructions, text, model=None):
    """Return the model's text answer (cached)."""
    from openai import OpenAI
    model = model or model_name()
    k = _key(model, instructions, text)
    cache = _load()
    if k in cache:
        return cache[k]
    client = OpenAI()
    resp = client.responses.create(model=model, instructions=instructions, input=text)
    answer = resp.output_text.strip()
    with _lock:
        cache[k] = answer
        with open(CACHE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(key=k, model=model, text=text[:200], answer=answer),
                               ensure_ascii=False) + '\n')
    return answer


def ask_many(instructions, texts, workers=8, model=None):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: ask(instructions, t, model), texts))


def check():
    """One cheap call to confirm the key and model work."""
    if not available():
        return 'OPENAI_API_KEY is not set (add it to .env)'
    try:
        return f'ok: {model_name()} answered "{ask("Reply with the single word: ok", "ping")}"'
    except Exception as e:
        return f'failed: {type(e).__name__}: {e}'


if __name__ == '__main__':
    print(check())
