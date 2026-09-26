"""Pretend to be an LLM streaming its reply.

Real LLMs send text in "tokens": word fragments of about 4 characters
(" buil", "ding", " an"). This sends a fixed text the same way, at a
steady rate, so tests are repeatable.
"""

import re
import time


def fake_llm_tokens(text, tokens_per_second=30.0, first_token_delay=0.0):
    tokens = re.findall(r"\s*\S{1,4}", text)
    start = time.perf_counter() + first_token_delay
    for i, token in enumerate(tokens):
        # Each token has a fixed arrival time. If we were busy (making audio)
        # when it arrived, it's handed over immediately, like a real stream
        # that queued up while we weren't looking.
        wait = start + i / tokens_per_second - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        yield token
