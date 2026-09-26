"""Check that every word was actually spoken, by transcribing the audio.

Whisper (a speech-to-text model) listens to what we made; we compare its
transcript with the text we asked for and count words that came out wrong,
missing or extra. 0 is perfect.

Whisper's *writing style* isn't a speaking mistake, so before comparing we
make both texts look alike:
  "3:15 PM" / "3.15 p.m."          -> 3 15 pm
  "108" / "one hundred and eight"  -> one hundred eight
  "harbour" / "harbor"             -> harbor
  "micro-breaks" / "microbreaks"   -> same words
  "isn't" / "is not"               -> is not
This is how we caught the 3922 voice saying "semis" for "semicolon".
"""

import re

import numpy as np

from toast.text_normalize import _ONES, _TENS, number_words  # noqa: F401  (number_words is re-exported)

_model = None

BRITISH_SPELLINGS = {
    "harbour": "harbor", "neighbourhood": "neighborhood", "neighbour": "neighbor",
    "colour": "color", "favourite": "favorite", "favour": "favor", "behaviour": "behavior",
    "honour": "honor", "labour": "labor", "flavour": "flavor", "centre": "center",
    "theatre": "theater", "metre": "meter", "programme": "program", "organise": "organize",
    "realise": "realize", "recognise": "recognize", "analyse": "analyze", "apologise": "apologize",
}

CONTRACTIONS = {
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot",
    "won't": "will not", "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "it's": "it is", "that's": "that is",
    "there's": "there is", "what's": "what is", "let's": "let us", "i'm": "i am",
    "you're": "you are", "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "i'll": "i will", "you'll": "you will",
    "i'd": "i would", "you'd": "you would",
}


def _whisper():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel("small.en", device="cpu", compute_type="int8")
    return _model


def transcribe(audio, sample_rate):
    # Whisper expects 16 kHz audio.
    times = np.linspace(0, len(audio) / sample_rate, int(len(audio) * 16000 / sample_rate), endpoint=False)
    audio16 = np.interp(times, np.arange(len(audio)) / sample_rate, audio).astype(np.float32)
    segments, _ = _whisper().transcribe(audio16, language="en", beam_size=5, condition_on_previous_text=False)
    return " ".join(segment.text.strip() for segment in segments)


def words(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b[a-z]+'[a-z]+\b", lambda m: CONTRACTIONS.get(m.group(), m.group()), text)
    text = re.sub(r"\b([ap])\.?\s?m\b\.?", r" \1m ", text)       # p.m. / pm / p m -> pm
    text = re.sub(r"(\d)[:.](\d\d)\b", r"\1 \2", text)            # 3:15 / 3.15 -> 3 15
    text = re.sub(r"(\d)(am|pm)\b", r"\1 \2", text)                # 15pm -> 15 pm
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)               # 1,000 -> 1000
    text = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", text)
    tokens = re.sub(r"[^a-z' ]", " ", text).split()
    tokens = [BRITISH_SPELLINGS.get(t, t) for t in tokens]
    # "one hundred and eight" -> "one hundred eight"
    return [t for i, t in enumerate(tokens)
            if not (t == "and" and 0 < i < len(tokens) - 1 and tokens[i - 1] in ("hundred", "thousand")
                    and (tokens[i + 1] in _ONES or tokens[i + 1] in _TENS))]


def _split_into_known(word, vocab, max_parts=3):
    """'ultralowlatency' -> ['ultra', 'low', 'latency'] if those are known words."""
    if word in vocab:
        return [word]
    if max_parts == 1:
        return None
    for cut in range(1, len(word)):
        if word[:cut] in vocab and (rest := _split_into_known(word[cut:], vocab, max_parts - 1)):
            return [word[:cut]] + rest
    return None


def _match_compounds(expected, heard):
    """Line up words that differ only in spacing: "microbreaks" vs
    "micro breaks", "semi colon" vs "semicolon"."""
    vocab = set(expected)
    result, i = [], 0
    while i < len(heard):
        word = heard[i]
        joined = next((size for size in (3, 2)
                       if word not in vocab and "".join(heard[i:i + size]) in vocab), None)
        if joined:  # several heard words make one expected word
            result.append("".join(heard[i:i + joined]))
            i += joined
            continue
        result += _split_into_known(word, vocab) or [word]
        i += 1
    return result


def word_errors(expected, heard):
    """Return (number of word mistakes, list of (expected, heard) differences)."""
    a = words(expected)
    b = _match_compounds(a, words(heard))
    # Classic edit distance over words, remembering the path to report mistakes.
    cost = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    cost[:, 0] = range(len(a) + 1)
    cost[0, :] = range(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1,
                             cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    mistakes, i, j = [], len(a), len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                mistakes.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            mistakes.append((a[i - 1], "(missing)"))
            i -= 1
        else:
            mistakes.append(("(extra)", b[j - 1]))
            j -= 1
    return int(cost[len(a), len(b)]), mistakes[::-1]
