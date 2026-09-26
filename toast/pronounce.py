"""Fix words the model says badly by respelling them.

Piper turns words into pronunciation symbols (IPA) before speaking. If a
voice keeps saying a word wrong, a respelling can nudge it. Check the
result with the word check (scripts/render.py --check-words, or
toast/word_check.py; see experiments/e05), because it's easy to fool
yourself: "semicolon" -> "semmy-colon" looked right on paper but the 3922
voice still said "semi". Some words are simply beyond a voice.

(Piper can also take exact IPA inside [[ ]], but the text around the
brackets is then read on its own: "a [[...]]" turns "a" into the letter
"AY". Respelling keeps every neighbouring word in context.)

See how Piper hears a word:
  .venv\\Scripts\\python -c "from toast.voices import load_voice; print(load_voice('3922')._voice.phonemize('a word'))"
"""

import re

RESPELLINGS = {
    # "word": "respelling",
}

_WORDS = re.compile(r"\b(" + "|".join(map(re.escape, RESPELLINGS)) + r")\b", re.IGNORECASE) if RESPELLINGS else None


def apply_respellings(text):
    if _WORDS is None:
        return text
    return _WORDS.sub(lambda m: RESPELLINGS[m.group().lower()], text)
