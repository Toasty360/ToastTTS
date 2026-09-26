"""Word emphasis: stress the words that carry the meaning.

  "I didn't say *he* stole it."   (someone else did)
  "I didn't *say* he stole it."   (I implied it)

A small TTS model can't know which word matters; the LLM writing the reply can.
So the LLM (or a person) marks words with *asterisks*, and this module performs
the emphasis the way people do: a pitch rise over the word, a little extra length,
and a little extra loudness. The voice itself is untouched (Praat PSOLA).

It needs to know where each word is in the audio: Piper reports per-phoneme
durations when the model has an alignment output. Add one to any Piper voice
(no retraining) with Piper's own tool:
  python -m piper.patch_voice_with_alignment models/<voice>.onnx --output models/aligned/<voice>.onnx
"""

import re

import numpy as np
import parselmouth
from parselmouth.praat import call

MARK = re.compile(r"\*([^*]+)\*")
# Strength presets. "subtle" was the first prototype; the listener heard only "a small
# difference, not sure", so "strong" follows how people mark contrastive focus: a bigger
# pitch accent, clearly longer, louder, AND post-focus compression (words after the
# focus get a flatter, lower melody), which is much of what makes contrast audible.
STRENGTHS = {
    #          pitch rise  lengthen  gain dB  post-focus: keep % of range, drop in semitones
    "subtle": (0.22,       1.18,     3.0,     1.0, 0.0),
    "strong": (0.42,       1.40,     4.5,     0.45, 1.5),   # ~+6 st accent
}
PITCH_RISE, LENGTHEN, GAIN_DB, _, _ = STRENGTHS["subtle"]


def parse(text):
    """'I didn't say *he* stole it.' -> ('I didn't say he stole it.', [3]) (word indices)."""
    plain, emphasized, index = [], [], 0
    for token in re.split(r"(\s+)", text):
        if not token or token.isspace():
            plain.append(token)
            continue
        if MARK.search(token):
            emphasized.append(index)
            token = MARK.sub(r"\1", token)
        plain.append(token)
        index += 1
    return "".join(plain), emphasized


def word_spans(alignments, sample_rate):
    """Sample ranges of each spoken word, from Piper's per-phoneme alignments.
    Words are separated by the space phoneme; '^' and '$' mark the sentence edges."""
    spans, position, start = [], 0, None
    for a in alignments:
        if a.phoneme in ("^", "$") or a.phoneme == " ":
            if start is not None:
                spans.append((start, position))
                start = None
        elif start is None and a.phoneme not in ".,;:!?":
            start = position
        position += a.num_samples
    if start is not None:
        spans.append((start, position))
    return spans


def emphasize(audio, sample_rate, spans, strength="subtle", sentence_end=None):
    """Apply pitch accent, lengthening and loudness to (start, end) sample spans, plus
    post-focus compression up to sentence_end (samples) for the "strong" preset."""
    if not spans:
        return audio
    rise, lengthen, gain_db, keep_range, drop_st = STRENGTHS[strength]
    audio = audio.astype(np.float64).copy()
    for start, end in spans:  # loudness first (a smooth bump over the word)
        n = end - start
        if n > 0:
            bump = np.sin(np.linspace(0, np.pi, n))
            audio[start:end] *= 10 ** (gain_db * bump / 20)
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch = call(manipulation, "Extract pitch tier")
    durations = call("Create DurationTier", "durations", 0, sound.duration)
    call(durations, "Add point", 0, 1.0)
    for start, end in spans:
        t0, t1 = start / sample_rate, end / sample_rate
        call(pitch, "Formula", f"if x >= {t0} and x <= {t1} then self * (1 + {rise} * "
                               f"sin(pi * (x - {t0}) / ({t1 - t0}))) else self fi")
        call(durations, "Add point", max(t0 - 0.002, 0), 1.0)
        call(durations, "Add point", t0, lengthen)
        call(durations, "Add point", t1, lengthen)
        call(durations, "Add point", t1 + 0.002, 1.0)
    if keep_range < 1.0 or drop_st:
        # post-focus compression: after the last focus word, squeeze the melody toward its
        # median (keep_range of its excursions) and lower it by drop_st semitones
        t_after = max(e for _, e in spans) / sample_rate
        t_end = (sentence_end or len(audio)) / sample_rate
        values = [call(pitch, "Get value at time", t) for t in np.arange(t_after, t_end, 0.01)]
        values = [v for v in values if v == v and v > 0]
        if values:
            ref = float(np.median(values))
            factor = 2 ** (-drop_st / 12)
            call(pitch, "Formula", f"if x > {t_after} and x <= {t_end} then "
                                   f"{ref} * (self / {ref}) ^ {keep_range} * {factor} else self fi")
    call([pitch, manipulation], "Replace pitch tier")
    call([manipulation, durations], "Replace duration tier")
    result = call(manipulation, "Get resynthesis (overlap-add)")
    return result.values[0].astype(np.float32)


def speak_with_emphasis(piper_voice, text, length_scale=1.0, strength="subtle"):
    """Synthesize text with *marked* words emphasized. piper_voice is a piper.PiperVoice
    whose model has an alignment output. Returns (audio, sample_rate, info)."""
    from piper import SynthesisConfig

    plain, marked = parse(text)
    sample_rate = piper_voice.config.sample_rate
    pieces, all_spans, offset, word_offset = [], [], 0, 0
    for chunk in piper_voice.synthesize(plain, SynthesisConfig(length_scale=length_scale), include_alignments=True):
        audio = chunk.audio_float_array
        spans = word_spans(chunk.phoneme_alignments or [], sample_rate)
        for index in marked:
            local = index - word_offset
            if 0 <= local < len(spans):
                all_spans.append((spans[local][0] + offset, spans[local][1] + offset))
        word_offset += len(spans)
        offset += len(audio)
        pieces.append(audio)
    audio = np.concatenate(pieces)
    words = len(plain.split())
    info = {"words_in_text": words, "words_found": word_offset, "emphasized": len(all_spans),
            "aligned": words == word_offset}
    if words != word_offset:  # espeak split words differently (e.g. numbers); don't guess
        return audio, sample_rate, {**info, "note": "word count mismatch: emphasis skipped"}
    return emphasize(audio, sample_rate, all_spans, strength), sample_rate, info
