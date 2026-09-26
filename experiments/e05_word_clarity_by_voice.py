"""E05: Is "semicolon" swallowed by our pipeline or by the voice itself?

Listening note: "it never said semicolon". Method: synthesize short
sentences with the plain model (no pipeline: no splitting, trimming or
fades) and let Whisper transcribe them. 2 takes each (output is random).
Also tries respellings, which change the pronunciation Piper is given.

Finding: both libritts voices (libritts_r-medium 3922 and libritts-high
p3922) say "semi"/"sema"/"semis" even mid-sentence and at their default
settings; no respelling fixes it. lessac and amy say it correctly. So it
is the voice model (trained on 904 speakers, little data each), not us.

  .venv\\Scripts\\python experiments\\e05_word_clarity_by_voice.py
"""

from toast.voices import load_voice
from toast.word_check import transcribe, word_errors

SENTENCES = [
    "A semicolon is useful here.",
    "Use a semicolon, then stop.",
    "It is about cadence.",
    "Does it handle subtle micro-breaks, like after a semicolon?",
]
VOICES = {
    "libritts_r-medium 3922 (file defaults)": "3922",
    "libritts_r-medium 3922 @0.5/0.8": "3922@0.5/0.8",
    "libritts-high p3922 @0.5/0.8": "piper:en_US-libritts-high:p3922@0.5/0.8",
    "lessac-medium": "lessac",
    "amy-medium": "amy",
}
RESPELLINGS = ["semicolon", "semi-colon", "semi colon", "semmy-colon"]


def main():
    for label, name in VOICES.items():
        voice = load_voice(name)
        print(f"== {label}")
        for sentence in SENTENCES:
            heard = [transcribe(voice.synthesize(sentence, speed=0.9), voice.sample_rate) for _ in range(2)]
            wrong = [word_errors(sentence, h)[0] for h in heard]
            print(f"  {sentence:<60} wrong {wrong}  heard: {heard}")

    voice = load_voice("3922@0.5/0.8")
    print("== respellings on libritts_r-medium 3922 @0.5/0.8 (3 takes)")
    for spelling in RESPELLINGS:
        phonemes = "".join(voice._voice.phonemize(f"a {spelling}")[0])
        heard = [transcribe(voice.synthesize(f"Use a {spelling} here.", speed=0.9), voice.sample_rate)
                 for _ in range(3)]
        print(f"  {spelling:<12} pronunciation {phonemes:<18} heard: {heard}")


if __name__ == "__main__":
    main()
