# Experiments

Focused investigations, each answering one question behind a design decision. Every script is self-contained and re-runnable, and its most recent output is saved in [`results/`](results/) with the date and exact command.

| # | Question | Finding | Output | Notebook |
|---|---|---|---|---|
| [E01](e01_compare_piper_kitten.py) | Piper vs Kitten: speed, and does a model survive being cut into pieces? | Piper is 15–30× faster; Kitten "whole" sounds good, "pieces" choppy (fragment prosody) | audio in `out/e01/` | [01](../research/notebook/01-baseline.md) |
| [E02](e02_model_pause_gaps.py) | Do models already pause at punctuation (so those pauses could be stretched)? | No: 3922 leaves 30–50 ms gaps; Kitten/amy leave gaps between ordinary words | [txt](results/e02_model_pause_gaps.txt) | [02](../research/notebook/02-natural-pauses.md) |
| [E03](e03_randomness_breathiness.py) | Does high `noise_scale` cause the "gasping"? | Not shown by this measure (lively 354 vs 329–360 breathy frames); cause unresolved | [txt](results/e03_randomness_breathiness.txt) | [02](../research/notebook/02-natural-pauses.md) |
| [E04](e04_piece_end_punctuation.py) | Does ending a piece in ";" rush the last word? | No meaningful difference over 6 runs (2.94 vs 2.98 s); an earlier single-run result was noise | [txt](results/e04_piece_end_punctuation.txt) | [02](../research/notebook/02-natural-pauses.md) |
| [E05](e05_word_clarity_by_voice.py) | Is "semicolon" lost in our pipeline or in the voice? | In the voice: both libritts models fail even mid-sentence, respellings don't help; lessac/amy are 100% correct | [txt](results/e05_word_clarity_by_voice.txt) | [04](../research/notebook/04-voice-intelligibility.md) |
| [E06](e06_cloud_pacing.py) | How do cloud voices pace speech vs amy? | Pauses already match; amy needed to be faster (1.2× → 179 wpm vs 183–195) | [txt](results/e06_cloud_pacing.txt) | [07](../research/notebook/07-cloud-reference-and-pacing.md) |
| [E07](e07_coval_time_format.py) | How much Coval WER is just Whisper's time formatting? | About 3.5 points per voice; amy's real mistakes listed | [txt](results/e07_coval_time_format.txt) | [06](../research/notebook/06-coval-comparison.md) |
| [E08](e08_kokoro_vs_amy.py) | Does Kokoro close the gap to cloud voices, and can it stream? | Pacing matches Soniox at 1.0× (194 vs 195 wpm); UTMOS ties it with amy; 344 ms TTFA, 1 stall at 10 tokens/s | [txt](results/e08_kokoro_vs_amy.txt) | [10](../research/notebook/10-kokoro.md) |
| [E09](e09_phrase_final_softening.py) | Is the last word before a pause softer in natural voices? | Yes for Kokoro/Soniox (−0.3 to −1.0 dB); amy's is louder (+1.1 dB) | [txt](results/e09_phrase_final_softening.txt) | [11](../research/notebook/11-phrase-final-softening.md) |
| [E10](e10_softening_amy.py) | Which softening setting matches them? | The measure is too noisy (±1 dB) to decide; 500 ms / −6 dB default; chosen by ear | [txt](results/e10_softening_amy.txt) | [11](../research/notebook/11-phrase-final-softening.md) |
| [E11](e11_pitch_profile.py) | Is amy "robotic" because its pitch moves too little? | Yes: pitch range 6.8 st vs 10.5–13.0 for natural voices; widening it helps range, not liveliness | [txt](results/e11_pitch_profile.txt) | [12](../research/notebook/12-robotic-melody-and-well.md) |
| [E12](e12_well.py) | Why does amy's "Well," sound different? | Same phonemes; spoken alone it gets a falling, final contour (Whisper hears "Well."); pause too long | [txt](results/e12_well.txt) | [12](../research/notebook/12-robotic-melody-and-well.md) |
| [E13](e13_vc_feasibility.py) (Modal) + [eval](e13_evaluate.py) + [artifacts](e13_artifact_check.py) | Can voice conversion put Kokoro's delivery into amy's voice? | kNN-VC: yes (91% of Kokoro's core melody, sounds like amy, 0 wrong words, no glitches) but 16 kHz output; OpenVoice v2: no | [eval](results/e13_vc_feasibility.txt), [artifacts](results/e13_artifact_check.txt) | [13](../research/notebook/13-vc-feasibility.md) |
| E14 (E13 script, `--source dailytalk`) | Same test with human speech and three converters | Seed-VC f0 (44.1 kHz) is the only full pass: 93% of the melody, bright, sounds like amy; UTMOS lower, so a quality gate is added | [eval](results/e14_vc_dailytalk.txt) | [14](../research/notebook/14-human-source-and-converter.md) |
| [E15](e15_voice_stability.py) | Is the "shaking / breaking" measurable (jitter, shimmer, HNR)? | Converted audio is the roughest (HNR 11.9 vs human 13.7), confirming the ear; the student's "breaking" barely shows (jitter 1.76 vs human 1.46, HNR 13.5), so it's likely pitch breaks or intonation these measures miss | [txt](results/e15_voice_stability.txt) | [15](../research/notebook/15-training.md) |
| E16 ([e16_pause_model](e16_pause_model/)) | Can a small model learn where to pause? | BiLSTM (93.6k params, F1 0.934) but mostly learned punctuation→pause (99.2% of training pauses follow `,` or `.`); opt-in only, hand-written table stays default | checkpoint + ONNX in dir | — |
| E17 ([e17_pause_ab](e17_pause_ab/)) | Table pauses vs learned pauses, by ear | Same voice/speed/seed, only pause durations differ; listening pairs in `wav/` | [readme](e17_pause_ab/README.md) | — |
| E18 ([e18_pause_longform](e18_pause_longform/)) | Does long-form audiobook data teach phrase breaks? | No: only 1.4% of pauses were not after punctuation, F1 0.354, model unusable; e16 stays the better learned model | [readme](e18_pause_longform/README.md) | — |
| E19 ([e19_stage2_ab](e19_stage2_ab/)) | Which stage-2 checkpoint sounds better? | Leg-1 best vs leg-2 best, same sentences and settings, blind; verdict pending listening | [wav](e19_stage2_ab/wav/) | — |

Run any of them from the project root:

```
.venv\Scripts\python experiments\e05_word_clarity_by_voice.py
```

**Notes:**

- Piper's output is random on every call, so experiments repeat runs where it matters. E04 exists because a single-run result turned out to be noise.
- E05–E07 use Whisper and/or UTMOS and take 1–3 minutes. E07 only reads saved CSVs.
- Outputs in `results/` were produced on 2026-09-25 on the machine described in [methodology](../research/methodology.md).