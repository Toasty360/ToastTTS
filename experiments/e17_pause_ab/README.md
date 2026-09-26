# e17_pause_ab — A/B: hand-set pause table vs learned pause model

Wires the e16 pause model into ToastEngine's pacing (`toast/pacing.py` takes
an optional `pause_ms` callable; `toast/learned_pauses.py` builds it from a
`PausePredictor`) and renders listening pairs: same voice, same piece cuts,
same seed — only the pause durations differ.

- **A (`*_a_table.wav`)**: current `PAUSES_MS` behavior — random pause in a
  hand-set range per punctuation (comma 180–260 ms, period 480–620 ms, …).
- **B (`*_b_learned.wav`)**: e16 model — 440 ms after commas, ~720 ms after
  sentence-final stops, no pause where it predicts none.

Sample texts were picked to stress the differences:

| file | why |
|---|---|
| `01_list` | the model's home turf: list commas + period |
| `02_opener` | opener rule (60–140 ms) vs the model's flat 440 ms comma |
| `03_flow` | subordinate-clause commas, no list |
| `04_semicolon` | honest gap: the model never saw `;`, predicts no pause; the table gives 300–380 ms |

## Re-render

```bash
~/workspace/.pause-venv/bin/python ab.py   # needs piper-tts + models/en_US-amy-medium.onnx
```

## Verdict

To be decided by listening, not metrics.
