# 03: Streaming: speaking while the LLM is still writing

*2026-09-25 · Code: `toast/stream_chunker.py`, `toast/pacing.py` (`stream_from_llm`), `toast/fake_llm.py`, `toast/player.py`, `scripts/live.py` · Tests: `tests/test_stream_chunker.py`*

## Question

Can speech of an LLM reply start almost immediately, with natural pauses, and never let the audio run dry?

## Setup

- **Fake LLM** (`toast/fake_llm.py`): sends `reference.txt` as ~4-character tokens (" buil", "ding") on a fixed schedule, by default 30 tokens/s, typical for a fast chat model. If the player is busy, tokens queue up just like a real stream.
- **Live chunker** (`toast/stream_chunker.py`): decides when enough text has arrived to speak the next piece.
- **Measurement:** TTFA is taken from the arrival of the **first token**, so it includes waiting for words. Playback is simulated to count stalls (see [methodology](../methodology.md)).

## Chunker rules

A cut happens only once **the start of the next word has been seen**. That one rule keeps these intact:

- "3:15" (no space after the colon)
- "1,000" (no space after the comma)
- "Dr. Smith", "e.g." (abbreviation list)

| Rule | Why |
|---|---|
| Cut at `. ? ! ; : ...` and dashes | natural pause points |
| Cut at a comma only after a short piece (≤ 4 words) | pauses for asides and list items, not inside long clauses |
| Cut before *because / but / although / though / unless / whereas* once the piece has 6+ words | a natural breathing point ("…leave by noon \| because traffic…") |
| **Flash:** the first piece may cut after 4 words if no pause point comes sooner, but never right after a number or initial ("at 3 \| PM") | fast start |
| **Running low:** if less than 0.4 s of audio is queued, accept any comma or a 4-word cut | a short hesitation sounds human; dead air doesn't |
| When hurrying, prefer a pause point within 6 words over the 4-word cut | "building a fast voice pipeline," beats "building a fast voice \| pipeline," |
| Cut long run-ons (24+ words) before a conjunction | safety ceiling |

## Results (3922 voice, then lessac, then amy)

| Scenario | TTFA from first token | Stalls |
|---|---|---|
| Reference paragraph, 30 tokens/s | **89 ms** | 0 |
| Reference paragraph, 10 tokens/s, *before* the running-low rule | 237 ms | **1 (900 ms of dead air)** |
| Reference paragraph, 10 tokens/s, *after* | 223 ms | 0 |
| Sentence with no early comma, 30 tokens/s | 250 ms | 0 |
| lessac-medium, 30 / 10 tokens/s | 93 / 225 ms | 0 / 0 (0 wrong words out of 95 in both) |
| amy-medium at 1.1×, 30 tokens/s | 93 ms | 0 |

**Where the 89 ms goes:** about 67 ms is waiting for the tokens "Well", "," and " to". The splitter needs the start of the next word to be sure the comma is a pause. The remaining ~20 ms is synthesis. So streaming TTFA is bounded by **LLM speed**, not by the TTS.

**The 10 tokens/s stall:** "building an ultra-low latency voice pipeline isn't just about raw speed;" takes about 1.6 s to arrive at 10 tokens/s, while only "Well, to be honest," (about 1 s) was playing. With the running-low rule, the chunker cut "building an ultra-low latency | voice pipeline…", which sounds like a thinking pause.

## A later fix: the first piece when the whole text arrives at once

When a TTS API receives the full text (as in the Coval benchmark, [06](06-coval-comparison.md)), every punctuation mark is visible immediately. The first piece therefore ran to the first comma, sometimes 10+ words, even with numbers read out in full. "There's a slight delay with your $347.89 order," took about 190 ms to synthesize.

After the fix (hurry → take the 4-word flash cut unless a pause point comes within 6 words), the first chunk arrived in 57–72 ms on those prompts. The leading-silence margin was also reduced from 25 ms to 10 ms, which cut the silence before the first audible sample from about 28 ms to about 15 ms.

## Laptop drift

The same live run measured 89–93 ms early in the session and 112–123 ms later. Synthesizing "Well," took about 20 ms early and 28 ms later on an unchanged chunker (checked: the cut happened at the same token). **Timings drift by 20–30 ms with machine state**, so comparisons are always made within one session.

→ Next: [04: voice intelligibility](04-voice-intelligibility.md)
