# Cloud reference recordings

Recordings of [`samples/reference.txt`](../samples/reference.txt) made by the author with commercial TTS providers, used to measure and listen against ([notebook/07](../research/notebook/07-cloud-reference-and-pacing.md)).

| File | Provider / voice | Notes |
|---|---|---|
| `deepgram/deepgram-aura-2-thalia-en.wav` | Deepgram Aura-2, Thalia | 24 kHz mono |
| `deepgram/deepgram-flux-hannah-en-1.1x-speed.wav` | Deepgram Flux, Hannah | generated at 1.1× speed |
| `soniox/soniox-tts-grace.wav` | Soniox TTS, Grace | 24 kHz mono; the benchmark's "best-sounding" reference |

**Don't use these for training, distillation, or as tuning targets.** Soniox's [terms of service](https://soniox.com/policies/terms-of-service) forbid using its services, outputs *or derived data* to train, fine-tune, improve, distill or develop any speech synthesis model. They also restrict using evaluation or benchmarking results for that purpose. Deepgram's terms should be assumed to be similar until checked.

These recordings were used earlier as measurement references for pacing and prosody (E06, E09, E11, E12). That use may fall under the "derived data … to improve" wording; it has been flagged and stopped ([decision D35](../research/decisions.md)). Future references use openly licensed human speech instead. Listening to them yourself for comparison is ordinary use.
