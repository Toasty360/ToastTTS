#!/usr/bin/env python3
"""Build the workshop paper PDF with fpdf2. Draft for validation."""
from fpdf import FPDF

OUT = "pause_control_draft.pdf"


class Paper(FPDF):
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font(SERIF, "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, str(self.page_no()), align="C")


pdf = Paper(format="A4")
pdf.set_margins(25, 22, 25)
pdf.set_auto_page_break(True, 22)
for style, fname in [("", "DejaVuSerif.ttf"), ("B", "DejaVuSerif-Bold.ttf")]:
    pdf.add_font("DSerif", style, f"/usr/share/fonts/truetype/dejavu/{fname}")
for style, fname in [("I", "DejaVuSerif-Italic.ttf"), ("BI", "DejaVuSerif-BoldItalic.ttf")]:
    pdf.add_font("DSerif", style, f"/usr/share/matplotlib/mpl-data/fonts/ttf/{fname}")
SERIF = "DSerif"
pdf.add_page()

# Title block
pdf.set_font(SERIF, "B", 20)
pdf.multi_cell(0, 9, "Controlling Pause Behavior\nWhen Fine-Tuning VITS-Based\nText-to-Speech", align="C")
pdf.ln(4)
pdf.set_font(SERIF, "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Toasty360", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Draft for validation \u2014 September 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(6)


def abstract(text):
    pdf.set_font(SERIF, "B", 10)
    pdf.cell(0, 6, "Abstract", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(SERIF, "", 9.5)
    pdf.multi_cell(0, 5.2, text, align="J")
    pdf.ln(4)


def section(title):
    pdf.ln(2)
    pdf.set_font(SERIF, "B", 12.5)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def subsection(title):
    pdf.ln(1)
    pdf.set_font(SERIF, "B", 11)
    pdf.cell(0, 6.5, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def body(text):
    pdf.set_font(SERIF, "", 10.5)
    pdf.multi_cell(0, 5.6, text, align="J")
    pdf.ln(2.5)


def table(headers, rows, widths, caption):
    pdf.ln(1)
    pdf.set_font(SERIF, "", 8.5)
    x0 = pdf.get_x()
    # header
    pdf.set_font(SERIF, "B", 8.5)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font(SERIF, "", 8.5)
    for row in rows:
        for val, w in zip(row, widths):
            pdf.cell(w, 6, val, border=1, align="C")
        pdf.ln()
    pdf.ln(1)
    pdf.set_font(SERIF, "I", 8.5)
    pdf.multi_cell(0, 4.8, caption, align="C")
    pdf.set_font(SERIF, "", 10.5)
    pdf.ln(3)


abstract(
    "Fine-tuning a VITS-based text-to-speech model on expressive read speech reproduces the "
    "speaker's pause habits, including over-long dramatic pauses. The stochastic duration predictor "
    "adds a second problem: pause tails vary by hundreds of milliseconds between identical renders. "
    "This paper describes three practical interventions developed while fine-tuning Piper on 1,546 "
    "clips of an expressive speaker (Expresso, speaker ex02). First, a measurement protocol: at "
    "noise_w=0.5 the median pause is stable but the tail is not, so every sentence is rendered three "
    "times and the median is reported. Second, pause-normalized fine-tuning: internal silences longer "
    "than 300 ms in the training audio are shortened to 200 ms before training; after 2 h 18 min of "
    "continued training, comma pauses fell from p50 330 ms to 140 ms (teacher: 180 ms) and the "
    "general-distribution tail collapsed from 960 ms to 280 ms. A control run \u2014 2.69 h of continued "
    "training on the unmodified data \u2014 left pauses near the stage-1 level, attributing the collapse "
    "to the normalization rather than to more training. Third, an opt-in synthesis-time pause "
    "clamp that hard-bounds intra-piece pauses without retraining. Voice quality was preserved: 11 vs "
    "12 word errors against the teacher, UTMOS 4.25 vs 4.35, and 0.74 similarity to the speaker's own "
    "recordings. Negative results are reported honestly: a learned pause model that only learned "
    "punctuation, and a long-form retrain that failed for lack of phrase-break signal."
)

section("1  Introduction")
body(
    "VITS models learn pause durations from their training data through a stochastic duration "
    "predictor. When the training speaker is a dramatic reader, the model reproduces dramatic pauses: "
    "comma pauses roughly twice as long as a neutral teacher voice, with tails approaching a full "
    "second. Separately, the predictor's randomness makes single-render pause measurements unreliable."
)
body(
    "This paper documents the interventions used to bring pauses under control while fine-tuning "
    "Piper, a VITS-based model, on Expresso speaker ex02: a measurement protocol, a data-side fix, and "
    "an inference-time guardrail. All three are small, cheap, and independent of the fine-tuning "
    "framework. Two abandoned approaches are reported as negative results."
)

section("2  Setup")
body(
    "The student is a Piper medium voice (about 15.8M parameters, 63.5 MB as ONNX), fine-tuned from "
    "en_US-amy-medium. The data is the Expresso dataset (CC BY-NC 4.0), speaker ex02, 1,546 "
    "quality-gated read clips. Training ran with PyTorch Lightning on a Modal L4 GPU. Evaluation uses "
    "40 targeted sentences (20 comma / 10 semicolon / 10 colon) plus 10 general sentences; pauses are "
    "measured as internal silences via energy-based detection; word errors via Whisper small.en; "
    "naturalness via UTMOS; voice similarity via ECAPA-TDNN speaker embeddings. Unless noted, "
    "inference uses noise_scale=0.3, noise_w=0.5."
)

section("3  The stochasticity problem")
body(
    "At the default noise_w=0.8, rendering the same sentence twice produced different pause "
    "placements: phantom pauses appearing and disappearing between renders. Reducing the "
    "duration-noise knob to noise_w=0.5 (with noise_scale=0.3) removed the phantom pauses with no "
    "retraining, an inference-only change. But the predictor stays stochastic: two identical runs of "
    "the pause evaluation gave different tails for the same voice (Table 1)."
)
table(
    ["run", "comma p50", "comma p95", "comma max", "general max"],
    [["1", "270 ms", "420 ms", "420 ms", "500 ms"],
     ["2", "280 ms", "529 ms", "700 ms", "980 ms"]],
    [22, 30, 30, 30, 30],
    "Table 1: Two identical pause evaluations of the stage-1 voice. The median is stable; the tail swings by hundreds of milliseconds.",
)
body(
    "The median is stable (270 to 280 ms) but the tail swings by hundreds of milliseconds run to run. "
    "At noise_w=0.5, a single render's longest pause is therefore not a reliable measurement. The "
    "protocol adopted for everything below: render each sentence three times and take the median for "
    "targeted sentences (pooling for the general distribution), which removes the tail noise."
)

section("4  Pause-normalized fine-tuning")
body(
    "Internal silences longer than 300 ms in the 1,546 gated clips were shortened to 200 ms with an "
    "audio-domain edit; the text was left unchanged. The distribution shift is shown in Table 2: the "
    "median pause is untouched (60 ms), while the tail is cut down (p95 420 to 207 ms, max 1,820 to 300 ms)."
)
table(
    ["split", "n pauses", "p50", "p95", "max"],
    [["before", "6,521", "60 ms", "420 ms", "1,820 ms"],
     ["after", "6,534", "60 ms", "207 ms", "300 ms"]],
    [28, 28, 28, 28, 28],
    "Table 2: Pause distribution in the training clips before and after normalization.",
)
body(
    "Stage 2 fine-tunes the stage-1 checkpoint on this normalized set (2 h 18 min of new GPU time, in "
    "two legs; comma pauses had already visibly collapsed after 0.65 h). Results are in Table 3 and "
    "Figure 1. Stage 1 had learned comma pauses roughly twice the teacher's (p50 330 vs 180 ms; p95 443 "
    "vs 260 ms) and produced general-distribution pauses up to about a second. After stage 2, the comma "
    "distribution sits inside the teacher's range (p50 140 vs 180 ms; p95 205 vs 260 ms; max 300 vs "
    "260 ms), and the general-distribution tail collapsed (p95 260 vs 600 ms; max 280 vs 960 ms). The "
    "stock libritts_r voice, which barely pauses at commas at all (p50 45 ms), is included as a lower reference."
)
table(
    ["voice", "comma p50", "comma p95", "comma max", "general p50", "general p95", "general max"],
    [["amy (teacher)", "180", "260", "260", "140", "220", "340"],
     ["ex02 stage 1", "330", "443", "500", "200", "600", "960"],
     ["ex02 stage 2", "140", "205", "300", "140", "260", "280"],
     ["libritts_r 3922", "45", "80", "80", "120", "310", "400"]],
    [34, 22, 22, 22, 22, 22, 22],
    "Table 3: Pause distributions (ms) across voices. All rows use the 3-repeat protocol.",
)
body(
    "One measurement caveat: the targeted punctuation rows use the longest internal silence per sentence "
    "as a proxy for the mark's pause, which can select an unrelated pause. The general-distribution rows "
    "(all internal silences of at least 100 ms) have no such proxy problem and tell the same story."
)

subsection("Ablation: continued training without normalization")
body(
    "To isolate the normalization from the effect of additional training, the stage-1 checkpoint was "
    "trained for a further 2.69 h on the unmodified dataset with identical hyperparameters (two legs; "
    "an L4 preemption was resumed from the run's own checkpoint). The resulting voice sits near stage 1 "
    "on every tail metric and far from stage 2 (Table 4). A small convergence effect is visible "
    "(comma p50 330 to 270 ms), but it does not explain the collapse into the teacher's range. The "
    "improvement is therefore attributed to the pause normalization."
)
table(
    ["voice", "comma p50", "comma p95", "comma max", "general p95", "general max"],
    [["stage 1", "330", "443", "500", "600", "960"],
     ["stage 2 (normalized, +2.31 h)", "140", "205", "300", "260", "280"],
     ["ablation (unmodified, +2.69 h)", "270", "448", "600", "506", "680"]],
    [52, 22, 22, 22, 22, 22],
    "Table 4: Ablation: continued training on the unmodified data leaves pauses near stage-1 levels (ms).",
)
pdf.ln(1)
pdf.set_font(SERIF, "", 8.5)
pdf.image("fig_comma_pauses.png", x=30, w=150)
pdf.ln(1)
pdf.set_font(SERIF, "I", 8.5)
pdf.multi_cell(0, 4.8, "Figure 1: Comma pause distributions across voices (Table 3, first three columns).", align="C")
pdf.ln(3)

section("5  Voice quality is preserved")
body(
    "Pause surgery must not damage the voice. Table 5 reports the standard 40-sentence evaluation "
    "(30 held-out Expresso gate texts plus reference and word-test sentences) for the teacher and the "
    "stage-2 voice. All six pre-registered checks pass: no more word errors than the teacher, UTMOS "
    "within 0.3 of the teacher (breakage alarm only), one clean voice (0.74 similar to the speaker's own "
    "recordings, against her own consistency of 0.48; 0.30 similar to amy, against a 0.37 ceiling), and "
    "time-to-first-audio and throughput within 10% of the teacher."
)
table(
    ["metric", "amy (teacher)", "ex02 stage 2"],
    [["word errors (lower better)", "12", "11"],
     ["UTMOS naturalness", "4.35", "4.25"],
     ["similarity to real ex02", "0.25", "0.74"],
     ["similarity to amy", "0.81", "0.30"],
     ["time to first audio", "144 ms", "107 ms"],
     ["throughput", "11.3x realtime", "13.3x realtime"]],
    [62, 40, 40],
    "Table 5: Standard voice metrics, teacher vs stage-2 voice.",
)
body(
    "Against the stage-1 baseline (17 word errors, UTMOS 4.01), stage 2 is cleaner on both counts while "
    "keeping voice identity (0.74 vs 0.75 similarity to the speaker's recordings)."
)

section("6  Synthesis-time pause clamp")
body(
    "As a guardrail independent of training, the pacing layer gained an opt-in max_pause_ms: internal "
    "silences longer than the cap are shortened by removing middle frames with a 3 ms crossfade. "
    "Leading and trailing silence and the hand-written inter-piece pauses are untouched; the clamp "
    "applies within synthesized pieces only. Table 6 shows the effect on the stage-1 voice at a 250 ms cap."
)
table(
    ["voice", "comma p95", "comma max", "general p95", "general max"],
    [["stage 1, clamp off", "443", "500", "600", "960"],
     ["stage 1, clamp on", "260", "260", "260", "280"]],
    [42, 25, 25, 25, 25],
    "Table 6: Pause clamp (250 ms cap) on the stage-1 voice, in ms.",
)
body(
    "The clamp is a hard upper bound on intra-piece pauses; it cannot fix pauses the model never "
    "produces (libritts_r's 45 ms commas are untouched), and inter-piece pauses set by the pacing table "
    "remain as written."
)

section("7  Negative results")
body(
    "A 93.6k-parameter BiLSTM pause model reached F1 0.934 on held-out data, but inspection showed it "
    "had mostly learned punctuation-to-pause mapping: 99.2% of its training pauses follow a comma or a "
    "period. It is kept as an opt-in; the hand-written table remains the default."
)
body(
    "A long-form retrain on 5.56 hours of audiobook narration failed cleanly (F1 0.354): only 1.4% of "
    "pauses in that data were not after punctuation, so there was no phrase-break signal to learn. "
    "Long-form data also did not buy word correctness: the stock libritts_r-medium voice renders "
    "\"semicolon\" as \"Sima\", while the ex02 and amy voices render it correctly."
)

section("8  Limitations")
body(
    "This is a single-speaker, single-architecture case study, not a general claim. The ablation "
    "isolates the normalization from the convergence effect for this speaker, but other speakers and "
    "architectures may behave differently. Listening evaluation so far "
    "is informal; a blind A/B between the two finalist checkpoints is in progress, with listening as "
    "the final quality gate."
)

section("9  Conclusion")
body(
    "Three cheap interventions \u2014 a 3-repeat measurement protocol, pause-normalized fine-tuning, and an "
    "opt-in synthesis-time clamp \u2014 brought a dramatic reader's pauses into the teacher's range without "
    "harming word correctness, naturalness, or voice identity. A controlled ablation attributes the "
    "improvement to the normalization rather than to additional training. The failures are part of the result: in "
    "this regime, pause behavior is dominated by punctuation, and neither a learned model nor long-form "
    "data changed that with the data available."
)

section("References")
pdf.set_font(SERIF, "", 9)
refs = [
    "J. Kim, J. Kong, and J. Son. Conditional variational autoencoder with adversarial learning for "
    "end-to-end text-to-speech. In ICML, 2021.",
    "Piper TTS. OHF-Voice. https://github.com/OHF-Voice/piper1-gpl.",
    "T. A. Nguyen et al. Expresso: A benchmark and analysis of discrete expressive speech resynthesis. "
    "In Interspeech, 2023.",
    "A. Radford et al. Robust speech recognition via large-scale weak supervision. In ICML, 2023.",
    "T. Saeki et al. UTMOS: UTokyo-SaruLab system for VoiceMOS Challenge 2022. In Interspeech, 2022.",
    "B. Desplanques, J. Thienpondt, and K. Demuynck. ECAPA-TDNN: Emphasized channel attention, "
    "propagation and aggregation in TDNN based speaker verification. In Interspeech, 2020.",
]
for r in refs:
    pdf.multi_cell(0, 5, r, align="J")
    pdf.ln(1.5)

pdf.output(OUT)
print("wrote", OUT)
