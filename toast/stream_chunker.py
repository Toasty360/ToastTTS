"""Decide when enough LLM text has arrived to speak the next piece.

Text arrives in small fragments (" buil", "ding", " an"). We keep it in a
buffer and cut off a piece as soon as we're sure about a natural pause
point. "Sure" means we have already seen the start of the next word, so:
  "3:15"   is never cut at the colon (no space after it)
  "1,000"  is never cut at the comma
  "Dr."    and "e.g." are not treated as sentence ends

Rules (same as the "smart" split):
  - cut at . ? ! ; : ... and dashes
  - cut at a comma only after a short piece ("Well,", "12,", "rhythm,");
    after a long clause the comma stays inside so the voice keeps its flow
  - cut before "because / but / although ..." once the piece has 6+ words
  - first piece: if no pause point turns up within a few words, cut anyway
    so speech can start fast ("flash" piece). A pause point that's already
    in view wins if it costs at most twice as long to synthesize.
  - running low: if the audio queued for playback is about to run out
    (a slow LLM), stop waiting for the ideal spot and take the first
    reasonable one. A short hesitation sounds human; dead air doesn't.
  - a piece that runs very long gets cut before "and / but / because ..."
"""

import re

from toast.splitter import ending_mark

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "approx."}
CONJUNCTIONS = {"and", "but", "because", "so", "which", "or", "while", "although"}
# Words where a speaker naturally takes a breath before continuing.
PIVOTS = {"because", "but", "although", "though", "unless", "whereas"}
# One-word openers: "Well," "So," "Okay,". Spoken alone, a model says them
# like a finished statement (research/notebook/12).
OPENERS = {"well", "so", "oh", "okay", "ok", "yes", "yeah", "no", "hmm", "actually", "honestly",
           "right", "sure", "look", "listen", "now", "alright", "anyway", "basically", "see"}


def is_opener(piece):
    return piece.endswith(",") and piece[:-1].strip().lower() in OPENERS


# Small words that lean on the next word: "ships at | 3:15" or "for the | team"
# sounds broken, so a quick cut never lands right after one.
LEANING_WORDS = {
    "a", "an", "the", "at", "to", "of", "in", "on", "for", "with", "by", "from", "into", "about",
    "and", "or", "my", "your", "our", "their", "his", "her", "its", "this", "that", "these",
    "those", "is", "are", "was", "were", "be", "as", "than", "very", "so",
}
PIVOT_MIN_WORDS = 6

# Synthesis time is a fixed overhead plus a cost per spoken character. The
# overhead is worth about 11 characters on amy and 22 on Kokoro (2026-10-02).
OVERHEAD_CHARS = 15
# A hurried piece may run to a pause point if that costs at most this many
# times the quick cut: "Wisdom is the right use of knowledge." stays whole.
MAX_COST_RATIO = 2.0


def synthesis_cost(text):
    # ponytail: a digit is read as several words ("347" → "three hundred
    # forty-seven"), so count it as 7 characters; time a real synthesis if this misleads.
    return OVERHEAD_CHARS + len(text) + 6 * sum(c.isdigit() for c in text)

# Punctuation followed by a space and the start of the next word, or a dash
# followed by the next word.
_CANDIDATE = re.compile(r"(?:\.\.\.|…|[,;:.?!])(?=\s+\S)|[—–](?=\s*\S)")
_WORD = re.compile(r"\S+(?=\s)")  # a word we know is complete (a space came after it)


class StreamChunker:
    def __init__(self, flash_words=4, short_piece_words=4, max_words=24, attach_openers=False):
        """attach_openers: keep "Well," with the words after it, so the model
        says it knowing more follows (at the cost of our pause after it)."""
        self.flash_words = flash_words
        self.attach_openers = attach_openers
        self.short_piece_words = short_piece_words
        self.max_words = max_words
        self._buffer = ""
        self._first_piece = True
        # Set by the pipeline: returns True when playback is about to run dry.
        self.running_low = lambda: False

    def feed(self, text):
        """Add new text; return any pieces that are ready: (piece, ends_sentence)."""
        self._buffer += text
        pieces = []
        while (cut := self._find_cut()) is not None:
            pieces.append(self._take(cut))
        return pieces

    def finish(self):
        """The LLM is done: whatever is left is the last piece."""
        pieces = self.feed("")
        if self._buffer.strip():
            pieces.append(self._take(len(self._buffer)))
        return pieces

    def _take(self, cut):
        piece = self._buffer[:cut].strip()
        self._buffer = self._buffer[cut:]
        self._first_piece = False
        return piece, ending_mark(piece) in (".", "?", "!")

    def _find_cut(self):
        hurry = self._first_piece or self.running_low()
        words = list(_WORD.finditer(self._buffer))
        # In a hurry, a few words now beat a pause point far away. (When the
        # whole text arrives at once, the first comma may be 10+ words in.)
        flash = self._flash_cut(words) if hurry and len(words) >= self.flash_words else None

        for match in _CANDIDATE.finditer(self._buffer):
            piece = self._buffer[:match.end()].strip()
            if self._is_good_cut(piece, hurry) and not self._glued_tail(piece, match.end(), hurry):
                cut = self._pivot_cut(words, before=match.start()) or match.end()
                # A pause point that's cheap enough to reach is worth the
                # wait; one that would delay the first sound a lot is not.
                too_slow = flash and (synthesis_cost(self._buffer[:cut].strip())
                                      > MAX_COST_RATIO * synthesis_cost(self._buffer[:flash].strip()))
                return flash if too_slow else cut

        if pivot := self._pivot_cut(words):
            return min(pivot, flash) if flash else pivot
        if flash:
            return flash
        if len(words) >= self.max_words:
            return self._long_piece_cut(words)
        return None

    def _is_good_cut(self, piece, hurry):
        mark = ending_mark(piece)
        last_word = piece.split()[-1].lower()
        if mark == "." and (last_word in ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word)):
            return False  # "Dr." or an initial like "J."
        if mark == "," and self.attach_openers:
            last_clause = piece[:-1].split(",")[-1].strip()
            if is_opener(last_clause + ","):
                return False
        if mark == "," and not hurry:
            clause = piece.rstrip(",").split(",")[-1]
            return len(clause.split()) <= self.short_piece_words
        return True

    def _glued_tail(self, piece, end, hurry):
        """"Good evening, | sir." sounds broken: a lone last word after a comma
        (sir, a name, please) is said in one breath with the words before it.
        A number stays a list item ("12, 45, 108.")."""
        if ending_mark(piece) != ",":
            return False
        tail = self._buffer[end:].split()
        if tail and re.match(r"[#$]?\d", tail[0]):
            return False
        if tail and tail[0][-1] in ".?!":
            return True
        # ponytail: a live stream's first piece can't wait to see the tail, so
        # "Good evening, | sir." is still cut there; waiting costs a word of TTFA.
        return len(tail) < 2 and not hurry

    def _pivot_cut(self, words, before=None):
        """Position just before a pivot word like "because", if the piece
        already has enough words in front of it."""
        for index, word in enumerate(words):
            if before is not None and word.start() >= before:
                break
            if index >= PIVOT_MIN_WORDS and word.group().lower() in PIVOTS:
                return word.start()
        return None

    def _flash_cut(self, words):
        # Don't cut right after a number or single letter ("at 3" may be
        # followed by "PM", "J" by "Smith") or a small leaning word ("ships
        # at | 3:15"). Move on to the next word instead.
        for word in words[self.flash_words - 1:]:
            text = word.group()
            if re.fullmatch(r"[#$]?\d[\d:.,%]*|[A-Za-z]\.?", text):
                continue
            if text.lower().strip(".,;:!?\"'") in LEANING_WORDS:
                continue
            return word.end()
        return None

    def _long_piece_cut(self, words):
        # Prefer cutting just before a joining word in the second half.
        for word in reversed(words[len(words) // 2:]):
            if word.group().lower() in CONJUNCTIONS:
                return word.start()
        return words[self.max_words - 1].end()
