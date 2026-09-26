import re

from toast.stream_chunker import StreamChunker


def pieces_from(text, token_size=4, **options):
    """Feed text in small fragments like an LLM would; return piece texts."""
    chunker = StreamChunker(**options)
    pieces = []
    for token in re.findall(rf"\s*\S{{1,{token_size}}}", text):
        pieces += chunker.feed(token)
    pieces += chunker.finish()
    return [piece for piece, _ in pieces]


def test_short_asides_and_list_items_get_their_own_piece():
    assert pieces_from("Well, to be honest, it works. Look: 12, 45, and 108.") == [
        "Well,", "to be honest,", "it works.", "Look:", "12,", "45,", "and 108.",
    ]


def test_comma_after_long_clause_stays_inside_the_piece():
    text = "You can easily tell whether a model is reciting data like a machine, or actually talking."
    assert pieces_from("Well, " + text) == ["Well,", text]


def test_times_and_big_numbers_are_not_cut():
    text = "Meet me at 3:15 PM with 1,000 dollars."
    assert pieces_from(text, flash_words=99) == [text]


def test_abbreviations_are_not_sentence_ends():
    assert pieces_from("Ask Dr. Smith today. Thanks.") == ["Ask Dr. Smith today.", "Thanks."]


def test_first_piece_is_cut_early_when_no_comma_comes():
    pieces = pieces_from("If the meeting starts soon we should leave now.")
    assert pieces[0] == "If the meeting starts"


def test_first_piece_stays_short_when_whole_text_arrives_at_once():
    chunker = StreamChunker()
    pieces = chunker.feed("There's a slight delay with your order, but it ships Friday.") + chunker.finish()
    assert pieces[0][0] == "There's a slight delay"


def test_flash_cut_does_not_split_a_number_from_its_unit():
    pieces = pieces_from("Meet me at 3 PM near the station please.", flash_words=4)
    assert pieces[0] == "Meet me at 3 PM"


def test_flash_cut_never_ends_on_a_small_leaning_word():
    chunker = StreamChunker()
    pieces = chunker.feed("Order #4521 ships at 3:15 PM today.") + chunker.finish()
    assert pieces[0][0] == "Order #4521 ships at 3:15 PM"


def test_openers_can_stay_attached_to_the_next_words():
    assert pieces_from("Well, to be honest, it works.")[:2] == ["Well,", "to be honest,"]
    assert pieces_from("Well, to be honest, it works.", attach_openers=True)[0] == "Well, to be honest,"


def test_pause_before_because_once_the_piece_is_long_enough():
    pieces = pieces_from("Well, we should leave by noon today because traffic is terrible.")
    assert pieces == ["Well,", "we should leave by noon today", "because traffic is terrible."]


def test_marks_which_pieces_end_a_sentence():
    chunker = StreamChunker()
    result = chunker.feed("Well, it works. Really") + chunker.finish()
    assert result == [("Well,", False), ("it works.", True), ("Really", False)]


def test_running_low_takes_the_first_reasonable_cut():
    text = "Well, building a fast voice pipeline, with lots of care and patience, takes time."
    relaxed = pieces_from(text)
    assert relaxed[1] == "building a fast voice pipeline, with lots of care and patience, takes time."

    chunker = StreamChunker()
    chunker.running_low = lambda: True
    hurried = chunker.feed(text + " ") + chunker.finish()
    assert hurried[1][0] == "building a fast voice pipeline,"
