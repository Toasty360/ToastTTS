from toast.text_normalize import normalize_for_speech


def test_hash_before_a_number_is_read_as_number():
    assert normalize_for_speech("Order #4521 ships today.") == "Order number 4521 ships today."


def test_hash_before_letters_is_dropped():
    assert normalize_for_speech("Your order #ORD-24589 shipped.") == "Your order ORD-24589 shipped."


def test_times_are_written_as_words():
    assert normalize_for_speech("at 3:15 PM today") == "at three fifteen PM today"
    assert normalize_for_speech("at 10:05 am") == "at ten oh five am"
    assert normalize_for_speech("at 9:00 AM") == "at nine AM"
    assert normalize_for_speech("meet at 12:00.") == "meet at twelve o'clock."
    assert normalize_for_speech("around 11:30 PM.") == "around eleven thirty PM."


def test_things_that_only_look_like_times_are_left_alone():
    assert normalize_for_speech("the score was 3:75") == "the score was 3:75"
    assert normalize_for_speech("version 12:34:56") == "version 12:34:56"


def test_ordinary_text_is_unchanged():
    text = "Well, it's about $29.99 and 45 minutes."
    assert normalize_for_speech(text) == text
