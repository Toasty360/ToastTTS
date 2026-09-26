from toast.word_check import number_words, word_errors


def errors(expected, heard):
    return word_errors(expected, heard)[0]


def test_time_formats_match():
    assert errors("at 3:15 PM without", "at 3.15 p.m. without") == 0
    assert errors("at 3:15 PM without", "at 3:15pm without") == 0


def test_numbers_in_words_match_digits():
    assert errors("these numbers: 12, 45, and 108.", "these numbers: twelve, forty-five, and one hundred and eight.") == 0
    assert errors("1,000 dollars", "one thousand dollars") == 0
    assert number_words(108) == "one hundred eight"


def test_british_spelling_matches():
    assert errors("the harbor and the neighborhood", "the harbour and the neighbourhood") == 0


def test_joined_and_split_words_match():
    assert errors("subtle micro-breaks", "subtle microbreaks") == 0
    assert errors("an ultra-low latency voice", "an ultralowlatency voice") == 0
    assert errors("a semicolon here", "a semi colon here") == 0


def test_contractions_match():
    assert errors("it isn't just about speed; it's about cadence", "it is not just about speed, it is about cadence") == 0


def test_real_mistakes_still_count():
    assert word_errors("after a semicolon", "after a semi") == (1, [("semicolon", "semi")])
    assert errors("about cadence", "about kings") == 1
    assert errors("the listener will notice", "the listener notice") == 1
