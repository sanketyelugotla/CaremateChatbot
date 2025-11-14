from tools.lang_utils import detect_language


def test_detect_language_english():
    assert detect_language('Hello, how are you?') in ('en', 'en')


def test_detect_language_spanish():
    # Simple Spanish sentence
    assert detect_language('¿Cómo te llamas?')[:2] == 'es'


def test_detect_language_empty():
    # empty or whitespace should fallback to 'en'
    assert detect_language('') == 'en'
