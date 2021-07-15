from nexus_test import __version__
from nexus_test.image_data_processor import spellcheck


def test_version():
    assert __version__ == '0.1.0'


def test_spellcheck():
    result = spellcheck("NOT MORE THAN TWO")
    assert result == "NOT MORE THAN TWO"
