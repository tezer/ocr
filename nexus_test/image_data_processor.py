import logging
import re
from typing import Dict, List, Tuple

from autocorrect import Speller

spell = Speller('en')
logger = logging.getLogger("__main__")

THRESHOLD = 80
replacements: Dict[int, List[Tuple[str, str, str]]] = dict()
dots = re.compile(r'^[cse0ona\.]+$')
best_score = 0


def spellcheck(s: str) -> str:
    result = []
    words = s.split()
    for word in words:
        if word.isupper():
            word = spell(word.lower())
            word = word.upper()
        else:
            word = spell(word)
        result.append(word)
    return ' '.join(result)


def get_score(data):
    lines = [line for line in data if line.startswith('5') and len(line) > 0]
    good = len(
        [line for line in lines if int(line.split('\t')[10]) >= THRESHOLD])
    bad = len(
        [line for line in lines if int(line.split('\t')[10]) < THRESHOLD])
    return good, bad, lines


def get_confidence_score(data: str, step: int, language: str) -> Tuple[int, int]:
    """
    Gets the number of well recognized words and the number of poorly recognized words For a data that has more
    recognized words than the previous best, recognized words are checked and replacements for wrong words found
    :param data: data from tesseract for each recognized substring
    :param step: current step
    :return: number of good words and number of bad words
    """
    # Skip the first line with headers
    data_lines = data.split('\n')[1:]
    good, bad, lines = get_score(data_lines)
    global best_score
    if good <= best_score:
        # if this run is not better than the previous best, than no need to process it
        return good, bad
    best_score = good
    good_words = set()
    replacements[step] = []
    for data_line in lines:
        d = data_line.split('\t')
        word = d[11].lower()
        confidence = int(d[10])
        if confidence >= THRESHOLD:
            good_words.add(word)
            logger.debug(f"Good: {d[11]} at {d[10]}")
        else:
            logger.debug(f"Bad: {d[11]} at {d[10]}")
            # Special case for poor dots line recognition
            if confidence < 10 and dots.match(d[11]):
                replacements[step].append(
                    (d[11], str("." * len(d[11])), d[10]))
            # If the word was already reliably recognized
            if len(word) > 4 and word in good_words:
                logger.debug("Whole: " + word)
                continue
            # OR if the world was a part of a reliably recognized string
            if len(word) > 4 and any([x for x in good_words if word in x]):
                logger.debug("Part: " + word)
                continue
            if language == 'en':
                corrected = spellcheck(d[11])
            else:
                corrected = d[11]
            if corrected != d[11]:
                replacements[step].append((d[11], corrected, d[10]))
                logger.debug(f"Replacement: {d[11]} > {corrected}; {d[10]}")
    return good, bad
