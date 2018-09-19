# this file is an adaptation from the work at mozilla deepspeech github.com/mozilla/DeepSpeech

import kenlm
import re
from heapq import heapify

import numpy as np
from pattern3.metrics import levenshtein_similarity, levenshtein

from util.rnn_util import ALLOWED_CHARS


def ler(ground_truth, prediction):
    """
    The LER is defined as the edit distance between two strings
    The score is normalized to a value between 0 and 1
    """
    return 1 - levenshtein_similarity(ground_truth, prediction)


def wer(ground_truth, prediction):
    """
    The WER is defined as the editing/Levenshtein distance on word level (not on character-level!).
    The score is normalized to a value between 0 and 1.
    """
    return ler(ground_truth.split(), prediction.split())


def wers(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR assert len(ground_truth) > 0: looks like data is missing!'
    rates = [wer(ground_truth, prediction) for (ground_truth, prediction) in zip(ground_truths, predictions)]
    return rates, np.mean(rates)


def lers(ground_truths, predictions):
    assert len(ground_truths) > 0, f'ERROR assert len(ground_truth) > 0: looks like data is missing!'
    assert len(ground_truths) == len(predictions), f'ERROR: not same number of ground truths and predictions!'
    rates = [levenshtein(ground_truth, prediction) for (ground_truth, prediction) in zip(ground_truths, predictions)]
    norm_rates = [ler(ground_truth, prediction) for (ground_truth, prediction) in zip(ground_truths, predictions)]
    return rates, np.mean(rates), norm_rates, np.mean(norm_rates)


def get_LM():
    """
    Lazy-load language model (TED corpus, Kneser-Ney, 4-gram, 30k word LM)
    :return: KenLM language model
    """
    global MODEL
    if MODEL is None:
        MODEL = kenlm.Model('./lm/libri-timit-lm.klm')
    return MODEL


def words(text):
    """
    splits a text into a list of words
    :param text: a text-string
    :return: list of word-strings
    """
    return re.findall(r'\w+', text.lower())


def score(word_list):
    """
    Use LM to calculate a log10-based probability for a given sentence (as a list of words)
    :param word_list:
    :return:
    """
    return get_LM().score(' '.join(word_list), bos=False, eos=False)


def correction(sentence):
    """
    Get most probable spelling correction for a given sentence.
    :param sentence:
    :return:
    """
    beam_width = 1024
    layer = [(0, [])]  # list of (score, 2-gram)-pairs
    for word in words(sentence):
        layer = [(-score(node + [word_c]), node + [word_c]) for word_c in candidate_words(word) for sc, node in layer]
        heapify(layer)
        layer = layer[:beam_width]
    return ' '.join(layer[0][1])


def candidate_words(word):
    """
    Generate possible spelling corrections for a given word.
    :param word: single word as a string
    :return: list of possible spelling corrections for each word
    """
    return known_words([word]) \
           or known_words(edits_1(word)) \
           or known_words(edits_2(word)) \
           or [word]  # fallback: the original word as a list


def known_words(word_list):
    """
    Filters out from a list of words the subset of words that appear in the vocabulary of KNOWN_WORDS.
    :param word_list: list of word-strings
    :return: set of unique words that appear in vocabulary
    """
    return set(w for w in word_list if w in KNOWN_WORDS)


def edits_1(word_list):
    """
    generates a list of all words with edit distance 1 for a list of words
    :param word_list: list of word-strings
    :return:
    """
    splits = [(word_list[:i], word_list[i:]) for i in range(len(word_list) + 1)]  # all possible splits

    deletes = [L + R[1:] for L, R in splits if R]  # all words with one character removed
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]  # all words with two swapped characters
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALLOWED_CHARS]  # all words with one character replaced
    inserts = [L + c + R for L, R in splits for c in ALLOWED_CHARS]  # all words with one character inserted
    return set(deletes + transposes + replaces + inserts)


def edits_2(word):
    """
    generates a list of all words with edit distance 2 for a list of words
    :param word: list of word-strings
    :return:
    """
    return (e2 for e1 in edits_1(word) for e2 in edits_1(e1))


# globals
MODEL = None
# Load known word set
with open('./lm/words.txt') as f:
    KNOWN_WORDS = set(words(f.read()))
