from unittest import TestCase

from hamcrest import assert_that, is_

from util.rnn_util import encode, decode
from utils import text_to_int_sequence, int_to_text_sequence


class TestEncodingDecoding(TestCase):

    def test_encoding(self):
        text = 'foo bar'
        encoded = encode(text)
        encoded_ref = text_to_int_sequence(text)
        assert_that(encoded, is_(encoded_ref))

    def test_decoding(self):
        int_sequence = [6, 15, 15, 0, 2, 1, 18, 28, 28]
        decoded = decode(int_sequence)
        decoded_ref = ''.join(int_to_text_sequence(int_sequence))
        assert_that(decoded, is_(decoded_ref))
