"""S-84: serve/emitted_reuse.py - a continuation tokenized the way the model wrote it."""
import unittest

from serve.emitted_reuse import EmittedReuse, align, common_prefix


class Tok:
    """A toy vocabulary: single letters, some merged pairs, two specials."""
    VOCAB = ["a", "b", "c", "d", "ab", "cd", "abc", "\xe4".encode("latin-1"), b"\xb8", b"\xad", "中".encode(), "<eos>", "<think>"]

    def __init__(self):
        self.tokens = self.VOCAB
        self.special_tokens = {"<eos>": 11, "<think>": 12}

    def token_bytes(self, i):
        t = self.VOCAB[i]
        return t if isinstance(t, bytes) else t.encode()

    def decode(self, ids):
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", "replace")


A, B, C, D, AB, CD, ABC, E4, B8, AD, ZHONG, EOS, THINK = range(13)


class AlignTest(unittest.TestCase):
    def setUp(self):
        self.tok = Tok()
        self.er = EmittedReuse(self.tok)

    def walk(self, kept, new, at=0):
        return align(kept, new, at, self.er._bytes, self.er._special)

    def test_same_text_other_ids(self):
        self.assertEqual(self.walk([A, B, C, D], [AB, CD]), (4, 2))
        self.assertEqual(self.walk([AB, C, D], [A, B, CD]), (3, 3))

    def test_stops_at_a_real_difference(self):
        self.assertEqual(self.walk([A, B, C, A], [AB, C, D]), (3, 2))     # "abc" is shared, then a / d
        self.assertEqual(self.walk([A, B], [AB, C]), (2, 1))

    def test_never_stops_inside_a_token(self):
        # kept "ab|c", new "abc|d": the only shared boundary before the difference is the start
        self.assertEqual(self.walk([AB, D], [ABC, D]), (0, 0))

    def test_a_character_split_across_tokens(self):
        self.assertEqual(self.walk([E4, B8, AD, A], [ZHONG, A]), (4, 2))

    def test_specials_match_by_id_only(self):
        self.assertEqual(self.walk([A, EOS, A], [A, EOS, A]), (3, 3))
        self.assertEqual(self.walk([A, B, EOS], [AB, THINK]), (2, 1))

    def test_common_prefix(self):
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix(list(range(10000)), list(range(10000))), 10000)
        self.assertEqual(common_prefix(list(range(10000)), list(range(9000)) + [0] * 1000), 9000)


class ResyncTest(unittest.TestCase):
    def setUp(self):
        self.tok = Tok()
        self.er = EmittedReuse(self.tok)
        self.er.MIN_SHARED = 1

    def emit(self, prompt, reply):
        got = list(self.er.track(prompt, iter(reply)))
        self.assertEqual([t for t in reply if t is not None], [t for t in got if t is not None])

    def test_a_continuation_takes_the_emitted_ids(self):
        self.emit([D, EOS], [A, B, None, C, EOS])                    # the model wrote a b c
        new = [D, EOS, ABC, EOS, D]                                  # the client sent "abc" back, then more
        out = self.er.resync(new)
        self.assertEqual(out, [D, EOS, A, B, C, EOS, D])
        self.assertEqual(self.tok.decode(out), self.tok.decode(new))

    def test_untouched_when_the_text_differs(self):
        self.emit([D, EOS], [A, B, C, EOS])
        new = [D, EOS, A, D, EOS]
        self.assertIs(self.er.resync(new), new)

    def test_untouched_when_it_already_continues(self):
        self.emit([D, EOS], [A, B, C, EOS])
        new = [D, EOS, A, B, C, EOS, D]
        self.assertIs(self.er.resync(new), new)

    def test_two_replies_of_one_conversation(self):
        self.emit([D, EOS], [A, B, EOS])
        self.emit([D, EOS, A, B, EOS, D, EOS], [C, D, EOS])          # the second turn went in with the emitted ids
        new = [D, EOS, AB, EOS, D, EOS, CD, EOS, A]
        self.assertEqual(self.er.resync(new), [D, EOS, A, B, EOS, D, EOS, C, D, EOS, A])
        self.assertEqual(len(self.er.seqs), 1)                       # the longer pass replaced the one it continues

    def test_the_longest_kept_one_wins_and_is_not_overwritten(self):
        self.emit([D, EOS], [A, B, EOS])                             # an older state of the conversation
        self.er.seqs.append(self.er.seqs[0][:2] + type(self.er.seqs[0])("i", [AB, EOS, C, D, EOS]))   # a later one
        new = [D, EOS, AB, EOS, CD, EOS]
        out = self.er.resync(new)
        self.assertEqual(out, [D, EOS, AB, EOS, C, D, EOS])

    def test_close_reaches_the_engine(self):
        closed = []

        def gen():
            try:
                yield A
                yield B
            finally:
                closed.append(True)
        g = self.er.track([D], gen())
        self.assertEqual(next(g), A)
        g.close()
        self.assertEqual(closed, [True])
        self.assertEqual(list(self.er.seqs[-1]), [D, A])


if __name__ == "__main__":
    unittest.main()
