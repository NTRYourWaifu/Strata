"""CPU-only regression for the opt-in Chinese thinking (S-78): python -m unittest serve.test_reasoning_language."""
import threading
import unittest
from serve.server import (Service, ByteTokenizer, MockEngine, ZH_THINK_LEAD, ZH_THINK_RETRY, reasoning_is_english)


class Tokenizer(ByteTokenizer):
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class RecordingEngine(MockEngine):
    def __init__(self, tokenizer, scripts):
        super().__init__(tokenizer, scripts, max_context=100000)
        self.calls = []

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.calls.append((list(ids), max_new))
        yield from super().generate(ids, max_new, sampling, cancel, **kwargs)


ENGLISH = ('The user is asking me to sort a list of numbers in place, so let me think about which way is the best one '
           'here and then write it down step by step.\n</think>\nEnglish answer.')
CHINESE = '使用者要我把一串數字排序，先想清楚要用哪一種排序法，再一步一步寫出來，最後檢查邊界情況。\n</think>\n中文答案。'


class ReasoningLanguage(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>user\n排序這串數字<|im_end|>\n<|im_start|>assistant\n<think>\n',
                                   parse_special=True)
        self.lead = self.tok.encode(ZH_THINK_LEAD)
        self.retry = self.tok.encode(ZH_THINK_RETRY)

    def run_case(self, scripts, zh="zh", request=None, max_new=4000, thinking=True):
        engine = RecordingEngine(self.tok, scripts)
        service = Service(engine, self.tok, None)
        service.reasoning_language = zh
        events = list(service.run(self.ids, thinking, [], max_new, request if request is not None else {},
                                  threading.Event()))
        text = {k: ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == k)
                for k in ('reasoning', 'content')}
        return engine, events, text

    def test_the_judge(self):
        self.assertTrue(reasoning_is_english('The user is asking me to sort a list of numbers in place'))
        self.assertFalse(reasoning_is_english('使用者要我把一串數字排序，先想清楚要用哪一種排序法'))
        # Chinese thinking about code keeps its identifiers: still Chinese
        self.assertFalse(reasoning_is_english('先看 quick_sort 的 pivot 怎麼選，再看 partition 函式裡 left 跟 right 兩個'
                                              '指標怎麼移動，最後確認 while 迴圈的結束條件有沒有寫對，還有遞迴的邊界'))
        self.assertFalse(reasoning_is_english('ok'))

    def test_off_by_default(self):
        engine, events, text = self.run_case([ENGLISH], zh=None)
        self.assertEqual(engine.calls[0][0], self.ids)
        self.assertEqual(len(engine.calls), 1)
        self.assertTrue(text['reasoning'].startswith('The user'))

    def test_chinese_thinking_is_one_pass_behind_the_lead(self):
        engine, events, text = self.run_case([CHINESE])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][0], self.ids + self.lead)
        self.assertEqual(engine.calls[0][1], 4000 - len(self.lead))
        self.assertEqual(text['reasoning'].strip(), ZH_THINK_LEAD + CHINESE.split('\n</think>')[0])
        self.assertEqual(text['content'], '中文答案。')
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_english_start_is_dropped_unseen_and_thought_again(self):
        engine, events, text = self.run_case([ENGLISH, CHINESE])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[1][0], self.ids + self.lead + self.retry)
        self.assertEqual(engine.calls[1][1], 4000 - len(self.lead) - len(self.retry))
        self.assertNotIn('The user', text['reasoning'])
        self.assertEqual(text['reasoning'].strip(), ZH_THINK_LEAD + ZH_THINK_RETRY + CHINESE.split('\n</think>')[0])
        self.assertEqual(text['content'], '中文答案。')
        done = events[-1][1]
        self.assertEqual(done['finish'], 'stop')
        # the dropped tokens are not counted: the lead, the retry lead, the second pass and its end token
        self.assertEqual(done['completion_tokens'],
                         len(self.lead) + len(self.retry) + len(self.tok.encode(CHINESE, parse_special=True)) + 1)

    def test_only_one_retry(self):
        engine, events, text = self.run_case([ENGLISH, ENGLISH])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(text['reasoning'].count('The user'), 1)       # the second English start goes out as it is
        self.assertEqual(text['content'], 'English answer.')

    def test_a_request_turns_it_off(self):
        engine, events, text = self.run_case([ENGLISH], request={'reasoning_language': 'off'})
        self.assertEqual(engine.calls[0][0], self.ids)
        self.assertEqual(len(engine.calls), 1)

    def test_a_request_turns_it_on(self):
        engine, events, text = self.run_case([ENGLISH, CHINESE], zh=None, request={'reasoning_language': 'zh'})
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(text['content'], '中文答案。')

    def test_thinking_shorter_than_the_check_goes_out_whole(self):
        engine, events, text = self.run_case(['Easy one.\n</think>\nDone.'])
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(text['reasoning'].strip(), ZH_THINK_LEAD + 'Easy one.')
        self.assertEqual(text['content'], 'Done.')

    def test_not_without_thinking_nor_with_a_small_cap(self):
        engine, events, text = self.run_case(['Plain answer.'], thinking=False)
        self.assertEqual(engine.calls[0][0], self.ids)
        engine, events, text = self.run_case([ENGLISH], max_new=40)
        self.assertEqual(engine.calls[0][0], self.ids)
        self.assertEqual(len(engine.calls), 1)


    def test_the_switch_file_turns_it_off_and_on_without_a_reload(self):
        import os, tempfile
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            engine = RecordingEngine(self.tok, [CHINESE])
            service = Service(engine, self.tok, None)
            service.reasoning_language, service.reasoning_language_file = "zh", path
            for flag, want in (("0", self.ids), ("1", self.ids + self.lead), ("", self.ids + self.lead)):
                with open(path, "w", encoding="ascii") as f:
                    f.write(flag)
                list(service.run(self.ids, True, [], 4000, {}, threading.Event()))
                self.assertEqual(engine.calls[-1][0], want)
            os.remove(path)                                 # no file = on
            list(service.run(self.ids, True, [], 4000, {}, threading.Event()))
            self.assertEqual(engine.calls[-1][0], self.ids + self.lead)
            with open(path, "w", encoding="ascii") as f:    # a request's own field still wins over the file
                f.write("0")
            list(service.run(self.ids, True, [], 4000, {"reasoning_language": "zh"}, threading.Event()))
            self.assertEqual(engine.calls[-1][0], self.ids + self.lead)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == '__main__':
    unittest.main()
