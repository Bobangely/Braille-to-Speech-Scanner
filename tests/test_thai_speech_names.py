"""Independent expected readings for the 44 Thai consonants."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import wave

from camera_reader import RealTimeBrailleScanner
from edge_stream import prepare_edge_text
from tts import prepare_thai_speech_text


EXPECTED = [
    ('ก', 'กอไก่'), ('ข', 'ขอไข่'), ('ฃ', 'ขอขวด'), ('ค', 'คอควาย'),
    ('ฅ', 'คอคน'), ('ฆ', 'คอระฆัง'), ('ง', 'งองู'), ('จ', 'จอจาน'),
    ('ฉ', 'ฉอฉิ่ง'), ('ช', 'ชอช้าง'), ('ซ', 'ซอโซ่'), ('ฌ', 'ชอเฌอ'),
    ('ญ', 'ยอหญิง'), ('ฎ', 'ดอชฎา'), ('ฏ', 'ตอปฏัก'), ('ฐ', 'ถอฐาน'),
    ('ฑ', 'ทอมณโฑ'), ('ฒ', 'ทอผู้เฒ่า'), ('ณ', 'นอเณร'), ('ด', 'ดอเด็ก'),
    ('ต', 'ตอเต่า'), ('ถ', 'ถอถุง'), ('ท', 'ทอทหาร'), ('ธ', 'ทอธง'),
    ('น', 'นอหนู'), ('บ', 'บอใบไม้'), ('ป', 'ปอปลา'), ('ผ', 'ผอผึ้ง'),
    ('ฝ', 'ฝอฝา'), ('พ', 'พอพาน'), ('ฟ', 'ฟอฟัน'), ('ภ', 'พอสำเภา'),
    ('ม', 'มอม้า'), ('ย', 'ยอยักษ์'), ('ร', 'รอเรือ'), ('ล', 'ลอลิง'),
    ('ว', 'วอแหวน'), ('ศ', 'สอศาลา'), ('ษ', 'สอฤๅษี'), ('ส', 'สอเสือ'),
    ('ห', 'หอหีบ'), ('ฬ', 'ลอจุฬา'), ('อ', 'อออ่าง'), ('ฮ', 'ฮอนกฮูก'),
]


class ThaiSpeechNameTests(unittest.TestCase):
    def test_all_44_letters_use_full_spoken_names_on_each_thai_path(self):
        self.assertEqual(len(EXPECTED), 44)
        for letter, name in EXPECTED:
            with self.subTest(letter=letter):
                self.assertEqual(prepare_thai_speech_text(letter), name)
                self.assertEqual(prepare_edge_text(letter, 'th-TH-NiwatNeural'), name)
                self.assertEqual(prepare_edge_text(letter, 'th-TH-PremwadeeNeural'), name)

    def test_boundaries_include_whitespace_punctuation_and_digits(self):
        for letter, name in EXPECTED:
            for left, right in [(' ', ' '), ('\n', '\n'), ('“', '”'), ('[', ']'),
                                ('1', '2'), ('๓', '๔'), ('/', ','), ('_', '_')]:
                with self.subTest(letter=letter, left=left):
                    source = left+letter+right
                    self.assertEqual(prepare_edge_text(source, 'th-TH-NiwatNeural'), left+name+right)

    def test_normal_words_marks_and_foreign_letters_are_not_split(self):
        source = 'กิน คน คำ คง ฃา กฃ ฅน ก่ กิ กุ กํ ก์ เก แก ใก ไก โก Aก กB ก\u0301'
        self.assertEqual(prepare_thai_speech_text(source), source)
        self.assertEqual(prepare_edge_text(source, 'th-TH-NiwatNeural'), source)
        self.assertEqual(prepare_edge_text('ก ข ฃ ฅ ฮ', 'en-US-JennyNeural'), 'ก ข ฃ ฅ ฮ')

    def test_numbered_lesson_keeps_order_and_other_numbers(self):
        source = 'ฃ พยัญชนะตัวที่ 3\n(ปัจจุบันเลิกใช้แล้ว)\nนิทานตัว ค\nพยัญชนะตัวที่ 4'
        expected = 'ขอขวด พยัญชนะตัวที่ สาม\n วงเล็บเปิด ปัจจุบันเลิกใช้แล้ว วงเล็บปิด \nนิทานตัว คอควาย\nพยัญชนะตัวที่ สี่'
        self.assertEqual(prepare_thai_speech_text(source), expected)
        for number, word in [('1', 'หนึ่ง'), ('๓', 'สาม'), ('11', 'สิบเอ็ด'), ('20', 'ยี่สิบ'),
                             ('21', 'ยี่สิบเอ็ด'), ('31', 'สามสิบเอ็ด'), ('44', 'สี่สิบสี่')]:
            self.assertEqual(prepare_thai_speech_text('พยัญชนะตัวที่ '+number+' ค'),
                             'พยัญชนะตัวที่ '+word+' คอควาย')
        self.assertEqual(prepare_thai_speech_text('เวลา 3 นาฬิกา 123 ๔๔'), 'เวลา 3 นาฬิกา 123 ๔๔')
        self.assertEqual(prepare_thai_speech_text('พยัญชนะตัวที่ 3คน'), 'พยัญชนะตัวที่ 3คน')

    def test_preprocessing_is_idempotent_and_does_not_reorder_units(self):
        source = '\n'.join(letter for letter, _ in EXPECTED)
        expected = '\n'.join(name for _, name in EXPECTED)
        spoken = prepare_thai_speech_text(source)
        self.assertEqual(spoken, expected)
        self.assertEqual(prepare_thai_speech_text(spoken), expected)
        self.assertEqual(prepare_edge_text(spoken, 'th-TH-NiwatNeural'), expected)

    def test_s_repeat_uses_cache_and_b_preserves_original_display(self):
        scanner = RealTimeBrailleScanner(res_preset='480p', tts_backend='edge')
        speaker = scanner._speech
        self.addCleanup(speaker.stop)
        source = 'พยัญชนะตัวที่ 3 ค\nฃ ฅ'
        scanner._confirmed_text = source
        jobs = []
        process = Mock()
        process.poll.return_value = None
        def start(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            return process
        with patch('piper_tts.subprocess.Popen', side_effect=start):
            scanner._handle_key(ord('s'), None, None)
            scanner._handle_key(ord('s'), None, None)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['text'], 'พยัญชนะตัวที่ สาม คอควาย\nขอขวด คอคน')
            with wave.open(jobs[0]['output'], 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(b'\0\0'*240)
            Path(jobs[0]['output']).with_name('ready').touch()
            process.poll.return_value = 0
            scanner._handle_key(ord('b'), None, None)
            scanner._handle_key(ord('s'), None, None)
            self.assertTrue(jobs[1]['cached'])
            self.assertEqual(jobs[1]['text'], jobs[0]['text'])
        self.assertEqual(scanner._confirmed_text, source)
