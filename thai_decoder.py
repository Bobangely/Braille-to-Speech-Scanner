"""API ถอดภาษาไทย: parser เบรลล์ดิบ แล้วแปลงเป็นข้อความไทย"""

import logging

from thai_parser import parse_braille
from thai_renderer import render_thai  # เรียงพยางค์เป็นตัวหนังสือ

logger = logging.getLogger(__name__)


def decode_thai_details(cells):
    """ถอดเซลล์ : ข้อความ พยางค์ และ token รายเซลล์"""
    parsed = parse_braille(cells)  #จัดพยางค์
    result = render_thai(parsed)
    if result['needs_review']:
        logger.debug('[THAI_COMPOSITION_REVIEW] cells=%r syllables=%r text=%r',
                     result['raw_cells'], result['syllables'], result['text'])
    return result


def decode_thai(cells): #ถอดเซลล์เป็นข้อความไทย
    return decode_thai_details(cells)['text']


def thai_verbose(cells): #คืน token รายเซลล์ 
    result = decode_thai_details(cells) #ถอดทั้งคำ
    tokens = result['raw_tokens']
    for syllable in result['syllables']:
        if not syllable['needs_review']:
            continue
        # ปัญหาตัดคำ ไม่ใช่เซลล์อ่านผิด 
        if syllable['status'] == syllable.get('boundary_status') and syllable.get('boundary_indices'):
            continue
        first = tokens[syllable['indices'][0]]
        if not first.get('warning'):  #ถ้าไม่มี warning จาก camera
            first['warning'] = syllable['status']
    return tokens