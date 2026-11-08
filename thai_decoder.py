"""API ถอดภาษาไทย: parser เบรลล์ดิบ แล้วแปลงเป็นข้อความไทย"""

import logging

from thai_braille import _break_before, cell_read_warning, is_unread_cell
from thai_parser import parse_braille  # จัดพยางค์จากเซลล์ 
from thai_renderer import render_thai  # เรียงพยางค์เป็นตัวหนังสือ

logger = logging.getLogger(__name__)


def decode_thai_details(cells, *, confidence_threshold=None, compatibility=False): #main
    result = render_thai(parse_braille(cells, compatibility=compatibility),
                         confidence_threshold=confidence_threshold)
    if result['needs_review'] and logger.isEnabledFor(logging.DEBUG):
        logger.debug('[THAI_COMPOSITION_REVIEW] raw_cells=%r decoded_cells=%r '
                     'syllables=%r composed_output=%r', result['raw_cells'],
                     result['raw_tokens'], result['syllables'], result['text'])
    return result


def decode_thai(cells, *, compatibility=False):
    return decode_thai_details(cells, compatibility=compatibility)['text'] 


def thai_verbose(cells, *, compatibility=False):
    """เก็บป้ายและช่วงเซลล์สำหรับวาดภาพ และแนบการตัดสินใจประกอบคำไว้วิเคราะห์"""
    result = decode_thai_details(cells, compatibility=compatibility)
    tokens = result['raw_tokens']
    for syllable in result['syllables']:
        token = tokens[syllable['indices'][0]] # token ตัวแรกของพยางค์
        token['composition'] = syllable # แนบผลพยางค์ไว้

        if syllable['needs_review']:
            # การตัดคำเปลี่ยนป้ายเซลล์ที่รู้แล้วให้ผิดไม่ได้
            # แยกช่องทางตรวจทานออกจากข้อผิดพลาดการตรวจจับและการจับคู่
            if (syllable['status'] == syllable.get('boundary_status')
                    and syllable.get('boundary_indices')):
                token['composition_warning'] = syllable['status']
                tokens[syllable['boundary_indices'][0]]['segmentation'] = syllable
            elif not token.get('warning'):
                token['warning'] = syllable['status']
    return tokens
