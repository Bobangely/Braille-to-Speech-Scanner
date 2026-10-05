"""ตรวจ raw encoding ก่อนให้สระ semantic เปลี่ยนรูปเขียนไทย"""

# อ้างอิงตารางที่ตรวจสอบแล้ว ไม่ดาวน์โหลดกฎระหว่างสแกน
RAW_STANDARD = 'LIBLOUIS_TH_G1_SCANNER_SUBSET'
RAW_RULES_SOURCE = (
    'https://github.com/liblouis/liblouis/blob/'
    'ebae95b51d88e0bcaf8d37718d59b65b2d175116/tables/th-g1.uti'
)

# ตรวจเฉพาะกฎ raw ที่มีหลักฐานแล้ว กฎแบ่งพยางค์อื่นยังใช้ parser เดิม
# ตารางบรรทัด 62 และ 233–235: ว = 2456, รูปเขียน ัว = 15
WRITTEN_VOWEL_FORMS = {'อัว': {'full': ('15',), 'reduced': ('2456',)}}


def raw_encoding_error(units):
    """ห้ามลดรูป ัว ที่ปรากฏใน raw เป็น ว โดยอาศัยตัวสะกดที่เดาเพิ่ม"""
    for unit in units:
        forms = WRITTEN_VOWEL_FORMS.get(unit.get('vowel'))
        if forms is None or not unit.get('final'):
            continue
        for component in unit.get('components', []):
            if (component['type'] == 'VOWEL'
                    and tuple(component.get('patterns', [])) == forms['full']):
                return 'NONSTANDARD_UA_CLOSED_FORM'
    return None


def candidate_validation(units, structural_error, *, compatibility=False, raw_available=True):
    """เก็บผลของทั้งสอง mode แต่รับ compatibility ต่อเมื่อเปิดใช้ชัดเจน"""
    standard_error = structural_error or (raw_encoding_error(units) if raw_available
                                         else 'RAW_PATTERNS_UNAVAILABLE')
    standard_valid = standard_error is None
    compatibility_valid = structural_error is None
    accepted = standard_valid or (compatibility and compatibility_valid)
    mode = 'standard' if standard_valid else 'compatibility' if accepted else None
    return dict(valid=accepted,
                validation_error=None if accepted else standard_error,
                standard_valid=standard_valid, standard_error=standard_error,
                compatibility_valid=compatibility_valid, compatibility_error=structural_error,
                accepted_mode=mode,
                acceptance_reason=('STANDARD_RAW_SUPPORTED' if standard_valid else
                                   'EXPLICIT_COMPATIBILITY' if accepted else
                                   'COMPATIBILITY_DISABLED' if compatibility_valid else
                                   'INVALID_STRUCTURE'))
