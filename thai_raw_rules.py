"""ตรวจ raw encoding ก่อนให้สระ semantic เปลี่ยนรูปเขียนไทย"""

# อ้างอิงตารางที่ตรวจสอบแล้ว: liblouis tables/th-g1.uti
# https://github.com/liblouis/liblouis/blob/ebae95b51d88e0bcaf8d37718d59b65b2d175116/tables/th-g1.uti

# ตรวจเฉพาะกฎ raw ที่มีแล้ว กฎแบ่งพยางค์อื่นยังใช้ parser
# ตารางบรรทัด 62 และ 233–235: ว = 2456, รูปเขียน ัว = 15
WRITTEN_VOWEL_FORMS = {'อัว': {'full': ('15',), 'reduced': ('2456',)}}


def raw_encoding_error(units):
    """ห้ามลดรูป ัว ที่ปรากฏใน raw เป็น ว """
    for unit in units: #ดูทีละพยาง
        forms = WRITTEN_VOWEL_FORMS.get(unit.get('vowel'))
        if forms is None or not unit.get('final'):
            continue
        for component in unit.get('components', []): #ดูทีละส่วนของพยางค์
            if (component['type'] == 'VOWEL' #สระ
                    and tuple(component.get('patterns', [])) == forms['full']):
                return 'NONSTANDARD_UA_CLOSED_FORM'
    return None
