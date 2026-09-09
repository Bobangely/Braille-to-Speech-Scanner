"""
Braille Reader - Configuration
================================
อักษรเบรลล์ Grade 1 (English) mapping และ detection parameters

Braille Cell Layout:
    (1) (4)
    (2) (5)
    (3) (6)

แต่ละ cell มี 6 ตำแหน่ง จุดที่มี/ไม่มี จะกำหนดตัวอักษร
"""

# =============================================================================
# Braille Grade 1 - ตัวอักษร a-z
# key = frozenset ของเลข dot ที่มีอยู่
# value = ตัวอักษรที่ตรงกัน
# =============================================================================
BRAILLE_TO_CHAR = {
    frozenset({1}):          'a',
    frozenset({1, 2}):       'b',
    frozenset({1, 4}):       'c',
    frozenset({1, 4, 5}):    'd',
    frozenset({1, 5}):       'e',
    frozenset({1, 2, 4}):    'f',
    frozenset({1, 2, 4, 5}): 'g',
    frozenset({1, 2, 5}):    'h',
    frozenset({2, 4}):       'i',
    frozenset({2, 4, 5}):    'j',
    frozenset({1, 3}):       'k',
    frozenset({1, 2, 3}):    'l',
    frozenset({1, 3, 4}):    'm',
    frozenset({1, 3, 4, 5}): 'n',
    frozenset({1, 3, 5}):    'o',
    frozenset({1, 2, 3, 4}): 'p',
    frozenset({1, 2, 3, 4, 5}): 'q',
    frozenset({1, 2, 3, 5}): 'r',
    frozenset({2, 3, 4}):    's',
    frozenset({2, 3, 4, 5}): 't',
    frozenset({1, 3, 6}):    'u',
    frozenset({1, 2, 3, 6}): 'v',
    frozenset({2, 4, 5, 6}): 'w',
    frozenset({1, 3, 4, 6}): 'x',
    frozenset({1, 3, 4, 5, 6}): 'y',
    frozenset({1, 3, 5, 6}): 'z',
    # สัญลักษณ์พิเศษ
    frozenset({3, 4, 5, 6}): '#',   # number indicator
    frozenset({6}):          ',',
    frozenset({2, 6}):       '!',
    frozenset({2, 3, 6}):    '?',
    frozenset({2, 5, 6}):    '.',
}

# Reverse mapping: ตัวอักษร -> dot positions
CHAR_TO_BRAILLE = {v: k for k, v in BRAILLE_TO_CHAR.items()}


# =============================================================================
# Detection Parameters - ค่าปรับสำหรับการตรวจจับจุดสี
# =============================================================================
class DetectionConfig:
    """ค่า config สำหรับ dot detection pipeline"""

    # ----- HSV Color Ranges -----
    # สีน้ำเงิน 
    BLUE_HSV_LOWER = (90, 80, 50)
    BLUE_HSV_UPPER = (135, 255, 255)

    # สีแดง 
    RED_HSV_LOWER_1 = (0, 80, 50)
    RED_HSV_UPPER_1 = (15, 255, 255)
    RED_HSV_LOWER_2 = (165, 80, 50)
    RED_HSV_UPPER_2 = (180, 255, 255)

    # สีเขียว
    GREEN_HSV_LOWER = (35, 80, 50)
    GREEN_HSV_UPPER = (85, 255, 255)

    # สีดำ 
    BLACK_INTENSITY_MAX = 80

    # ----- Blob Detection -----
    MIN_DOT_AREA = 8          # พื้นที่ pixel ขั้นต่ำของจุด (รองรับภาพขนาดเล็ก/ครอป และตัด noise)
    MAX_DOT_AREA = 8000       # พื้นที่ pixel สูงสุดของจุด
    MIN_CIRCULARITY = 0.45    # ค่า circularity ขั้นต่ำ (1.0 = วงกลมสมบูรณ์)

    # ----- Morphological Operations -----
    MORPH_KERNEL_SIZE = 3     # ขนาด kernel สำหรับ morphology

    # ----- Grid Clustering -----
    CLUSTER_TOLERANCE = 0.5   # สัดส่วนของ dot_spacing ที่ใช้จัด cluster

    # ----- สีที่รองรับ -----
    SUPPORTED_COLORS = ['blue', 'red', 'green', 'black']
    DEFAULT_COLOR = 'blue'
