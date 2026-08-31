"""
Braille Reader - Text to Speech (TTS) Module
=============================================
แปลงข้อความที่อ่านได้ (ทั้งภาษาไทยและอังกฤษ) เป็นเสียงพูด
รองรับ:
1. Offline mode ผ่าน pyttsx3 (SAPI5 บน Windows, espeak บน Linux/ARM)
2. Online/High-Quality mode ผ่าน gTTS (Google TTS ภาษาไทยและอังกฤษเสียงธรรมชาติ)
"""

import os
import sys
import threading


class TextToSpeech:
    """TTS Controller สำหรับออกเสียงข้อความทั้งภาษาไทยและอังกฤษ"""

    def __init__(self, default_lang='english'):
        self.default_lang = default_lang
        self._offline_engine = None
        self._init_offline_engine()

    def _init_offline_engine(self):
        """กำหนดค่า offline TTS engine (pyttsx3)"""
        try:
            import pyttsx3
            self._offline_engine = pyttsx3.init()
            self._offline_engine.setProperty('rate', 150)    # ความเร็วในการพูด
            self._offline_engine.setProperty('volume', 1.0)  # ระดับเสียง (0.0 - 1.0)
        except Exception as e:
            self._offline_engine = None

    def speak(self, text, lang='english', method='auto', save_file=None):
        """
        ออกเสียงข้อความ

        Parameters
        ----------
        text : str
            ข้อความที่ต้องการพูด
        lang : str, optional
            ภาษา ('english' หรือ 'thai')
        method : str, optional
            วิธีออกเสียง ('auto', 'offline', 'online')
        save_file : str, optional
            path ไฟล์เสียงที่ต้องการบันทึก (เช่น 'output/speech.mp3')

        Returns
        -------
        bool
            True หากออกเสียงสำเร็จ
        """
        if not text or not text.strip():
            return False

        text = text.strip()
        lang_lower = lang.lower()
        is_thai = lang_lower in ('thai', 'th')

        # เลือก method
        if method == 'auto':
            # ภาษาไทยแนะนำใช้ gTTS ถ้ามีเน็ตเพื่อสำเนียงไทยที่ถูกต้อง
            # ถ้า offline หรือภาษาอังกฤษ ใช้ pyttsx3
            if is_thai:
                success = self._speak_online(text, lang='th', save_file=save_file)
                if not success:
                    success = self._speak_offline(text, is_thai=True)
                return success
            else:
                success = self._speak_offline(text, is_thai=False)
                if not success:
                    success = self._speak_online(text, lang='en', save_file=save_file)
                return success
        elif method == 'online':
            t_lang = 'th' if is_thai else 'en'
            return self._speak_online(text, lang=t_lang, save_file=save_file)
        else:
            return self._speak_offline(text, is_thai=is_thai)

    def _speak_offline(self, text, is_thai=False):
        """ออกเสียงแบบ offline ผ่าน pyttsx3"""
        if self._offline_engine is None:
            self._init_offline_engine()

        if self._offline_engine is None:
            return False

        try:
            # เลือก voice ให้ตรงกับภาษาถ้ามี
            voices = self._offline_engine.getProperty('voices')
            target_voice = None

            if is_thai:
                for v in voices:
                    v_name = v.name.lower()
                    if 'thai' in v_name or 'pattara' in v_name or 'th-th' in v_name or 'thailand' in v_name:
                        target_voice = v.id
                        break
            else:
                for v in voices:
                    v_name = v.name.lower()
                    if 'zira' in v_name or 'david' in v_name or 'english' in v_name or 'en-us' in v_name:
                        target_voice = v.id
                        break

            if target_voice:
                self._offline_engine.setProperty('voice', target_voice)

            self._offline_engine.say(text)
            self._offline_engine.runAndWait()
            return True
        except Exception as e:
            print(f"  [TTS Offline Error]: {e}")
            return False

    def _speak_online(self, text, lang='th', save_file=None):
        """ออกเสียงแบบ online ผ่าน gTTS (Google TTS) คุณภาพเสียงชัดเจน"""
        try:
            from gtts import gTTS
            import tempfile

            tts = gTTS(text=text, lang=lang, slow=False)

            if save_file:
                os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
                tts.save(save_file)
                audio_path = save_file
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
                    audio_path = tf.name
                tts.save(audio_path)

            # เล่นไฟล์เสียง
            self._play_audio(audio_path)

            # ลบ temp file หากไม่ได้ระบุ save_file
            if not save_file and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

            return True
        except Exception as e:
            return False

    def _play_audio(self, audio_file):
        """เล่นไฟล์เสียงตาม OS"""
        try:
            if sys.platform.startswith('win'):
                # ใช้ Windows Media Player / native command
                os.system(f'start "" /min wmplayer "{audio_file}"')
            elif sys.platform.startswith('linux'):
                os.system(f'mpg123 -q "{audio_file}" || aplay "{audio_file}" || ffplay -nodisp -autoexit "{audio_file}"')
            elif sys.platform.startswith('darwin'):
                os.system(f'afplay "{audio_file}"')
        except Exception:
            pass


# Global singleton instance
_tts_instance = None

def speak(text, lang='english', method='auto', save_file=None):
    """
    ฟังก์ชันเรียกใช้งานง่ายสำหรับออกเสียงข้อความ
    """
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TextToSpeech()
    return _tts_instance.speak(text, lang=lang, method=method, save_file=save_file)
