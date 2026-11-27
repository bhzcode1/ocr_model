"""AuraNav voice output: offline Piper TTS (English + Hindi voices), beeps, interruptible playback.

    speaker = Speaker()
    speaker.beep("start")             # "I heard the button, reading..."
    speaker.say("निकास EXIT, ahead.")   # Hindi words -> Hindi voice, the rest -> English voice
    speaker.stop()                    # cut off speech, e.g. when the next read starts

Voices live in ./voices (python -m piper.download_voices --data-dir voices en_US-lessac-medium hi_IN-priyamvada-medium).
Fallback per language: Piper voice -> espeak-ng -> silent (logged; the CLI still prints the text).
"""
import io
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("auranav.speech")
VOICES = Path(__file__).with_name("voices")
RATE = 22050  # Piper medium voices and espeak-ng both output 22.05 kHz


def is_devanagari(text):
    return any("ऀ" <= c <= "ॿ" for c in text)


def split_scripts(text):
    """'निकास EXIT' -> [('hi', 'निकास'), ('en', 'EXIT')]. Numbers/punctuation stay with the run they're in."""
    runs = []
    for word in text.split():
        lang = "hi" if is_devanagari(word) else "en" if any(c.isalpha() for c in word) else None
        if runs and lang in (None, runs[-1][0]):
            runs[-1][1].append(word)
        else:
            runs.append([lang or "en", [word]])
    return [(lang, " ".join(words)) for lang, words in runs]


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16), w.getframerate()


def to_wav(samples, rate=RATE):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(np.asarray(samples, np.int16).tobytes())
    return b.getvalue()


def tone(*parts):
    """parts = (freq_hz, seconds) pairs -> int16 samples, with short fades so it doesn't click."""
    out = []
    for f, sec in parts:
        t = np.arange(int(RATE * sec)) / RATE
        env = np.minimum(1, np.minimum(t, sec - t) / 0.01)
        out.append(0.3 * 32767 * env * np.sin(2 * np.pi * f * t))
    return np.concatenate(out)


BEEPS = {"start": tone((880, 0.08)),                 # read started (button registered)
         "error": tone((330, 0.15), (220, 0.3))}     # something is broken: camera, model


def piper_voice(name):
    path = VOICES / f"{name}.onnx"
    if not path.exists():
        return None
    try:
        from piper import PiperVoice
    except ImportError:
        return None
    voice = PiperVoice.load(path)

    def synth(text):  # punctuation-only text gives no chunks -> empty audio
        chunks = list(voice.synthesize(text))
        return np.concatenate([np.zeros(0, np.int16)] + [c.audio_int16_array for c in chunks]), voice.config.sample_rate
    synth("ok")  # warm-up: the first synthesis is ~20x slower than the rest
    return synth


def espeak_voice(lang):
    exe = shutil.which("espeak-ng")
    if not exe:
        return None
    return lambda text: read_wav(subprocess.run([exe, "-v", lang, "--stdout", text],
                                                capture_output=True, check=True).stdout)


def cloned_voices(sample):
    """sample = a few seconds of someone's speech -> {"en": fn, "hi": fn} speaking in that voice (XTTS-v2).
    {} if coqui-tts is missing or the model can't load; the caller falls back to Piper."""
    try:
        from TTS.api import TTS
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").synthesizer.tts_model
        gpt, spk = model.get_conditioning_latents(audio_path=[str(sample)])  # once: listening to the sample is slow
    except Exception as e:  # not installed, license not accepted, no sample file...
        log.warning("voice clone from %s unavailable (%s): using Piper", sample, e)
        return {}

    def voice(lang):
        def synth(text):
            wav = np.asarray(model.inference(text, lang, gpt, spk)["wav"])
            return (np.clip(wav, -1, 1) * 32767).astype(np.int16), 24000  # XTTS outputs 24 kHz floats
        return synth
    return {lang: voice(lang) for lang in ("en", "hi")}


class Speaker:
    def __init__(self, en="en_US-lessac-medium", hi="hi_IN-priyamvada-medium", clone=None, voices=None, play=None):
        """clone = speech sample (path, relative to this folder) to speak in that voice instead of Piper's.
        voices = {"en": fn, "hi": fn}, fn(text) -> (int16 samples, rate); play(wav_bytes, wait). Both for tests."""
        if voices is None:
            cloned = cloned_voices(Path(__file__).parent / clone) if clone else {}
            voices = {lang: cloned.get(lang) or piper_voice(name) or espeak_voice(lang)
                      for lang, name in (("en", en), ("hi", hi))}
            for lang, fn in voices.items():
                if fn is None:
                    log.warning("no %s voice (Piper model in %s or espeak-ng): it won't be spoken", lang, VOICES)
        self.voices = voices
        self._play = play or self._play_default
        self._proc = None

    def say(self, text, wait=False):
        """Speak text, interrupting anything already playing. wait=True blocks until done."""
        audio = []
        for lang, chunk in split_scripts(text):
            fn = self.voices.get(lang)
            if fn:
                a, rate = fn(chunk)
                if rate != RATE:  # resample: voices of different rates are glued into one clip
                    a = np.interp(np.arange(0, len(a), rate / RATE), np.arange(len(a)), a)
                audio += [a, np.zeros(int(RATE * 0.12))]
        if audio:
            self._play(to_wav(np.concatenate(audio)), wait)

    def beep(self, kind, wait=False):
        self._play(to_wav(BEEPS[kind]), wait)

    def stop(self):
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, 0)
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _play_default(self, data, wait):
        self.stop()
        if sys.platform == "win32":  # dev machines; winsound can't play from memory asynchronously
            import winsound
            path = Path(tempfile.gettempdir()) / "auranav_say.wav"
            path.write_bytes(data)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | (0 if wait else winsound.SND_ASYNC))
        elif aplay := shutil.which("aplay"):  # Raspberry Pi OS: alsa-utils
            self._proc = subprocess.Popen([aplay, "-q", "-"], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            t = threading.Thread(target=self._proc.communicate, args=(data,), daemon=True)
            t.start()
            if wait:
                t.join()
        else:
            log.warning("no audio player found (install alsa-utils for aplay)")
