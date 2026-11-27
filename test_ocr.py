"""Self-check: python test_ocr.py"""
import time

import cv2
import numpy as np

from ocr import OCR


def sign(*rows):
    """Render a white sign; rows = [(text, x), ...] per line."""
    img = np.full((80 * len(rows) + 40, 900, 3), 255, np.uint8)
    for i, row in enumerate(rows):
        for text, x in row:
            cv2.putText(img, text, (x, 70 + 80 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


ocr = OCR()

# reading order: two words on one row read left-to-right, then the next row
assert [l.text for l in ocr.read(sign([("Platform", 20), ("Two", 500)], [("Exit", 20)]))] \
    == ["Platform", "Two", "Exit"]

# hazard text is spoken first, with a Warning prefix
assert ocr.speak(sign([("Room 204", 20)], [("CAUTION", 20)])) == "Warning: CAUTION. Room 204."

# "stop" alone is a hazard, inside a place name it is not
assert ocr.speak(sign([("STOP", 20)])) == "Warning: STOP."

# blank frame -> nothing to say
assert ocr.speak(np.full((300, 300, 3), 255, np.uint8)) == ""

# edge cases rescued by the retry pass
exit_sign = sign([("EXIT", 20)])
assert ocr.speak((exit_sign * 0.04).astype(np.uint8)) == "EXIT.", "very dark frame"
assert ocr.speak(cv2.rotate(sign([("Bus Stop", 20)]), cv2.ROTATE_90_CLOCKWISE)) == "Bus Stop.", "sideways text"

# camera burst: the sharpest frame wins
from ocr import sharpness
assert sharpness(exit_sign) > sharpness(cv2.GaussianBlur(exit_sign, (0, 0), 3))

# Hindi / bilingual (signs are browser-rendered for correct Devanagari, cached in bench_signs/)
from render_sign import render_sign
pad = lambda im: cv2.copyMakeBorder(im, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(128, 128, 128))
assert ocr.speak(pad(render_sign("निकास"))) == "निकास."
assert ocr.speak(pad(render_sign("प्लेटफार्म 3 Platform 3"))) == "प्लेटफार्म 3 Platform 3."
assert ocr.speak(pad(render_sign("सावधान फर्श गीला है"))).startswith("Warning: सावधान"), "Hindi hazard"
assert ocr.speak(cv2.rotate(pad(render_sign("शौचालय")), cv2.ROTATE_90_COUNTERCLOCKWISE)) == "शौचालय.", "sideways Hindi"

# English-only mode still works (faster, no Devanagari model)
assert OCR(hindi=False).speak(exit_sign) == "EXIT."

# 1-character reads: a clear "5" is kept, but only above short_score (stray clutter is usually unsure)
five = sign([("5", 400)])
assert ocr.speak(five) == "5."
assert OCR(short_score=1.01).speak(five) == "", "single char below short_score is dropped"

# --- for blind users: where the text is, aiming hints, no repeats, voice routing ---
from ocr import Announcer, Result, unrotate

# location: the frame is 900 px wide, so x=20 is the left third, x=380 the middle, x=720 the right
assert ocr.describe(exit_sign).speech == "EXIT, on your left."
assert ocr.describe(sign([("EXIT", 380)])).speech == "EXIT, ahead."
assert ocr.describe(sign([("CAUTION", 20), ("EXIT", 720)])).speech == "Warning: CAUTION, on your left. EXIT, on your right."
assert ocr.describe(exit_sign).hints == [], "a good read has no hints"

# sideways text: boxes are mapped back to the original frame, so location still works
side = np.full((900, 900, 3), 255, np.uint8)
side[:, 760:880] = cv2.rotate(sign([("Bus Stop", 20)]), cv2.ROTATE_90_CLOCKWISE)  # sideways, in the right third
assert ocr.describe(side).speech == "Bus Stop, on your right."
img = np.zeros((40, 60), np.uint8); img[5, 7] = 1  # pixel at x=7, y=5
for r in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
    y, x = np.argwhere(cv2.rotate(img, r))[0]
    assert unrotate(np.array([[x, y]]), r, img.shape).tolist() == [[7, 5]]

# aiming hints
assert ocr.describe(np.zeros((480, 640, 3), np.uint8)).hints == ["Too dark"]
assert ocr.describe(np.full((480, 640, 3), 200, np.uint8)).hints == [], "blank wall: nothing to say"
assert "Text cut off on the left, turn left" in ocr.describe(sign([("Platform", -45)])).hints
assert ocr.describe(np.zeros((480, 640, 3), np.uint8)).announcement == "No text found. Too dark."

# auto mode: the same sign isn't repeated; hazards come back sooner; OCR jitter still counts as the same
now = [0.0]
ann = Announcer(clock=lambda: now[0])
res = lambda *texts, hazard=False: Result(". ".join(texts) + ".", [type("L", (), {"text": t}) for t in texts], [], hazard)
assert ann.filter(res("Platform 3")) == "Platform 3."
now[0] = 5; assert ann.filter(res("Platfrom 3")) is None, "jitter"
assert ann.filter(res("Exit")) == "Exit.", "a new sign is spoken"
assert ann.filter(res("Warning: CAUTION", hazard=True)) == "Warning: CAUTION."
now[0] = 12; assert ann.filter(res("Warning: CAUTION", hazard=True)) is None
now[0] = 16; assert ann.filter(res("Warning: CAUTION", hazard=True)) == "Warning: CAUTION.", "hazard repeats after 10 s"
now[0] = 31; assert ann.filter(res("Platform 3")) == "Platform 3.", "sign repeats after 30 s"
assert ann.filter(Result("")) is None

# voice: each script goes to its own voice, numbers stay with their words, all in one clip
from speech import Speaker, split_scripts
assert split_scripts("निकास EXIT") == [("hi", "निकास"), ("en", "EXIT")]
assert split_scripts("प्लेटफार्म 3 Platform 3.") == [("hi", "प्लेटफार्म 3"), ("en", "Platform 3.")]
said, played = [], []
fake = lambda lang: lambda text: (said.append((lang, text)), (np.zeros(100, np.int16), 22050))[1]
spk = Speaker(voices={"en": fake("en"), "hi": fake("hi")}, play=lambda data, wait: played.append(data))
spk.say("Warning: खतरा DANGER, ahead.")
assert said == [("en", "Warning:"), ("hi", "खतरा"), ("en", "DANGER, ahead.")] and len(played) == 1
spk.beep("start"); assert len(played) == 2

t = time.perf_counter()
ocr.read(sign([("Watch your step", 20)]))
print(f"ok  ({(time.perf_counter() - t) * 1000:.0f} ms per frame on this machine)")
