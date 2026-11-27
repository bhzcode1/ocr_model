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

t = time.perf_counter()
ocr.read(sign([("Watch your step", 20)]))
print(f"ok  ({(time.perf_counter() - t) * 1000:.0f} ms per frame on this machine)")
