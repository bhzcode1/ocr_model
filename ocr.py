"""AuraNav OCR module: camera frame -> text lines -> sentence for TTS.

Runs fully offline on a Raspberry Pi 5 (RapidOCR = PP-OCR models on ONNX Runtime).
Reads English and Hindi (Devanagari), including bilingual signs; OCR(hindi=False) for English only.
Hindi costs ~+8% time on English frames (paired median) and is ~4x faster on Hindi frames than
English-only mode, which misreads them and burns its retry passes.

    python ocr.py sign.jpg        # read an image file
    python ocr.py --camera 0      # ENTER = "read" button, q = quit (no window)
"""
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

# Words that get spoken first, with a "Warning" prefix (feeds the Priority Engine).
HAZARD_TRIGGERS = {"caution", "danger", "warning", "emergency", "beware", "fire",
                   # Hindi: caution, caution(noun), danger (both spellings), warning, emergency, fire
                   "सावधान", "सावधानी", "खतरा", "ख़तरा", "चेतावनी", "आपातकालीन", "आग"}
HAZARD_EXACT = {"stop", "रुकें", "रुको"}  # only alone: a STOP sign is a hazard, "Bus Stop" is not


@dataclass
class Line:
    text: str
    score: float
    box: np.ndarray  # 4x2 corner points, pixels

    @property
    def is_hazard(self):
        words = self.text.lower().strip(" .!।").split()  # । = Hindi full stop
        return bool(HAZARD_TRIGGERS & set(words)) or " ".join(words) in HAZARD_EXACT


class OCR:
    def __init__(self, min_score=0.6, max_side=960, hindi=True):
        # Tuning knobs: raise min_score if it reads noise, lower it for blurry/far signs.
        # max_side caps the detector input; bigger = small/far text found, but slower on the Pi.
        # (RapidOCR's default upscales the *short* side to 736 -> 5-9 s on wide frames.)
        # hindi=False skips the Devanagari model: English-only, faster.
        self.min_score = min_score
        p = {"Det.limit_type": "max", "Det.limit_side_len": max_side, "Global.log_level": "error"}
        self.engine = RapidOCR(params=p)
        # Default PP-OCRv6 has no Devanagari. This 8 MB PP-OCRv5 model reads Hindi and English;
        # only its recognizer is used (detection is shared with self.engine).
        self.hindi = RapidOCR(params={**p, "Rec.lang_type": LangRec.DEVANAGARI, "Rec.model_type": ModelType.MOBILE,
                                      "Rec.ocr_version": OCRVersion.PPOCRV5}) if hindi else None

    def read(self, frame, retry_below=0.85):
        """BGR frame -> Lines in reading order (top-to-bottom, left-to-right).

        Fast path is one OCR pass. Only when it finds nothing, or is unsure (mean score
        < retry_below), it retries on brightened / sharpened / 90-degree-rotated copies and
        keeps the most confident result. Measured on hard synthetic frames: 0.949 -> 0.976
        character accuracy, median latency unchanged (retries only cost time on bad frames).
        """
        best = self._pass(frame)
        if best and np.mean([l.score for l in best]) >= retry_below:
            return reading_order(best)
        for img in (enhance(frame), sharpen(frame),
                    cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)):
            lines = self._pass(img)  # rotated passes: boxes are in rotated coords, fine for ordering
            if confidence(lines) > confidence(best):
                best = lines
        return reading_order(best)

    def _pass(self, img):
        # RapidOCR remembers use_det/use_cls/use_rec between calls: always pass all three.
        if self.hindi:
            reads = self._read_bilingual(img)
        else:
            r = self.engine(img, use_det=True, use_cls=True, use_rec=True)
            reads = [] if r.txts is None else zip(r.txts, r.scores, r.boxes)
        return [Line(t.strip(), float(s), np.asarray(b)) for t, s, b in reads
                if s >= self.min_score and any(c.isalnum() for c in t)]  # drop punctuation-only junk

    def _read_bilingual(self, img):
        """Detect once, read every box with both recognizers (batched), route by script.

        A box whose Hindi read contains Devanagari keeps it, otherwise the English model's read wins.
        (Picking by score fails: on 'निकास EXIT' the English model confidently reads just 'EXIT'.)
        """
        boxes = self.engine(img, use_det=True, use_cls=False, use_rec=False).boxes
        if boxes is None:
            return []
        crops = [crop_box(img, b) for b in boxes]
        en_imgs, _ = self.engine.cls_and_rotate(crops)
        en = self.engine.recognize_txt(en_imgs)
        r = self.hindi.recognize_txt(crops)
        hi = list(zip(r.txts, r.scores))
        # The shared upside-down classifier flips Devanagari, so it isn't used for Hindi:
        # instead unsure Hindi reads get one batched retry rotated 180 degrees.
        unsure = [i for i, (_, s) in enumerate(hi) if s < 0.85]
        if unsure:
            flipped = self.hindi.recognize_txt([cv2.rotate(crops[i], cv2.ROTATE_180) for i in unsure])
            for i, t, s in zip(unsure, flipped.txts, flipped.scores):
                if s > hi[i][1]:
                    hi[i] = (t, s)
        return [(ht, hs, b) if is_devanagari(ht) else (et, es, b)
                for (ht, hs), et, es, b in zip(hi, en.txts, en.scores, boxes)]

    def speak(self, frame):
        return to_speech(self.read(frame))


def is_devanagari(text):
    return any("ऀ" <= c <= "ॿ" for c in text)


def crop_box(img, box):
    """Perspective-crop a detected 4-point box; tall crops are turned sideways (vertical text)."""
    pts = np.asarray(box, np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    c = cv2.warpPerspective(img, M, (max(w, 1), max(h, 1)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.rot90(c).copy() if h >= 1.5 * w else c


def confidence(lines):
    """Total confident characters: prefers the pass that read more text, more surely."""
    return sum(l.score * len(l.text) for l in lines)


def enhance(img):
    """Night/low-contrast fix: gamma-lift dark frames to mid-grey, denoise very dark ones, CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, m = lab[..., 0], lab[..., 0].mean()
    if m < 90:
        L = (255 * (L / 255.0) ** (np.log(0.5) / np.log(max(m, 1) / 255))).astype(np.uint8)
        if m < 40:
            L = cv2.fastNlMeansDenoising(L, None, 10, 7, 21)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen(img):
    """Unsharp mask: helps mild motion blur / defocus."""
    return cv2.addWeighted(img, 1.8, cv2.GaussianBlur(img, (0, 0), 3), -0.8, 0)


def sharpness(img):
    """Variance of the Laplacian: higher = sharper. Used to pick the best camera frame."""
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def reading_order(lines):
    """Group boxes into rows (centers within half a line height), then sort each row by x."""
    lines = sorted(lines, key=lambda l: l.box[:, 1].mean())
    rows = []
    for l in lines:
        cy, h = l.box[:, 1].mean(), np.ptp(l.box[:, 1])
        if rows and abs(cy - rows[-1][0]) < h / 2:
            rows[-1][1].append(l)
        else:
            rows.append([cy, [l]])
    return [l for _, row in rows for l in sorted(row, key=lambda l: l.box[:, 0].min())]


def to_speech(lines):
    """Lines -> one sentence for TTS. Hazard text is spoken first."""
    if not lines:
        return ""
    hazards = [l.text for l in lines if l.is_hazard]
    rest = [l.text for l in lines if not l.is_hazard]
    parts = (["Warning: " + ". ".join(hazards)] if hazards else []) + rest
    return ". ".join(parts) + "."


def main(argv):
    ocr = OCR()
    if argv[:1] == ["--camera"]:
        # No preview window: the glasses have no screen (and headless OpenCV has no GUI).
        cap = cv2.VideoCapture(int(argv[1]) if len(argv) > 1 else 0)
        if not cap.isOpened():
            sys.exit("Cannot open camera")
        while input("ENTER = read, q = quit > ").strip().lower() != "q":
            # Head-worn camera shakes: grab a burst (also flushes stale buffered frames)
            # and OCR only the sharpest one. Tuning knob: more frames = steadier, slower.
            frames = [f for ok, f in (cap.read() for _ in range(8)) if ok]
            if not frames:
                print("Camera read failed.")
                continue
            print(ocr.speak(max(frames[3:] or frames, key=sharpness)) or "No text found.")
        cap.release()
    elif argv:
        frame = cv2.imread(argv[0])
        if frame is None:
            sys.exit(f"Cannot read image: {argv[0]}")
        print(ocr.speak(frame) or "No text found.")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
