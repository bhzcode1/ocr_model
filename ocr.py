"""AuraNav OCR module: camera frame -> text lines -> sentence for TTS, for blind users.

Runs fully offline on a Raspberry Pi 5 (RapidOCR = PP-OCR models on ONNX Runtime).
Reads English and Hindi (Devanagari), including bilingual signs; OCR(hindi=False) for English only.
Hindi costs ~+8% time on English frames (paired median) and is ~4x faster on Hindi frames than
English-only mode, which misreads them and burns its retry passes.

    python ocr.py sign.jpg [--speak]     # read an image file
    python ocr.py --camera 0 [--speak]   # ENTER = "read" button, q = quit (no window)
    python ocr.py --auto 0 --speak       # hands-free: announce each new sign once
Settings: config.toml next to this file (or --config PATH). Frames are never saved to disk.
"""
import argparse
import difflib
import logging
import sys
import time
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

from speech import is_devanagari

log = logging.getLogger("auranav.ocr")

# Words that get spoken first, with a "Warning" prefix (feeds the Priority Engine).
HAZARD_TRIGGERS = {"caution", "danger", "warning", "emergency", "beware", "fire",
                   # Hindi: caution, caution(noun), danger (both spellings), warning, emergency, fire
                   "सावधान", "सावधानी", "खतरा", "ख़तरा", "चेतावनी", "आपातकालीन", "आग"}
HAZARD_EXACT = {"stop", "रुकें", "रुको"}  # only alone: a STOP sign is a hazard, "Bus Stop" is not
ROTATIONS = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)


@dataclass
class Line:
    text: str
    score: float
    box: np.ndarray  # 4x2 corner points, pixels, in the original frame

    @property
    def is_hazard(self):
        words = self.text.lower().strip(" .!।").split()  # । = Hindi full stop
        return bool(HAZARD_TRIGGERS & set(words)) or " ".join(words) in HAZARD_EXACT


@dataclass
class Result:
    speech: str                                 # hazards first, with "on your left"/"ahead"/"on your right"
    lines: list = field(default_factory=list)   # Lines in reading order
    hints: list = field(default_factory=list)   # aiming guidance, [] when the read is good
    hazard: bool = False

    @property
    def announcement(self):
        """Everything to say for a button press: the text (or "No text found.") then the hints."""
        return " ".join([self.speech or "No text found."] + [h + "." for h in self.hints])


class OCR:
    def __init__(self, min_score=0.6, max_side=960, hindi=True, retry_below=0.85, locations=True, guide=None,
                 short_score=0.95):
        # Tuning knobs: raise min_score if it reads noise, lower it for blurry/far signs.
        # max_side caps the detector input; bigger = small/far text found, but slower on the Pi.
        # (RapidOCR's default upscales the *short* side to 736 -> 5-9 s on wide frames.)
        # hindi=False skips the Devanagari model: English-only, faster.
        # guide = guidance() thresholds, see config.toml [guidance].
        # short_score: 1-character reads (a stray "5" from room clutter) need this much confidence.
        self.min_score, self.retry_below, self.locations, self.guide = min_score, retry_below, locations, guide or {}
        self.short_score = short_score
        self.unclear = 0  # boxes the detector found but couldn't read confidently (first pass of last read)
        p = {"Det.limit_type": "max", "Det.limit_side_len": max_side, "Global.log_level": "error"}
        self.engine = RapidOCR(params=p)
        # Default PP-OCRv6 has no Devanagari. This 8 MB PP-OCRv5 model reads Hindi and English;
        # only its recognizer is used (detection is shared with self.engine).
        self.hindi = RapidOCR(params={**p, "Rec.lang_type": LangRec.DEVANAGARI, "Rec.model_type": ModelType.MOBILE,
                                      "Rec.ocr_version": OCRVersion.PPOCRV5}) if hindi else None

    def read(self, frame, retry_below=None):
        """BGR frame -> Lines in reading order (top-to-bottom, left-to-right).

        Fast path is one OCR pass. Only when it finds nothing, or is unsure (mean score
        < retry_below), it retries on brightened / sharpened / 90-degree-rotated copies and
        keeps the most confident result. Measured on hard synthetic frames: 0.949 -> 0.976
        character accuracy, median latency unchanged (retries only cost time on bad frames).
        """
        retry_below = self.retry_below if retry_below is None else retry_below
        best, rot = self._pass(frame), None
        self.unclear = self._dropped
        if not best or np.mean([l.score for l in best]) < retry_below:
            for r, img in ((None, enhance(frame)), (None, sharpen(frame)),
                           *((r, cv2.rotate(frame, r)) for r in ROTATIONS)):
                lines = self._pass(img)
                if confidence(lines) > confidence(best):
                    best, rot = lines, r
        best = reading_order(best)  # rotated passes: ordered in rotated coords, i.e. along the text
        if rot is not None:
            for l in best:
                l.box = unrotate(l.box, rot, frame.shape)
        return best

    def describe(self, frame, hints=True):
        """The one call for the app: frame -> Result (speech, lines, aiming hints, hazard flag)."""
        t = time.perf_counter()
        lines = self.read(frame)
        res = Result(to_speech(lines, frame.shape[1] if self.locations else None), lines,
                     guidance(frame, lines, self.unclear, **self.guide) if hints else [],
                     any(l.is_hazard for l in lines))
        # no text content at INFO: what a blind user reads (letters, medicine labels) is private
        log.info("read %d lines in %.0f ms, hazard=%s, hints=%s",
                 len(lines), (time.perf_counter() - t) * 1000, res.hazard, res.hints)
        return res

    def _pass(self, img):
        # RapidOCR remembers use_det/use_cls/use_rec between calls: always pass all three.
        if self.hindi:
            reads = self._read_bilingual(img)
        else:
            r = self.engine(img, use_det=True, use_cls=True, use_rec=True)
            reads = [] if r.txts is None else zip(r.txts, r.scores, r.boxes)
        reads = [(t, s, b) for t, s, b in reads if any(c.isalnum() for c in t)]  # drop punctuation-only junk
        self._dropped = sum(s < self.min_score for _, s, _ in reads)
        return [Line(t.strip(), float(s), np.asarray(b)) for t, s, b in reads if s >= self.min_score
                and (s >= self.short_score or sum(c.isalnum() for c in t) >= 2)]

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
        """Plain sentence, no locations or hints (use describe() in the app)."""
        return to_speech(self.read(frame))


def crop_box(img, box):
    """Perspective-crop a detected 4-point box; tall crops are turned sideways (vertical text)."""
    pts = np.asarray(box, np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    c = cv2.warpPerspective(img, M, (max(w, 1), max(h, 1)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.rot90(c).copy() if h >= 1.5 * w else c


def unrotate(box, rot, shape):
    """Box points from a cv2.rotate(frame, rot) image -> points in the original frame of `shape`."""
    h, w = shape[:2]
    x, y = box[:, 0], box[:, 1]
    return np.stack([y, h - 1 - x], 1) if rot == cv2.ROTATE_90_CLOCKWISE else np.stack([w - 1 - y, x], 1)


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


def where(box, width):
    """Box center -> horizontal third of the frame (the camera faces where the head faces)."""
    cx = box[:, 0].mean() / width
    return "on your left" if cx < 1 / 3 else "on your right" if cx > 2 / 3 else "ahead"


def to_speech(lines, width=None):
    """Lines -> one sentence for TTS. Hazard text is spoken first. With the frame width, each run
    of lines in the same third gets one location tag: "Warning: CAUTION, ahead. Exit, on your left." """
    hazards = [l for l in lines if l.is_hazard]
    rest = [l for l in lines if not l.is_hazard]
    parts = [tagged(group, width) for group in (hazards, rest) if group]
    if hazards:
        parts[0] = "Warning: " + parts[0]
    return ". ".join(parts) + "." if parts else ""


def tagged(lines, width):
    if not width:
        return ". ".join(l.text for l in lines)
    groups = []  # [location, [texts]]
    for l in lines:
        loc = where(l.box, width)
        if groups and groups[-1][0] == loc:
            groups[-1][1].append(l.text)
        else:
            groups.append([loc, [l.text]])
    return ". ".join(f"{'. '.join(texts)}, {loc}" for loc, texts in groups)


def guidance(frame, lines, unclear=0, dark=40, blur=60, flat=12, edge=0.01, small=0.03, unsure=0.85):
    """Aiming hints for someone who can't see the frame. [] when the read looks good.

    ponytail: fixed heuristics on brightness / Laplacian blur / box geometry, tuned on synthetic frames.
    Tune dark/blur/edge/small in config.toml on real glasses footage (blur depends on resolution).
    """
    h, w = frame.shape[:2]
    if not lines:
        if frame.mean() < dark:
            return ["Too dark"]
        if frame.std() > flat and sharpness(frame) < blur:  # flat = blank wall, not blur
            return ["Image blurry, hold still"]
        return ["Text unclear, move closer or hold still"] if unclear else []
    boxes = np.concatenate([l.box for l in lines])
    hints = [f"Text cut off {side}, turn {turn}" for cut, side, turn in (
        (boxes[:, 0].min() < edge * w, "on the left", "left"),
        (boxes[:, 0].max() > (1 - edge) * w, "on the right", "right"),
        (boxes[:, 1].min() < edge * h, "at the top", "up"),
        (boxes[:, 1].max() > (1 - edge) * h, "at the bottom", "down")) if cut]
    if np.mean([l.score for l in lines]) < unsure:
        tiny = np.median([np.ptp(l.box[:, 1]) for l in lines]) < small * h
        hints.append("Move closer" if tiny else "Hold still")
    return hints


def norm(s):
    """Compare texts ignoring case, spaces and punctuation (keeps Devanagari vowel signs)."""
    s = s.lower().replace("warning:", "")
    return "".join(ch for ch in s if not (ch.isspace() or unicodedata.category(ch)[0] in "PS"))


class Announcer:
    """Auto mode: say each sign once, not every frame. Hazards are repeated sooner."""

    def __init__(self, repeat_after=30, hazard_repeat=10, similar=0.8, clock=time.monotonic):
        self.repeat_after, self.hazard_repeat, self.similar, self.clock = repeat_after, hazard_repeat, similar, clock
        self.recent = {}  # normalized text -> when it was last spoken

    def filter(self, result):
        """Result -> speech to say now, or None if it was said recently (fuzzy: OCR jitters frame to frame)."""
        if not result.speech:
            return None
        now, key = self.clock(), norm(" ".join(l.text for l in result.lines))  # not speech: locations shift
        wait = self.hazard_repeat if result.hazard else self.repeat_after
        if any(now - t < wait and difflib.SequenceMatcher(None, k, key).ratio() >= self.similar
               for k, t in self.recent.items()):
            return None
        self.recent = {k: t for k, t in self.recent.items() if now - t < self.repeat_after}
        self.recent[key] = now
        return result.speech


def sharpest(cap, burst=8):
    """Head-worn camera shakes: grab a burst (also flushes stale buffered frames), keep the sharpest."""
    frames = [f for ok, f in (cap.read() for _ in range(burst)) if ok]
    return max(frames[3:] or frames, key=sharpness) if frames else None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="image file to read")
    ap.add_argument("--camera", type=int, metavar="N", help="camera index, ENTER = read")
    ap.add_argument("--auto", type=int, metavar="N", help="camera index, hands-free mode")
    ap.add_argument("--speak", action="store_true", help="speak through the speaker (else only print)")
    ap.add_argument("--config", default=Path(__file__).with_name("config.toml"), type=Path)
    a = ap.parse_args(argv)
    if a.image is None and a.camera is None and a.auto is None:
        ap.error("give an image, --camera N or --auto N")
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print Devanagari: don't crash before speaking
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = tomllib.loads(a.config.read_text(encoding="utf-8")) if a.config.exists() else {}

    speaker = None
    if a.speak:
        from speech import Speaker
        speaker = Speaker(**cfg.get("speaker", {}))

    def say(text, wait=True):
        print(text, flush=True)
        if speaker:
            speaker.say(text, wait=wait)

    try:
        ocr = OCR(**cfg.get("ocr", {}), guide=cfg.get("guidance", {}))
    except Exception:
        log.exception("OCR models failed to load")
        if speaker:
            speaker.beep("error", wait=True)
        raise

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"Cannot read image: {a.image}")
        say(ocr.describe(frame).announcement)
        return

    cam = cfg.get("camera", {})
    cap = cv2.VideoCapture(a.camera if a.auto is None else a.auto)  # no preview window: glasses have no screen
    if not cap.isOpened():
        say("Camera not working.")
        sys.exit(1)  # under systemd: restart and try again
    try:
        if a.auto is not None:
            announcer = Announcer(**cfg.get("announcer", {}))
            while True:
                frame = sharpest(cap, cam.get("burst", 8))
                if frame is None:
                    say("Camera not working.")
                    time.sleep(2)
                    continue
                text = announcer.filter(ocr.describe(frame, hints=False))  # hints off: too chatty hands-free
                if text:
                    say(text)  # wait: don't talk over ourselves
        while input("ENTER = read, q = quit > ").strip().lower() != "q":
            if speaker:
                speaker.beep("start")  # reading takes ~1 s: confirm the press right away
            frame = sharpest(cap, cam.get("burst", 8))
            say(ocr.describe(frame).announcement if frame is not None else "Camera not working.")
    finally:
        cap.release()


if __name__ == "__main__":
    main(sys.argv[1:])
