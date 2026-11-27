# AuraNav OCR

Offline sign reader for the AuraNav smart glasses: **camera frame → text lines → one sentence for text-to-speech**.
Runs fully on-device on a Raspberry Pi 5 using [RapidOCR](https://github.com/RapidAI/RapidOCR) (PaddleOCR models on ONNX Runtime). No internet needed.

Reads **English**, **Hindi (Devanagari)**, and **bilingual** signs (e.g. `निकास EXIT`).

## Quick start

```bash
pip install -r requirements.txt

python ocr.py sign.jpg       # read an image file
python ocr.py --camera 0     # live camera: ENTER = read, q = quit
python test_ocr.py           # self-check
python bench.py              # hard-case accuracy/latency benchmark
```

Models download automatically on first run, then everything works offline.

## Use from code

```python
import cv2
from ocr import OCR

ocr = OCR()                        # OCR(hindi=False) for English-only (faster)
frame = cv2.imread("sign.jpg")
print(ocr.speak(frame))            # "Warning: CAUTION. Wet Floor."
for line in ocr.read(frame):       # Line(text, score, box) in reading order
    print(line.text, line.score, line.is_hazard)
```

## How it works

1. **Detect** text boxes once (detector input capped at `max_side=960` px so wide camera frames stay fast on the Pi).
2. **Recognize** every box with both the English and the Devanagari model (batched). If the Hindi read contains Devanagari it wins, otherwise the English read is used. Picking by confidence score doesn't work: on `निकास EXIT` the English model confidently reads just `EXIT`.
3. **Retry only when unsure.** If nothing is found or the mean score is below `0.85`, the frame is retried brightened (gamma + denoise + CLAHE), sharpened, and rotated ±90°; the most confident pass wins. Good frames pay nothing extra.
4. **Reading order:** boxes are grouped into rows, then sorted left-to-right.
5. **Speech:** hazard words (`caution`, `danger`, `warning`, `fire`, `सावधान`, `खतरा`, …) are spoken first with a `Warning:` prefix. `STOP` alone counts as a hazard; `Bus Stop` does not.

In camera mode the glasses grab a burst of 8 frames and OCR only the sharpest one (Laplacian variance), which handles head shake and flushes stale buffered frames. There is no preview window, since the glasses have no screen.

## Tuning knobs

| Parameter | Default | Effect |
|---|---|---|
| `OCR(min_score=...)` | `0.6` | Raise if it reads noise; lower for blurry or far signs |
| `OCR(max_side=...)` | `960` | Bigger finds smaller/farther text but is slower |
| `OCR(hindi=...)` | `True` | `False` skips the Devanagari model (English-only, faster) |
| `read(retry_below=...)` | `0.85` | Confidence below which the retry passes run |
| Camera burst size | `8` frames | More frames = steadier pick, slower |

Hindi mode costs about +8% time on English frames and is about 4x faster on Hindi frames than English-only mode, which misreads them and burns its retry passes.

## Files

| File | Purpose |
|---|---|
| `ocr.py` | The OCR module and CLI |
| `test_ocr.py` | Assert-based self-check (reading order, hazards, dark/rotated frames, Hindi) |
| `bench.py` | Synthetic benchmark: 14 hard cases (dark, glare, motion blur, tilt, tiny text, JPEG noise...) × English / Hindi / bilingual |
| `render_sign.py` | Renders Devanagari signs through headless Edge/Chrome (Pillow can't shape Devanagari without libraqm) |
| `bench_signs/` | Cached Hindi sign renders, so tests and the benchmark run without a browser |

## Raspberry Pi notes

- For the benchmark's English signs, install fonts: `sudo apt install fonts-dejavu fonts-liberation`.
- New Hindi test signs need Chromium to render; the cached ones in `bench_signs/` don't.
