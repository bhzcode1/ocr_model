"""Hard-case OCR benchmark: synthetic signs in camera-like frames. Run: python bench.py"""
import difflib, glob, random, time, unicodedata
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

# Windows fonts, else whatever TTFs the Pi has (sudo apt install fonts-dejavu fonts-liberation)
FONTS = [f for n in ("arial", "arialbd", "verdana", "times", "impact", "calibri")
         for f in glob.glob(rf"C:\Windows\Fonts\{n}.ttf")] or sorted(glob.glob("/usr/share/fonts/truetype/*/*.ttf"))[:6]
TEXTS = ["EXIT", "Platform 3", "CAUTION Wet Floor", "Room 204", "PUSH", "Pharmacy Open 24 Hours",
         "No Entry", "Bus Stop", "Ticket Counter", "Restrooms", "Gate B12", "Keep Left",
         "Main Street", "Emergency Exit", "Lift to Floors 1 5", "Danger High Voltage"]
HINDI = ["निकास", "सावधान", "प्रवेश निषेध", "बस स्टॉप", "शौचालय", "खतरा", "प्लेटफार्म 3", "टिकट घर",
         "दवाखाना", "आपातकालीन निकास", "सावधान फर्श गीला है", "पूछताछ", "महिला डिब्बा", "धक्का दें",
         "बाहर जाने का रास्ता", "मुख्य सड़क"]
BILINGUAL = ["निकास EXIT", "सावधान CAUTION", "शौचालय Toilet", "प्लेटफार्म 3 Platform 3", "पूछताछ Enquiry",
             "खतरा DANGER", "टिकट घर Ticket Counter", "प्रवेश निषेध No Entry"]


def pil_sign(text, size, fg, bg, font):
    f = ImageFont.truetype(font or FONTS[0], size)
    l, t, r, b = f.getbbox(text)
    sign = Image.new("RGB", (r - l + size, b - t + size), bg[::-1])
    ImageDraw.Draw(sign).text((size // 2 - l, size // 2 - t), text, font=f, fill=fg[::-1])
    return cv2.cvtColor(np.array(sign), cv2.COLOR_RGB2BGR)


def frame(text, rng, size=60, fg=(20, 20, 20), bg=(235, 235, 235), font=None):
    W, H = 1280, 720
    base = rng.integers(60, 200, (H // 8, W // 8, 3), np.uint8)
    img = cv2.resize(cv2.GaussianBlur(base, (5, 5), 0), (W, H))  # cluttered background
    if text.isascii():
        sign = pil_sign(text, size, fg, bg, font)
    else:  # Devanagari needs real shaping: browser render (font choice ignored, Nirmala UI)
        from render_sign import render_sign
        sign = render_sign(text, size, fg, bg)
    sh, sw = sign.shape[:2]
    x, y = rng.integers(0, max(1, W - sw)), rng.integers(0, max(1, H - sh))
    sw, sh = min(sw, W - x), min(sh, H - y)
    img[y:y + sh, x:x + sw] = sign[:sh, :sw]
    return img


def rot(img, deg):  # expanded canvas so the sign never leaves the frame
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def persp(img, k):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * k, h * k], [w, 0], [w, h], [w * k, h * (1 - k)]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


def motion(img, n):
    k = np.zeros((n, n), np.float32); k[n // 2] = 1 / n
    return cv2.filter2D(img, -1, k)


def noise(img, s, rng):
    return np.clip(img + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def glare(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    g = np.exp(-(((xx - w * .45) ** 2 + (yy - h * .45) ** 2) / (2 * (w * .25) ** 2)))[..., None]
    return np.clip(img * (1 - .55 * g) + 255 * .55 * g, 0, 255).astype(np.uint8)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)


CASES = {
    "very_dark":   lambda t, r, f: noise((frame(t, r, font=f) * .05).astype(np.uint8), 3, r),
    "low_contrast":lambda t, r, f: frame(t, r, fg=(128, 128, 128), bg=(145, 145, 145), font=f),
    "glare":       lambda t, r, f: glare(glare(frame(t, r, fg=(110, 110, 110), bg=(190, 190, 190), font=f))),
    "motion_blur": lambda t, r, f: motion(frame(t, r, font=f), 25),
    "mild_motion": lambda t, r, f: motion(frame(t, r, font=f), 13),
    "defocus":     lambda t, r, f: cv2.GaussianBlur(frame(t, r, font=f), (0, 0), 4),
    "tiny_far":    lambda t, r, f: frame(t, r, size=11, font=f),
    "small_blur":  lambda t, r, f: cv2.GaussianBlur(frame(t, r, size=16, font=f), (0, 0), 1.2),
    "tilt_40":     lambda t, r, f: rot(frame(t, r, font=f), 40),
    "tilt_90":     lambda t, r, f: rot(frame(t, r, font=f), 90),
    "tilt_270":    lambda t, r, f: rot(frame(t, r, font=f), -90),
    "perspective": lambda t, r, f: persp(frame(t, r, font=f), .42),
    "dark_blur":   lambda t, r, f: noise(motion((frame(t, r, font=f) * .1).astype(np.uint8), 11), 6, r),
    "jpeg_noise":  lambda t, r, f: jpeg(noise(frame(t, r, size=24, font=f), 35, r), 15),
}


def dataset(texts=TEXTS, n=6, seed=0):
    rng, pr = np.random.default_rng(seed), random.Random(seed)
    return [(c, t, fn(t, rng, pr.choice(FONTS))) for c, fn in CASES.items() for t in pr.sample(texts, n)]


def norm(s):
    s = s.lower().replace("warning:", "")
    # drop spaces/punctuation only; isalnum() would also drop Devanagari vowel signs (matras)
    return "".join(ch for ch in s if not (ch.isspace() or unicodedata.category(ch)[0] in "PS"))


def score(read_fn, data):
    by, ms = {}, []
    for c, truth, img in data:
        t = time.perf_counter(); out = read_fn(img); ms.append((time.perf_counter() - t) * 1000)
        by.setdefault(c, []).append(difflib.SequenceMatcher(None, norm(truth), norm(out)).ratio())
    return {c: np.mean(v) for c, v in by.items()}, np.median(ms)


def report(name, res):
    by, ms = res
    print(f"{name:22s} mean={np.mean(list(by.values())):.3f} med={ms:5.0f}ms  " +
          " ".join(f"{c[:6]}={v:.2f}" for c, v in by.items()), flush=True)


if __name__ == "__main__":
    from ocr import OCR
    for hindi in (False, True):
        ocr = OCR(hindi=hindi)
        for name, texts in (("english", TEXTS), ("hindi", HINDI), ("bilingual", BILINGUAL)):
            data = dataset(texts)
            ocr.read(data[0][2])  # warm-up
            report(f"{name} hindi={hindi}", score(lambda im: " ".join(l.text for l in ocr.read(im)), data))
