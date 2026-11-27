"""Real-world evaluation on the Bharat Scene Text Dataset (BSTD, IIT Jodhpur, Apache-2.0, images CC BY-SA 4.0).

Real photos of Indian streets, shops and signs with word-level ground truth in Hindi and English.
https://github.com/Bhashini-IITJ/BharatSceneTextDataset

    python eval_bstd.py fetch            # stream ~300 test images out of the 17.7 GB zip (range requests)
    python eval_bstd.py run              # score the current config.toml settings
    python eval_bstd.py run --part dev   # half the images; "test" = the other half (tune on dev, report on test)
    python eval_bstd.py run --set ocr:max_side=1280 --set params:Det.model_type=medium   # try settings

Metric: end-to-end word match, ignoring position. Ground-truth words vs words the OCR read, both
normalized (case, punctuation), counted as a multiset. Precision = read words that are right,
recall = ground-truth words found. "Sign-size recall" only counts words at least 3% of the image
height, which is roughly what a wearer would ask about; tiny background text is the rest.
"""
import argparse
import collections
import io
import json
import random
import statistics
import sys
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ZIP_URL = "https://drive.usercontent.google.com/download?id=1S7KUYfB-lQvbu6GtvZxxDPOD5ZZ080M0&export=download&confirm=t"
DATA = Path(__file__).with_name("data") / "bstd"
SCRIPTS = {"english", "hindi"}  # what the glasses read; images with other scripts are skipped
IGNORE = {"unk", "na"}  # illegible / unlabelled words: allowed in an image, not scored


def script(p):
    return p["script_language"].strip().lower()
MAX_SIDE = 1600  # images are stored downscaled: phone photos are bigger than the glasses camera


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so zipfile can pull single members."""

    def __init__(self, url, chunk=1 << 20):
        self.url, self.chunk, self.pos, self.buf_start, self.buf = url, chunk, 0, 0, b""
        self.size = int(self._get(0, 0)[1].split("/")[-1])

    def _get(self, a, b):
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "python"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read(), r.headers["Content-Range"]
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        end = self.pos + n
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            self.buf_start = self.pos
            self.buf = self._get(self.pos, min(self.pos + max(n, self.chunk), self.size) - 1)[0]
        out = self.buf[self.pos - self.buf_start:end - self.buf_start]
        self.pos += len(out)
        return out


def fetch(n, seed=0):
    DATA.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(RangeFile(ZIP_URL))
    names = {i.filename.split("/", 1)[-1]: i for i in z.infolist()}  # drop the top "detection/" folder
    ann_path = DATA / "BSTD_annotations.json"
    if not ann_path.exists():
        member = names["BSTD_v17.57.json"]  # the version the README documents
        print(f"annotations: {member.filename} ({member.file_size / 1e6:.0f} MB)", flush=True)
        ann_path.write_bytes(z.read(member))
    ann = json.loads(ann_path.read_text(encoding="utf-8"))

    def usable(rec):
        langs = {script(p) for p in rec["annotations"].values()} - IGNORE
        return rec["split"] == "test" and langs and langs <= SCRIPTS
    pool = sorted(k for k, r in ann.items() if usable(r))
    with_hi = [k for k in pool if any(script(p) == "hindi" for p in ann[k]["annotations"].values())]
    print(f"test images with only English/Hindi text: {len(pool)} ({len(with_hi)} contain Hindi)", flush=True)
    rng = random.Random(seed)  # half the sample has Hindi, so both scripts get a real score
    pick = rng.sample(with_hi, min(n // 2, len(with_hi)))
    pick += rng.sample(sorted(set(pool) - set(pick)), n - len(pick))
    gt = {}
    for i, key in enumerate(sorted(pick)):
        rec = ann[key]
        out = DATA / "images" / f"{key}.jpg"
        if not out.exists():
            img = cv2.imdecode(np.frombuffer(z.read(names[rec["image_name"]]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            s = MAX_SIDE / max(img.shape[:2])
            scale = min(1.0, s)
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        if scale < 1 else img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            orig_h = img.shape[0]
        else:
            orig_h = None
        gt[key] = {"words": [{"text": p["text"], "script": script(p),
                              "h": float(np.ptp(np.asarray(p["coordinates"], float)[:, 1]))}
                             for p in rec["annotations"].values() if script(p) not in IGNORE],
                   "orig_h": orig_h, "lighting": rec.get("lighting"), "environment": rec.get("environment"),
                   "url": rec.get("url")}
        print(f"\r{i + 1}/{len(pick)} images", end="", flush=True)
    old = json.loads((DATA / "gt.json").read_text(encoding="utf-8")) if (DATA / "gt.json").exists() else {}
    for k, v in gt.items():  # keep orig_h from the first download
        if v["orig_h"] is None and k in old:
            v["orig_h"] = old[k]["orig_h"]
    (DATA / "gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {len(gt)} images to {DATA}")


def words(text):
    from ocr import norm
    return [w for w in (norm(t) for t in text.split()) if w and w != "###"]


def score(ocr, part=None, limit=None, verbose=False):
    gt = json.loads((DATA / "gt.json").read_text(encoding="utf-8"))
    keys = sorted(gt)
    if part:  # deterministic split: tune on dev, report on test
        keys = [k for i, k in enumerate(keys) if (i % 2 == 0) == (part == "dev")]
    keys = keys[:limit]
    c = collections.Counter()
    ms = []
    for k in keys:
        img = cv2.imread(str(DATA / "images" / f"{k}.jpg"))
        t = time.perf_counter()
        lines = ocr.read(img)
        ms.append((time.perf_counter() - t) * 1000)
        pred = collections.Counter(w for l in lines for w in words(l.text))
        g = gt[k]
        big = 0.03 * (g["orig_h"] or img.shape[0])
        truth = [(w, x["script"], x["h"] >= big) for x in g["words"] for w in words(x["text"])]
        left = pred.copy()
        for w, script, is_big in truth:
            hit = left[w] > 0
            left[w] -= hit
            c[f"{script}_gt"] += 1
            c[f"{script}_hit"] += hit
            c["big_gt"] += is_big
            c["big_hit"] += hit and is_big
        c["pred"] += sum(pred.values())
        c["hit"] += sum(min(pred[w], n) for w, n in collections.Counter(w for w, _, _ in truth).items())
        c["gt"] += len(truth)
        if verbose:
            print(f"{k}: {sum(min(pred[w], n) for w, n in collections.Counter(w for w, _, _ in truth).items())}"
                  f"/{len(truth)} words, {ms[-1]:.0f} ms", flush=True)
    p, r = c["hit"] / max(c["pred"], 1), c["hit"] / max(c["gt"], 1)
    return {"images": len(keys), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "english_recall": c["english_hit"] / max(c["english_gt"], 1),
            "hindi_recall": c["hindi_hit"] / max(c["hindi_gt"], 1),
            "sign_size_recall": c["big_hit"] / max(c["big_gt"], 1),
            "median_ms": statistics.median(ms), "p90_ms": float(np.percentile(ms, 90)),
            "words": c["gt"], "hindi_words": c["hindi_gt"], "sign_size_words": c["big_gt"]}


def show(name, s):
    print(f"{name:34s} F1={s['f1']:.3f}  P={s['precision']:.3f}  R={s['recall']:.3f}  "
          f"en={s['english_recall']:.3f}  hi={s['hindi_recall']:.3f}  sign={s['sign_size_recall']:.3f}  "
          f"med={s['median_ms']:.0f}ms  p90={s['p90_ms']:.0f}ms  ({s['images']} imgs)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("-n", type=int, default=300)
    r = sub.add_parser("run")
    r.add_argument("--part", choices=["dev", "test"])
    r.add_argument("--limit", type=int)
    r.add_argument("-v", action="store_true")
    r.add_argument("--set", action="append", default=[], metavar="SECTION:KEY=VALUE",
                   help="override a setting; SECTION is ocr, params or hindi_params")
    r.add_argument("--config", default=Path(__file__).with_name("config.toml"), type=Path)
    a = ap.parse_args()
    if a.cmd == "fetch":
        return fetch(a.n)
    sys.stdout.reconfigure(errors="replace")
    from ocr import OCR
    cfg = tomllib.loads(a.config.read_text(encoding="utf-8")) if a.config.exists() else {}
    o = cfg.setdefault("ocr", {})
    for item in a.set:
        sec, kv = item.split(":", 1)
        k, v = kv.split("=", 1)
        try:
            v = tomllib.loads(f"v = {v}")["v"]  # numbers / booleans; anything else stays a string
        except tomllib.TOMLDecodeError:
            pass
        (o if sec == "ocr" else o.setdefault(sec, {}))[k] = v
    ocr = OCR(**cfg.get("ocr", {}), guide=cfg.get("guidance", {}))
    ocr.read(np.full((64, 64, 3), 255, np.uint8))  # warm-up
    show(" ".join(a.set) or f"config.toml ({a.part or 'all'})", score(ocr, a.part, a.limit, a.v))


if __name__ == "__main__":
    main()
