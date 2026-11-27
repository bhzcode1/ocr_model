"""Render a sign with correct Devanagari shaping (matras, conjuncts) via headless Edge/Chrome.

Pillow can't shape Devanagari without libraqm, so Hindi test images go through a browser.
Renders are cached in bench_signs/, so the browser runs once per unique sign.
"""
import hashlib, html, os, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np

CACHE = Path(__file__).with_name("bench_signs")
BROWSER = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                            shutil.which("chromium"), shutil.which("chromium-browser"))
                if p and os.path.exists(p)), None)


def render_sign(text, size=60, fg=(20, 20, 20), bg=(235, 235, 235), font="Nirmala UI"):
    """-> BGR image of `text` on a `bg` rectangle with ~size/2 padding. fg/bg are BGR."""
    key = hashlib.md5(repr((text, size, fg, bg, font)).encode()).hexdigest()[:12]
    png = CACHE / f"{key}.png"
    if not png.exists():
        if not BROWSER:
            raise RuntimeError("Need Edge/Chrome/Chromium to render Devanagari signs")
        CACHE.mkdir(exist_ok=True)
        css = lambda c: f"rgb({c[2]},{c[1]},{c[0]})"
        page = CACHE / f"{key}.html"
        page.write_text(f"<html><body style='margin:0;background:{css(bg)}'><span style='font:{size}px \"{font}\";"
                        f"color:{css(fg)};white-space:nowrap;padding:{size // 2}px;display:inline-block'>"
                        f"{html.escape(text)}</span></body></html>", encoding="utf-8")
        # own profile dir, else a running browser window swallows the launch and nothing is written
        profile = Path(tempfile.gettempdir()) / "auranav_render_profile"
        subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
                        f"--user-data-dir={profile}", f"--screenshot={png}", "--window-size=1600,400",
                        page.as_uri()], capture_output=True, timeout=60)
        page.unlink()
    img = cv2.imread(str(png))
    # crop the page to the ink plus padding
    ys, xs = np.where(np.abs(img.astype(int) - bg).sum(2) > np.abs(np.subtract(fg, bg)).sum() / 2)
    pad = size // 2
    return img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad].copy()


if __name__ == "__main__":
    cv2.imwrite(str(CACHE / "_demo.png"), render_sign("आपातकालीन निकास EXIT"))
    print("wrote", CACHE / "_demo.png")
