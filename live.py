"""Live webcam viewer for testing: see what the OCR reads, hear it spoken.

    python live.py                  # opens http://localhost:8000 in the browser
    python live.py --camera 1 --no-speak
    python live.py --host 0.0.0.0   # watch from a phone on the same Wi-Fi (exposes the camera to that network!)

The page is for developers and sighted helpers; the blind user only hears the voice.
Hands-free: each new sign is spoken once (Announcer). "Read now" button / Space key: speak everything + aiming hints.
"""
import argparse
import collections
import json
import logging
import threading
import time
import tomllib
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from ocr import OCR, Announcer, sharpness, where

log = logging.getLogger("auranav.live")


class Live:
    def __init__(self, cap, ocr, speaker, announcer):
        self.cap, self.ocr, self.speaker, self.announcer = cap, ocr, speaker, announcer
        self.frames, self.lock = collections.deque(maxlen=5), threading.Lock()
        self.state = {"lines": [], "speech": "", "hints": [], "ms": 0, "spoken": ""}
        self.read_now = threading.Event()
        self.muted, self.camera_ok = speaker is None, True

    def capture(self):
        while True:
            ok, frame = self.cap.read()
            self.camera_ok = ok
            if ok:
                with self.lock:
                    self.frames.append(frame)
            else:
                time.sleep(0.5)

    def recognize(self):
        while True:
            with self.lock:
                frames = list(self.frames)
            if not frames:
                time.sleep(0.1)
                continue
            frame = max(frames, key=sharpness)  # head shake: sharpest of the last few frames
            try:
                self.step(frame)
            except Exception:  # keep the view alive: one bad frame mustn't kill the thread
                log.exception("read failed")
                time.sleep(1)

    def step(self, frame):
        t = time.perf_counter()
        res = self.ocr.describe(frame)
        new = self.announcer.filter(res)  # always: remembers what was said
        text = res.announcement if self.read_now.is_set() else new
        self.read_now.clear()
        w = frame.shape[1]
        self.state = {"lines": [{"text": l.text, "score": round(l.score, 2), "where": where(l.box, w),
                                 "hazard": l.is_hazard, "box": l.box.astype(int).tolist()} for l in res.lines],
                      "speech": res.speech, "hints": res.hints, "ms": round((time.perf_counter() - t) * 1000),
                      "spoken": text or self.state["spoken"]}
        if text and self.speaker and not self.muted:
            self.speaker.say(text, wait=True)  # don't talk over ourselves; video keeps running meanwhile

    def jpeg(self):
        """Latest frame with numbered boxes (numbers match the list on the page; amber = hazard)."""
        f = self.frames[-1].copy() if self.frames else np.zeros((480, 640, 3), np.uint8)
        for i, l in enumerate(self.state["lines"], 1):
            pts = np.array(l["box"], np.int32)
            color = (0, 170, 255) if l["hazard"] else (80, 220, 80)
            cv2.polylines(f, [pts], True, color, 2)
            cv2.putText(f, str(i), (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def handler(live):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                s = {**live.state, "muted": live.muted, "can_speak": live.speaker is not None, "camera": live.camera_ok}
                self.send(json.dumps(s, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            elif self.path == "/stream":  # MJPEG: the browser shows it as a live <img>
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        j = live.jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j)
                                         + j + b"\r\n")
                        time.sleep(1 / 15)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # tab closed
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/read":
                live.read_now.set()
            elif self.path == "/mute":
                live.muted = not live.muted
            else:
                return self.send_error(404)
            self.send(b"{}", "application/json")
    return Handler


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AuraNav Live OCR</title>
<style>
  :root { --bg:#111418; --panel:#1b2027; --text:#e8ecf1; --muted:#8b95a3; --ok:#50dc50; --warn:#ffaa00; --line:#2a313b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.45 system-ui, "Nirmala UI", sans-serif; }
  header { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0 auto 0 0; font-weight:600; }
  button { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:8px;
           padding:8px 14px; font:inherit; cursor:pointer; }
  button.primary { background:#2f6fed; border-color:#2f6fed; }
  button:disabled { opacity:.4; cursor:default; }
  main { display:grid; grid-template-columns:minmax(0,3fr) minmax(280px,2fr); gap:16px; padding:16px; }
  @media (max-width:800px) { main { grid-template-columns:1fr; } }
  .video { background:#000; border-radius:10px; overflow:hidden; }
  .video img { display:block; width:100%; height:auto; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:12px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 8px; font-weight:600; }
  #spoken { font-size:22px; min-height:1.45em; }
  #hints:empty::before { content:"None, the read looks good"; color:var(--muted); }
  #hints div { color:var(--warn); }
  ol { margin:0; padding-left:22px; }
  li { padding:3px 0; } li.hazard .t { color:var(--warn); font-weight:600; }
  .meta { color:var(--muted); font-size:13px; }
  #status { font-size:13px; color:var(--muted); }
  #status.bad { color:#ff6b6b; }
</style></head>
<body>
<header>
  <h1>AuraNav Live OCR</h1>
  <span id="status">starting…</span>
  <button id="mute">Mute</button>
  <button id="read" class="primary">Read now (Space)</button>
</header>
<main>
  <div class="video"><img src="/stream" alt="Live camera view with detected text boxes"></div>
  <div>
    <section><h2>Last spoken</h2><div id="spoken" aria-live="polite"></div></section>
    <section><h2>Aiming hints</h2><div id="hints"></div></section>
    <section><h2>Text in view</h2><ol id="lines"></ol><div id="empty" class="meta">No text in view</div></section>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const post = path => fetch(path, {method: "POST"});
$("read").onclick = () => post("/read");
$("mute").onclick = () => post("/mute");
addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); post("/read"); } });
async function tick() {
  try {
    const s = await (await fetch("/state")).json();
    $("spoken").textContent = s.spoken;  // textContent, never innerHTML: sign text is untrusted
    $("hints").replaceChildren(...s.hints.map(h => Object.assign(document.createElement("div"), {textContent: h})));
    $("lines").replaceChildren(...s.lines.map(l => {
      const li = document.createElement("li");
      li.className = l.hazard ? "hazard" : "";
      li.append(Object.assign(document.createElement("span"), {className: "t", textContent: l.text}),
                Object.assign(document.createElement("span"), {className: "meta", textContent: `  ${l.where} · ${l.score}`}));
      return li;
    }));
    $("empty").hidden = s.lines.length > 0;
    $("mute").textContent = s.muted ? "Unmute" : "Mute";
    $("mute").disabled = !s.can_speak;
    $("status").textContent = s.camera ? `${s.ms} ms per read` : "Camera not working";
    $("status").className = s.camera ? "" : "bad";
  } catch { $("status").textContent = "Disconnected"; $("status").className = "bad"; }
  setTimeout(tick, 300);
}
tick();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to view from other devices")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--config", default=Path(__file__).with_name("config.toml"), type=Path)
    a = ap.parse_args()
    cfg = tomllib.loads(a.config.read_text(encoding="utf-8")) if a.config.exists() else {}

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {a.camera}")
    print("loading models...", flush=True)
    speaker = None
    if not a.no_speak:
        from speech import Speaker
        speaker = Speaker(**cfg.get("speaker", {}))
    live = Live(cap, OCR(**cfg.get("ocr", {}), guide=cfg.get("guidance", {})), speaker,
                Announcer(**cfg.get("announcer", {})))
    for target in (live.capture, live.recognize):
        threading.Thread(target=target, daemon=True).start()

    server = ThreadingHTTPServer((a.host, a.port), handler(live))
    server.daemon_threads = True
    url = f"http://localhost:{a.port}"
    print(f"live view: {url}   (Ctrl+C to quit)", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
