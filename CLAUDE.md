# feverDetection — Project Status

Thermal-vision system that flags feverish animals/humans, driving a laser
pointer at them via an ESP32-controlled pan/tilt servo mount.

## Architecture (two separate devices/repos)

- **thermalCam-Pi** — a *separate project/repo*
  (`~/innovation/thermalCam-Pi`, its own git history), NOT part of this
  repo. Runs on a Raspberry Pi with a USB "thermal-look" camera + MLX90614
  IR point sensor physically attached. Default port `5000`. Owns all
  sensor hardware and temperature-estimation math. Key endpoints:
  - `GET /video_feed?map=<name>` — MJPEG, color-mapped, always 640x480.
  - `GET /pixel_temp?x=<0-1>&y=<0-1>` — estimated °C at a normalized point,
    computed server-side from the true raw sensor frame (linear
    extrapolation from the MLX90614's single center-point reading).
  - `GET /temperature_data`, `/temp_range`, `/system_status`.
  - Known limitation: MLX90614 I2C reads fail ~30-60% of the time on this
    hardware combo (masked by retry + moving-average in
    `mlx90614_robust.py`); `status` fields go `"ERROR"` / `null` on bad
    polls — callers must handle that, not assume a reading is available.

- **This repo (`feverDetection`)** — the inference engine. Runs on a
  Raspberry Pi 5 or Jetson, with **no camera or MLX90614 attached**. Pulls
  thermalCam-Pi's video feed over HTTP, runs YOLO, and asks thermalCam-Pi
  for temperature at each detection. See `RPI/README.md` for the full
  three-stage pipeline (video → species detection → fever verdict) and env
  var reference.
  - `RPI/feverDetection.py` — main Flask app (port `8080`).
  - `RPI/fever_estimator.py` — species temp ranges + fever verdict, driven
    by rate-limited `/pixel_temp` queries.
  - `RPI/thermalcam_client.py` — HTTP client for thermalCam-Pi's API.
  - `RPI/pointer_mapper.py` — normalizes detection coords into the fixed
    640x480 space the ESP32 firmware (`esp32/thermalPointer/`) assumes,
    rate-limits pointer moves.
  - `RPI/device_scanner.py` — LAN scan to auto-discover the ESP32 by MAC
    vendor prefix.
  - `JETSON/feverDetection_jetson.py` — thin launcher, not a separate
    implementation. Adds `../RPI` to `sys.path` and runs
    `RPI/feverDetection.py` unmodified via `runpy` — one shared trained
    model (Ultralytics), one shared codebase, across both devices.
    Ultralytics uses CUDA automatically when available; the only
    per-device difference is `YOLO_WEIGHTS` (plain `.pt` on the Pi vs. a
    TensorRT `.engine` export on Jetson). Serves the same routes on the
    same port `8080` as the Pi.
  - `JETSON/v2_thermalNet.py` — older, self-contained Jetson script (own
    crude local calibration, no species-specific fever margins). Kept
    separate/untouched per explicit instruction — do not merge or retire
    without being asked.

## History / why it looks like this

- Originally (`v4`/`v5`/`v6_thermalCam.py`, all deleted) this repo's Pi
  code did everything on one device: local camera capture, local MLX90614
  reads, YOLO detection, fever math, and ESP32 control, all bundled in one
  Flask app. That version is gone.
- First revamp attempt (superseded): split into detection stage (species
  only) + a local `fever_estimator.py` that itself read a local
  `mlx90614_reader.py` and calibrated gray-pixel-intensity → °C via an
  EMA-adjusted scale — this assumed the MLX90614 was attached to *this*
  device.
- Corrected: the MLX90614 is attached to thermalCam-Pi, a separate device
  that already exposes ready-made per-point temperature estimates via
  `/pixel_temp`. `fever_estimator.py` and `feverDetection.py` were rewired
  to be a pure HTTP client of that API — no local calibration math, no
  local camera capture (video is pulled from thermalCam-Pi's `/video_feed`
  via `cv2.VideoCapture(url)`). The local `mlx90614_reader.py` duplicate
  was deleted as dead code.
- First Jetson script (superseded): a `jetson_inference.detectNet`-based
  rewrite, mirroring the Pi pipeline but with its own GPU detection stack.
  Rejected once the user confirmed they want **one shared trained model**
  across devices — `jetson_inference` expects its own model/ONNX format,
  which would've meant training/exporting two different models. Replaced
  with the `runpy`-based launcher described above.

## UI (`RPI/static/feverDetection.html`)

Also served as-is on Jetson via the launcher above — one shared UI, no
separate Jetson template.

- Design reference: `~/innovation/thermalCam-Pi/templates/thermal_index.html`
  (a *different* project's page, for its physical camera/MLX90614 viewer).
  The user pointed at it explicitly as "simple and great, looks very
  professional" — full-bleed video, minimal floating translucent glass
  chrome (`rgba(0,0,0,0.5)` + `backdrop-filter: blur(8px)` + thin
  `rgba(255,255,255,0.1)` borders), Apple system font, restrained
  blue/green/red accents, 0.15–0.2s transitions. `feverDetection.html` was
  rebuilt to match that exact visual language — same CSS variable names,
  same pill/panel/modal shapes — rather than reusing an ad-hoc style.
- Iteration history (each rejected before the next, in order):
  1. Boxed sidebar+content dashboard (v6-era) — plain, no design pass.
  2. Card-grid dashboard (dark, green accent) — rejected as "still ugly":
     actually rendering it in headless Chrome (see below) showed the
     video card as a huge dead black rectangle when idle, side panel not
     filling the height, weak type hierarchy.
  3. "Thermal HUD" rebuild (amber/cyan glow, corner brackets, scanlines,
     monospace) — rejected as "too flat/generic" was the stated complaint
     going in, but the *style itself* wasn't what the user wanted once
     they pointed at the thermalCam-Pi reference instead.
  4. Current: full-bleed video + floating glass pills/panel/modal, matching
     thermalCam-Pi's reference page directly.
- Bugs found and fixed post-rebuild (all via actually rendering the page in
  headless Chrome + reading pixels, not guessing):
  - Top bar had 3 status pills (Server/Pointer/thermalCam-Pi) that
    duplicated the detailed Devices panel below → collapsed to one
    aggregate "All Systems Active"/"Attention Needed" pill.
  - `.panel { overflow: hidden; max-height: calc(100vh - 76px) }` silently
    clipped the bottom of the settings panel (Y Max slider, Apply Range
    button) on short viewports with no way to reach it → changed to
    `overflow-y: auto` (+ `-webkit-overflow-scrolling: touch`,
    `overscroll-behavior: contain`).
  - That scrollbar then rendered in the browser's default (light/system)
    color, clashing with the dark glass panel → added
    `scrollbar-width`/`scrollbar-color` (Firefox) and `::-webkit-scrollbar*`
    (Chromium/WebKit) rules matching the panel's translucent-white
    language. **Not visually confirmed** — headless Chrome in this sandbox
    doesn't render the scrollbar thumb at all (likely an overlay-scrollbar
    quirk of this specific environment), so the fix is correct-per-spec
    but unverified pixel-for-pixel. Worth a real-browser check.
- Verification method used throughout: `google-chrome --headless
  --disable-gpu --no-sandbox --virtual-time-budget=6000
  --run-all-compositor-stages-before-draw --screenshot=...` against the
  actual running Flask server (plus a `sed`-toggled copy of the file with
  `.visible` classes forced on, to screenshot the settings panel / detections
  modal in their open states) — then reading the PNG back in. Plain
  `--screenshot` without `--virtual-time-budget` is a trap: it captures
  before async `fetch()`/`setInterval` calls resolve, showing permanently
  stale placeholder text even though the app is working fine.

## Current state (as of this note)

Done:
- `feverDetection.py` + `fever_estimator.py` + `pointer_mapper.py` +
  `thermalcam_client.py` + `static/feverDetection.html` written, all
  compile cleanly (`python3 -m py_compile`).
- Old `v6_thermalCam.py`, `static/index.html`, `static/v3_Index.html`
  removed.
- Top-level `README.md` and `RPI/README.md` updated to describe the
  two-device split.

- `JETSON/feverDetection_jetson.py` is now a thin `runpy` launcher of
  `RPI/feverDetection.py` (no `jetson_inference` dependency). Smoke-tested
  on this dev machine (`timeout 3 python3 feverDetection_jetson.py`) — it
  imports cleanly, resolves `RPI/static/` correctly regardless of launch
  directory, and serves Flask on port 8080. All of `flask`/`cv2`/
  `requests`/`ultralytics`/`numpy` are installed in this dev environment.

Not done / untested:
- **No real hardware smoke test yet** — never run against a live
  thermalCam-Pi instance, real ESP32, real YOLO weights, or actual Jetson
  hardware with CUDA/TensorRT. Only verified locally that the process
  starts and serves without crashing.
- UI has been verified only via headless Chrome screenshots on this dev
  machine, never on the actual device's display/browser. Specifically
  unconfirmed: the settings-panel scrollbar color fix (see UI section
  above), and general look/touch behavior on whatever screen the
  Pi/Jetson is actually hooked up to.
- `THERMALCAM_HOST` now defaults to `thermalcam.local` — confirmed correct
  via a parallel remote commit (see "Remote divergence" below), which
  independently hardcoded that same hostname. Previously defaulted to the
  wrong guess `pi4.local`.

## Remote divergence (2026-08-09/10) — resolved via merge

While this session was rebuilding `RPI/`, the same user was independently
pushing directly to `origin/main` from elsewhere: they moved the
Raspberry-Pi camera/MLX90614 source code out of this repo entirely into
the standalone `thermalCam-Pi` repo (deleting a `ThermalCam Pi/` folder
that had briefly lived here), and separately rewrote `JETSON/v2_thermalNet.py`
into its own self-contained `jetson_inference` script that also talks to
thermalCam-Pi's real API (`/video_feed`, `/pixel_temp`) — convergent with
this session's approach, but implemented independently with no shared code,
plus its own `/detections` endpoint on port 8081.

Resolved by merging `origin/main` into `main` (commit history has the
details) with explicit choices, confirmed with the user first:
- Kept `RPI/feverDetection.py` and friends (different component than what
  was moved to `thermalCam-Pi` — this is the inference/pointer-control
  engine, not camera+sensor source code).
- Discarded the remote's `v2_thermalNet.py` rewrite entirely — standardized
  on `JETSON/feverDetection_jetson.py` (the `runpy` launcher, one shared
  model/codebase) as the only Jetson path. `JETSON/thermalNet.py` and
  `v2_thermalNet.py` are both gone now.
- **Caught and fixed a silent-breakage risk**: `RPI/device_scanner.py` was
  untouched locally but deleted upstream (as part of deleting the old
  `RPI/` dir); a raw merge auto-deletes files that are "unmodified on our
  side, deleted on theirs" with **no conflict marker** — but
  `feverDetection.py` still imports it. Had to explicitly
  `git checkout main -- RPI/device_scanner.py` after resolving the real
  conflicts, since git won't flag this kind of loss on its own. Worth
  re-checking with `git diff HEAD` after any future merge in this repo.
