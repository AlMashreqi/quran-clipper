"""pipeline/render_woven_box.py -- the `woven_box` style renderer.

A square (1:1) reel: the footage sits in a rounded-corner box inset from
the canvas edge, the gap filled solid black. The footage is dimmed and
crossed with a procedurally-generated woven-cloth texture (no external
asset -- generated in Pillow the way fx.py's snow layer is, rather than
shipping an extracted texture image). Captions are plain white Jomhuria
Arabic with a small solid (unblurred) drop shadow, centered over the
footage inside the box. Arabic-only, no English, no verse-number ornament
-- same caption posture as `bars`.

Called by generate.py with the resolved plan; not a standalone CLI.
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

FFMPEG = os.environ.get("QC_FFMPEG") or "ffmpeg"

if not features.check("raqm"):
    sys.exit("FATAL: Pillow lacks RAQM (HarfBuzz+FriBiDi) -- Arabic would be "
             "laid out unjoined, left to right. Use tools/render-venv/bin/python.")

FONT_PATH = os.path.join(FONT_DIR, "Jomhuria-Regular.ttf")

# Fixed house style, deliberately NOT config-exposed: tuned once against
# the reference screenshot (docs/Screenshot_20260827_124044_Instagram.jpg).
BOX_FRAC = 0.90          # video box side, as a fraction of the canvas side
CORNER_FRAC = 0.06       # box corner radius, as a fraction of box width
SHADOW = {"opacity": 0.55, "offset_frac": 0.006, "color": (0, 0, 0)}

# Caption text as a fraction of the box height; shrinks to fit like
# render_default's shared_font_size (one size for every card in the reel).
AR_SIZE_FRAC = 0.115
AR_MAX_WIDTH_FRAC = 0.86     # of box width

# The signature sits inside the footage near the bottom of the box, not in
# the black margin (too thin to hold a legible line).
SIGNATURE_SIZE_FRAC = 0.034     # of box height
SIGNATURE_Y_FRAC = 0.925        # of box height, from the box's own top
SIGNATURE_ALPHA = 0.80

# Woven-cloth texture: two perpendicular sets of hairline threads, each
# jittered and lightly blurred so the weave isn't a perfect grid. Kept
# pale (200-255) so `blend=multiply` only lifts a subtle grain rather than
# darkening the shot -- the actual dimming is a separate, config-driven step.
WEAVE_PITCH_FRAC = 0.012       # thread spacing, fraction of box width
WEAVE_DARK = 205               # darkest thread value (0-255)
WEAVE_JITTER = 1.4             # px, per-thread offset

# Caption fade in/out, seconds -- same feel as render_default's cards.
FADE = 0.4

# Everything intake-side is already held to this by fetch.py's MAX_FPS, so
# the black-canvas background source can be declared at a fixed rate rather
# than reprobing the (now square-cropped) video -- same convention as
# render_bars.py's FPS constant.
FPS = 30

FACE_MODEL = os.path.join(ROOT, "tools", "models",
                          "face_detection_yunet_2023mar.onnx")
FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


# ---------- subject detection (vision) --------------------------------------
# Ported from render_default.py's detect_subject -- kept duplicated rather
# than imported so this renderer stays independently runnable, matching
# render_bloom.py's precedent.

def detect_subject(video_path, info, tmp, samples=5):
    """Median YuNet face box over sampled frames, or None on a weak signal."""
    try:
        import cv2
    except ImportError:
        print("      opencv not installed -- skipping subject detection")
        return None
    if not os.path.exists(FACE_MODEL):
        os.makedirs(os.path.dirname(FACE_MODEL), exist_ok=True)
        print("      fetching face model -> %s" % os.path.basename(FACE_MODEL))
        try:
            run(["curl", "-sL", "-o", FACE_MODEL, FACE_MODEL_URL])
        except subprocess.CalledProcessError:
            print("      face model download failed -- skipping detection")
            return None

    detector = cv2.FaceDetectorYN.create(FACE_MODEL, "",
                                         (info["width"], info["height"]),
                                         score_threshold=0.6)
    tmp_png = os.path.join(tmp, "_probe.png")
    boxes = []
    for k in range(samples):
        t = info["duration"] * (k + 1) / (samples + 1)
        run([FFMPEG, "-y", "-v", "error", "-ss", "%.2f" % t, "-i", video_path,
             "-frames:v", "1", "-update", "1", tmp_png])
        frame = cv2.imread(tmp_png)
        if frame is None:
            continue
        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
            boxes.append((int(x), int(y), int(w), int(h)))
    if len(boxes) * 2 < samples:
        print("      subject detection weak (%d/%d frames) -- ignoring"
              % (len(boxes), samples))
        return None
    med = lambda vals: sorted(vals)[len(vals) // 2]  # noqa: E731
    x, y = med([b[0] for b in boxes]), med([b[1] for b in boxes])
    w, h = med([b[2] for b in boxes]), med([b[3] for b in boxes])
    return {"cx": x + w // 2, "cy": y + h // 2, "w": w, "h": h, "hits": len(boxes)}


def square_crop_filter(info, subject):
    """Crop the source to 1:1, centered on the subject's face (geometric
    center when detection failed). None if the source is already square."""
    w, h = info["width"], info["height"]
    if w == h:
        return None, 0, 0
    side = min(w, h)
    if w > h:
        center = subject["cx"] if subject else w // 2
        x = max(0, min(w - side, center - side // 2))
        return "crop=%d:%d:%d:0" % (side, side, x), x, 0
    center = subject["cy"] if subject else h // 2
    y = max(0, min(h - side, center - side // 2))
    return "crop=%d:%d:0:%d" % (side, side, y), 0, y


def output_side(side):
    """Canvas side for a square reel: short-side-clamp rule from
    render_default.output_size, applied to the one dimension there is."""
    if side < 720:
        s = 720.0 / side
    elif side > 1080:
        s = 1080.0 / side
    else:
        s = 1.0
    return int(round(side * s / 2)) * 2


# ---------- texture + mask ---------------------------------------------------

def woven_texture(w, h, seed=0):
    """A pale hairline weave, pure Pillow -- multiplied over the dimmed
    footage for the fabric look in the reference screenshot. Not tileable
    or animated: one frame, held static like a still overlay."""
    import random
    rnd = random.Random(seed)
    tex = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(tex)
    pitch = max(2, int(round(w * WEAVE_PITCH_FRAC)))
    for x in range(0, w, pitch):
        jx = x + rnd.uniform(-WEAVE_JITTER, WEAVE_JITTER)
        v = WEAVE_DARK + rnd.randrange(0, 255 - WEAVE_DARK)
        d.line([(jx, 0), (jx, h)], fill=v, width=1)
    for y in range(0, h, pitch):
        jy = y + rnd.uniform(-WEAVE_JITTER, WEAVE_JITTER)
        v = WEAVE_DARK + rnd.randrange(0, 255 - WEAVE_DARK)
        d.line([(0, jy), (w, jy)], fill=v, width=1)
    return tex.filter(ImageFilter.GaussianBlur(0.6)).convert("RGB")


_SS = 4     # rounded-corner mask supersampling (Pillow's rounded_rectangle
            # has no AA), same technique as render_bars.draw_pill


def rounded_mask(w, h, radius):
    tile = Image.new("L", (w * _SS, h * _SS), 0)
    ImageDraw.Draw(tile).rounded_rectangle(
        [0, 0, w * _SS - 1, h * _SS - 1], radius=radius * _SS, fill=255)
    return tile.resize((w, h), Image.LANCZOS)


# ---------- type ------------------------------------------------------------

_PROBE = ImageDraw.Draw(Image.new("RGBA", (10, 10)))


def shared_font_size(texts, base, max_width, floor=20):
    """One size fitting every caption -- ported from render_default so the
    Arabic doesn't visibly change size card to card."""
    size = base
    while size > floor:
        font = ImageFont.truetype(FONT_PATH, size)
        if max(_PROBE.textbbox((0, 0), t, font=font)[2] for t in texts) <= max_width:
            return size
        size = int(size * 0.94)
    return size


def render_text_png(text, size, box_w):
    """One line to a tight transparent PNG: white ink over a solid,
    unblurred, offset-down shadow -- a crisper lift than render_default's
    blurred shadow, matching the reference's harder-edged look."""
    font = ImageFont.truetype(FONT_PATH, size)
    bbox = _PROBE.textbbox((0, 0), text, font=font)
    off = max(1, int(round(SHADOW["offset_frac"] * box_w)))
    pad = off + 4
    w = (bbox[2] - bbox[0]) + pad * 2
    h = (bbox[3] - bbox[1]) + pad * 2
    ink = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(ink).text((pad - bbox[0], pad - bbox[1]), text, font=font,
                             fill=(255, 255, 255, 255))
    solid = Image.new("RGBA", (w, h),
                      SHADOW["color"] + (int(SHADOW["opacity"] * 255),))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.paste(solid, (off, off), ink.split()[3])
    return Image.alpha_composite(shadow, ink), w, h


def assert_fits(kind, i, png_w, box_w):
    if png_w > box_w:
        raise SystemExit(
            "%s of card %d is %dpx wide and overflows the %dpx video box"
            % (kind, i + 1, png_w, box_w))


def trim_to_ink(card):
    """Ported verbatim from render_default -- overlay pays only for a
    card's non-transparent pixels; snapped to even coords for yuv420 chroma."""
    box = card.getbbox()
    if box is None:
        return card, 0, 0
    x0, y0 = box[0] - box[0] % 2, box[1] - box[1] % 2
    x1 = min(card.width, box[2] + box[2] % 2)
    y1 = min(card.height, box[3] + box[3] % 2)
    return card.crop((x0, y0, x1, y1)), x0, y0


# ---------- compositing -----------------------------------------------------

def composite(video, geom, texture_path, mask_path, overlays, out_path, dim,
              duration, canvas):
    """Black canvas <- dimmed+textured+masked footage in its box <- caption
    cards (each fading in/out over its own window) <- signature."""
    box_w, box_h, box_x, box_y = geom
    eps = 0.01
    cmd = [FFMPEG, "-y", "-i", video,
           "-loop", "1", "-i", texture_path,
           "-loop", "1", "-i", mask_path]
    for ov in overlays:
        cmd += ["-loop", "1", "-i", ov["path"]]

    parts = [
        "color=c=black:s=%dx%d:r=%s:d=%.3f[canvas]"
        % (canvas["w"], canvas["h"], canvas["fps"], duration),
        "[0:v]scale=%d:%d:flags=lanczos,format=rgb24,"
        "colorchannelmixer=rr=%s:gg=%s:bb=%s[dim]" % (box_w, box_h, dim, dim, dim),
        "[1:v]format=rgb24[tex]",
        "[dim][tex]blend=all_mode=multiply[grain]",
        "[2:v]format=gray[maskg]",
        "[grain][maskg]alphamerge[boxed]",
        "[canvas][boxed]overlay=%d:%d:format=auto[framed0]" % (box_x, box_y),
        # Without this, overlay negotiates a pixel format for [framed0] that
        # the caption loop's rgba fades then mismatch against -- observed as
        # the whole frame washing green. Pin it explicitly before the loop.
        "[framed0]format=rgba[framed]",
    ]
    label = "framed"
    base = 3
    for i, ov in enumerate(overlays):
        if ov.get("full"):
            parts.append("[%d:v]format=rgba[fx%d]" % (i + base, i))
            parts.append("[%s][fx%d]overlay=x=%d:y=%d[ov%d]"
                         % (label, i, ov["x"], ov["y"], i))
        else:
            end = max(ov["start"], ov["end"] - eps)
            fade = min(FADE, max(0.0, (end - ov["start"]) / 2))
            parts.append(
                "[%d:v]format=rgba,"
                "fade=t=in:st=%.3f:d=%.3f:alpha=1,"
                "fade=t=out:st=%.3f:d=%.3f:alpha=1[fx%d]"
                % (i + base, ov["start"], fade, max(0.0, end - fade), fade, i))
            parts.append(
                "[%s][fx%d]overlay=x=%d:y=%d:enable='between(t,%.3f,%.3f)'[ov%d]"
                % (label, i, ov["x"], ov["y"], ov["start"], end, i))
        label = "ov%d" % i
    parts.append("[%s]format=yuv420p[vout]" % label)

    run(cmd + ["-filter_complex", ";".join(parts), "-map", "[vout]",
               "-map", "0:a", "-t", "%.3f" % duration,
               "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", out_path])


# ---------- the style entry point -------------------------------------------

def render(plan):
    cfg, info = plan["cfg"], dict(plan["info"])
    arabic = plan["arabic"]
    tmp, source = plan["tmp"], plan["src"]

    if not info["width"]:
        raise SystemExit("woven_box needs a video source (no audio-only input)")

    if info["width"] != info["height"]:
        subject = detect_subject(source, info, tmp) if cfg["detect_subject"] else None
        if subject:
            print("      face at x=%d y=%d (%dx%dpx, %d frames)"
                  % (subject["cx"], subject["cy"], subject["w"],
                     subject["h"], subject["hits"]))
        print("      cropping to 1:1 on the subject...")
        vf, cx, cy = square_crop_filter(info, subject)
        squared = os.path.join(tmp, "squared.mp4")
        run([FFMPEG, "-y", "-v", "error", "-i", source, "-vf", vf,
             "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", squared])
        source = squared
        import generate
        info = {**generate.get_video_info(source), "duration": info["duration"]}
        print("      cropped at x=%d y=%d -> %dx%d"
              % (cx, cy, info["width"], info["height"]))

    canvas_side = output_side(info["width"])
    box_w = int(round(canvas_side * BOX_FRAC / 2)) * 2
    box_x = box_y = (canvas_side - box_w) // 2
    box_h = box_w
    radius = box_w * CORNER_FRAC
    print("      canvas %dx%d, box %dx%d at (%d,%d), corner %.0fpx"
          % (canvas_side, canvas_side, box_w, box_h, box_x, box_y, radius))

    png_dir = os.path.join(tmp, "cards")
    os.makedirs(png_dir, exist_ok=True)
    texture_path = os.path.join(tmp, "texture.png")
    woven_texture(box_w, box_h, seed=hash(source) & 0xffff).save(texture_path)
    mask_path = os.path.join(tmp, "mask.png")
    rounded_mask(box_w, box_h, radius).save(mask_path)

    texts = [strip_display_marks(p["text"]) for p in arabic]
    ar_size = shared_font_size(texts, int(box_h * AR_SIZE_FRAC),
                               int(box_w * AR_MAX_WIDTH_FRAC))
    print("      jomhuria @ %dpx" % ar_size)

    center_x = box_x + box_w // 2 + int(cfg["x_offset"])
    center_y = box_y + box_h // 2 + int(cfg["y_offset"])

    overlays = []
    for i, (t, ph) in enumerate(zip(texts, arabic)):
        img, w, h = render_text_png(t, ar_size, box_w)
        assert_fits("arabic", i, w, box_w)
        card = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
        card.paste(img, (center_x - w // 2, center_y - h // 2), img)
        cropped, ox, oy = trim_to_ink(card)
        path = os.path.join(png_dir, "%03d.png" % i)
        cropped.save(path)
        overlays.append({"path": path, "start": ph["start"], "end": ph["end"],
                         "x": ox, "y": oy})

    if cfg["signature"]:
        sig_size = max(12, int(box_h * SIGNATURE_SIZE_FRAC))
        sig_img, sw, sh = render_text_png(cfg["signature"], sig_size, box_w)
        alpha = sig_img.getchannel("A").point(lambda v: int(v * SIGNATURE_ALPHA))
        sig_img.putalpha(alpha)
        card = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
        card.paste(sig_img, (canvas_side // 2 - sw // 2,
                             box_y + int(box_h * SIGNATURE_Y_FRAC) - sh // 2
                             + int(cfg["signature_offset"])), sig_img)
        cropped, ox, oy = trim_to_ink(card)
        sig_path = os.path.join(png_dir, "signature.png")
        cropped.save(sig_path)
        overlays.append({"path": sig_path, "x": ox, "y": oy, "full": True})

    composite(source, (box_w, box_h, box_x, box_y), texture_path, mask_path,
              overlays, plan["out"], cfg["dim"], info["duration"],
              {"w": canvas_side, "h": canvas_side, "fps": FPS})
    print("Done.")


def strip_display_marks(text):
    # Same tajweed-annotation strip as render_default.strip_display_marks --
    # ported rather than imported to keep this file self-contained.
    for c in ("۟", "ۭ", "ـ"):
        text = text.replace(c, "")
    return text
