"""pipeline/render_bloom.py -- the `bloom` (3rd) style renderer.

A bloomy, glowing static picture background style with rich atmospheric effects
and glowing text overlays:

Processing order for background:
  1. Slight blur/softening (radius=2)
  2. Bloom (intensity: 30%, threshold: 80%/204, radius: 25px, softness: 80)
  3. Glow (intensity: 12%, opacity: 25%, spread: 15px, blend: screen)
  4. Fog / Haze overlay (color: #E7E1D8, opacity: 10%, blur: 30px, screen)
  5. Film grain (monochrome noise, amount: 14)
  6. Text + text glow (text: #F8F5EE, outer glow: #FFFFFF opacity 72% blur 18px, shadow: #000000 opacity 20% blur 6px)
  7. Finishing touches (saturation: -8%, warmth: +5, fade: lift 10%, vignette: -8)

Called by generate.py with the resolved plan; not a standalone CLI.
"""
import math
import os
import random
import subprocess
import sys

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT_DIR = os.path.join(ROOT, "assets", "fonts")

FFMPEG = os.environ.get("QC_FFMPEG") or "ffmpeg"

if not features.check("raqm"):
    sys.exit("FATAL: Pillow lacks RAQM (HarfBuzz+FriBiDi) -- Arabic would be "
             "laid out unjoined, left to right. Use tools/render-venv/bin/python.")

ARABIC_FONTS = {
    "uthmanic_hafs": os.path.join(FONT_DIR, "uthmanic_hafs_v22.ttf"),
    "thuluth": os.path.join(FONT_DIR, "AM_Thulth_Regular_0.1.ttf"),
}
ENGLISH_FONTS = {
    "albertus": os.path.join(FONT_DIR, "Albertus MT Lt Regular.ttf"),
    "gentium": os.path.join(FONT_DIR, "GentiumPlus-Regular.ttf"),
}

DISPLAY_STRIP_CHARS = {"۟", "ۭ", "ـ"}
EN_AR_SIZE_RATIO = 121.0 / 176.0  # 121px English for 176px Arabic
FADE = 0.4

SIGNATURE_SIZE_FRAC = 39.0 / 1920.0  # 39px at 1920h frame height
SIGNATURE_Y_FRAC = 0.945      # center of the line
SIGNATURE_ALPHA = 0.87

# Text Glow parameters specified for the 3rd style
TEXT_COLOR = (248, 245, 238)      # #F8F5EE
OUTER_GLOW_COLOR = (255, 255, 255) # #FFFFFF
OUTER_GLOW_OPACITY = 0.72          # 65-80%
OUTER_GLOW_BLUR = 18.0             # 15-25 px
SHADOW_COLOR = (0, 0, 0)
SHADOW_OPACITY = 0.20              # 15-25%
SHADOW_BLUR = 6.0                  # 5-8 px

FACE_MODEL = os.path.join(ROOT, "tools", "models", "face_detection_yunet_2023mar.onnx")
FACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def strip_display_marks(text):
    return "".join(c for c in text if c not in DISPLAY_STRIP_CHARS)


# ---------- vision / subject detection ---------------------------------------

def detect_subject(video_path, info, tmp, samples=5):
    try:
        import cv2
    except ImportError:
        print("      opencv not installed -- skipping subject detection")
        return None
    if not os.path.exists(FACE_MODEL):
        os.makedirs(os.path.dirname(FACE_MODEL), exist_ok=True)
        try:
            run(["curl", "-sL", "-o", FACE_MODEL, FACE_MODEL_URL])
        except subprocess.CalledProcessError:
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
        return None
    med = lambda vals: sorted(vals)[len(vals) // 2]
    x, y = med([b[0] for b in boxes]), med([b[1] for b in boxes])
    w, h = med([b[2] for b in boxes]), med([b[3] for b in boxes])
    return {"cx": x + w // 2, "cy": y + h // 2, "w": w, "h": h, "hits": len(boxes)}


def vertical_crop_filter(info, subject, target_w=1080, target_h=1920):
    crop_w = int(info["height"] * target_w / target_h) // 2 * 2
    center = subject["cx"] if subject else info["width"] // 2
    x = max(0, min(info["width"] - crop_w, center - crop_w // 2))
    return ("crop=%d:%d:%d:0,scale=%d:%d:flags=lanczos"
            % (crop_w, info["height"], x, target_w, target_h)), x


def output_size(w, h):
    if not w or not h:
        return 1080, 1920
    if h >= w:
        short = max(720, min(1080, w))
        return short, int(short * h / w) // 2 * 2
    short = max(720, min(1080, h))
    return int(short * w / h) // 2 * 2, short


# ---------- text rendering with bloom & outer glow ---------------------------

def font_height(font):
    bbox = font.getbbox("Ay؀")
    return bbox[3] - bbox[1]


def measure_text_box(text, font_path, font_size):
    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox


def fit_font_size(text, font_path, target_h, max_w):
    size = target_h
    while size > 10:
        w, _, _ = measure_text_box(text, font_path, size)
        if w <= max_w:
            return size
        size -= 1
    return 10


def shared_font_size(texts, font_path, target_h, max_w):
    return min((fit_font_size(t, font_path, target_h, max_w) for t in texts if t),
               default=target_h)


def render_text_png_glowing(text, font_path, font_size):
    """Measures text box and creates text alpha mask."""
    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return text, tw, th, bbox


def assert_fits(label, card_idx, w, center_x, frame_w):
    half = w // 2
    if center_x - half < 0 or center_x + half > frame_w:
        raise ValueError(
            "card %d %s text (%dpx wide) overflows frame bounds [0, %d] "
            "at center_x=%d." % (card_idx + 1, label, w, frame_w, center_x))


def wrap_by_width(text, font, max_w, max_chars):
    words = text.split()
    if not words:
        return []
    dummy = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(dummy)
    w_of = lambda s: d.textbbox((0, 0), s, font=font)[2]
    lines, line = [], []
    for w in words:
        cand = " ".join(line + [w])
        if line and (w_of(cand) > max_w or len(cand) > max_chars):
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return lines


def trim_to_ink(img):
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return img, 0, 0
    return img.crop(bbox), bbox[0], bbox[1]


# ---------- background bloom & atmospheric effects ----------------------------

def process_bloom_background(bg_image):
    """Processes static picture background with bloomy & glowing effects:
    1. Slight blur/softening (radius=2)
    2. Bloom (intensity 30%, threshold 80%/204, radius 25px)
    3. Glow (intensity 12%, opacity 25%, spread 15px)
    4. Fog overlay (#E7E1D8, opacity 10%, blur 30px)
    5. Film grain (monochrome noise 14)
    6. Finishing touches (saturation -8%, warmth +5, fade 10%, vignette -8%)
    """
    bg = bg_image.convert("RGB")
    w, h = bg.size

    # 1. Reduced blur / softening for sharp image detail
    bg = bg.filter(ImageFilter.GaussianBlur(0.8))

    # 2. Subtle Background Bloom (threshold 80% luma -> 204, radius 25.0, 10% intensity)
    gray = bg.convert("L")
    mask = gray.point(lambda p: 255 if p > 204 else 0)
    hl = Image.new("RGB", (w, h), (0, 0, 0))
    hl.paste(bg, mask=mask)
    bloomed_hl = hl.filter(ImageFilter.GaussianBlur(25.0))
    screen_bloom = ImageChops.screen(bg, bloomed_hl)
    bg = Image.blend(bg, screen_bloom, 0.10)

    # 3. Subtle Background Glow (8% intensity)
    glow_map = bg.filter(ImageFilter.GaussianBlur(30.0))
    glow_screen = ImageChops.screen(bg, ImageEnhance.Brightness(glow_map).enhance(0.08))
    bg = Image.blend(bg, glow_screen, 0.08)

    # 4. Fog / Haze overlay (#E7E1D8, opacity 8%, blur 30px)
    fog_layer = Image.new("RGB", (w, h), (231, 225, 216))
    fog_screen = ImageChops.screen(bg, fog_layer)
    bg = Image.blend(bg, fog_screen, 0.08)

    # 5. Film grain (monochrome noise 8 blended at 10% opacity)
    noise = Image.effect_noise((w // 2, h // 2), 8).resize((w, h), Image.Resampling.NEAREST).convert("RGB")
    bg = Image.blend(bg, ImageChops.overlay(bg, noise), 0.10)

    # 6. Dim background image (65% brightness) so text & text glow stand out vividly
    bg = ImageEnhance.Brightness(bg).enhance(0.65)

    # 7. Finishing touches
    # Saturation: -8%
    bg = ImageEnhance.Color(bg).enhance(0.92)

    # Warmth: +5 (boost R channel +5, G +2)
    r, g, b = bg.split()
    r = r.point(lambda p: min(255, p + 5))
    g = g.point(lambda p: min(255, p + 2))
    bg = Image.merge("RGB", (r, g, b))

    # Vignette: -8 (soft edge darkening)
    v_mask = Image.new("L", (w, h), 0)
    d_v = ImageDraw.Draw(v_mask)
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.6), int(h * 0.6)
    d_v.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    v_mask = ImageOps.invert(v_mask.filter(ImageFilter.GaussianBlur(100)))
    v_mask = v_mask.point(lambda p: int(p * 0.18))
    dark_bg = Image.new("RGB", (w, h), (0, 0, 0))
    bg = Image.composite(dark_bg, bg, v_mask)

    return bg


# ---------- ffmpeg video compositing ----------------------------------------

def composite(bg_image_path, overlays, out_path, audio_source, duration, size):
    width, height = size
    eps = 0.001
    cmd = [FFMPEG, "-y", "-v", "error"]

    # Input 0: processed static picture background
    cmd += ["-loop", "1", "-i", bg_image_path]

    # Input 1: audio track from source
    cmd += ["-i", audio_source]
    audio_map = "1:a"

    # Input 2..N: overlays
    base = 2
    for ov in overlays:
        cmd += ["-loop", "1", "-i", ov["path"]]

    parts = []
    label = "0:v"
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
               "-map", audio_map, "-t", "%.3f" % duration,
               "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", out_path])


# ---------- style entry point ------------------------------------------------

def render(plan):
    cfg, info = plan["cfg"], dict(plan["info"])
    arabic, english = plan["arabic"], plan["english"]
    tmp, source = plan["tmp"], plan["src"]

    if cfg["arabic_font"] not in ARABIC_FONTS:
        raise SystemExit("arabic_font must be one of %s" % sorted(ARABIC_FONTS))
    if cfg["english_font"] not in ENGLISH_FONTS:
        raise SystemExit("english_font must be one of %s" % sorted(ENGLISH_FONTS))

    ar_font_path = ARABIC_FONTS[cfg["arabic_font"]]
    en_font_path = ENGLISH_FONTS[cfg["english_font"]]

    still = cfg.get("background_image")
    if still:
        info["width"], info["height"] = cfg["frame_size"]
        print("      still background -> %dx%d: %s"
              % (info["width"], info["height"], os.path.basename(still)))
        bg_raw = Image.open(still).resize(cfg["frame_size"], Image.Resampling.LANCZOS)
    else:
        # Extract representative frame from source video if no still is specified
        tmp_frame = os.path.join(tmp, "source_frame.png")
        t_sample = min(2.0, info["duration"] / 2)
        run([FFMPEG, "-y", "-v", "error", "-ss", "%.2f" % t_sample, "-i", source,
             "-frames:v", "1", "-update", "1", tmp_frame])
        bg_raw = Image.open(tmp_frame)
        if info["width"] > info["height"]:
            # Default to 9:16 vertical crop for bloom style
            crop_w = int(info["height"] * 1080 / 1920) // 2 * 2
            x = (info["width"] - crop_w) // 2
            bg_raw = bg_raw.crop((x, 0, x + crop_w, info["height"])).resize((1080, 1920), Image.Resampling.LANCZOS)
            info["width"], info["height"] = 1080, 1920

    width, height = info["width"], info["height"]
    print("      processing bloom & glowing background effects (%dx%d)..." % (width, height))
    bg_processed = process_bloom_background(bg_raw)
    processed_bg_path = os.path.join(tmp, "processed_bloom_bg.png")
    bg_processed.save(processed_bg_path)

    center_x = width // 2 + int(cfg["x_offset"])
    margin = int(width * (1.0 - cfg["text_width"]) / 2)
    max_ar_width = 2 * min(center_x - margin, (width - margin) - center_x)

    texts = [strip_display_marks(p["text"]) for p in arabic]
    target_ar_h = height * (176.0 / 1920.0) * (cfg["arabic_scale"] / 1.1375)
    ar_size = shared_font_size(texts, ar_font_path,
                               round(target_ar_h),
                               max_ar_width)
    en_size = max(12, round(ar_size * EN_AR_SIZE_RATIO * cfg["english_scale"]))
    print("      arabic: %s @ %dpx | english: %s @ %dpx (ratio %.2f)"
          % (cfg["arabic_font"], ar_size, cfg["english_font"], en_size, en_size / ar_size))

    ar_pngs = [render_text_png_glowing(t, ar_font_path, ar_size) for t in texts]
    ar_h = max(th for _, _, th, _ in ar_pngs)

    en_font = ImageFont.truetype(en_font_path, en_size)
    en_wrapped = [wrap_by_width(p["text"], en_font, max_ar_width,
                                cfg["english_max_chars"]) if p["text"] else []
                  for p in english]
    en_line_h = int(en_size * 1.25)
    en_h = max((len(ls) for ls in en_wrapped), default=0) * en_line_h

    gap = int(height * cfg["line_gap"])
    block_h = ar_h + gap + en_h
    block_top = (height - block_h) // 2 + int(cfg["y_offset"])
    print("      block: top=%d height=%d" % (block_top, block_h))

    png_dir = os.path.join(tmp, "cards_bloom")
    os.makedirs(png_dir, exist_ok=True)
    overlays = []
    ar_font = ImageFont.truetype(ar_font_path, ar_size)
    en_font = ImageFont.truetype(en_font_path, en_size)

    png_dir = os.path.join(tmp, "cards_bloom")
    os.makedirs(png_dir, exist_ok=True)
    overlays = []

    for i, (ar_ph, lines, ph) in enumerate(zip(arabic, en_wrapped, arabic)):
        ar_text = strip_display_marks(ar_ph["text"])
        
        # 1. Measure positions
        bbox_ar = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), ar_text, font=ar_font)
        aw = bbox_ar[2] - bbox_ar[0]
        assert_fits("arabic", i, aw, center_x, width)
        tx_ar = center_x - (bbox_ar[2] + bbox_ar[0]) // 2
        ty_ar = block_top + (ar_h - (bbox_ar[3] - bbox_ar[1])) // 2 - bbox_ar[1]

        # Create card text alpha mask
        alpha_mask = Image.new("L", (width, height), 0)
        d_alpha = ImageDraw.Draw(alpha_mask)
        d_alpha.text((tx_ar, ty_ar), ar_text, font=ar_font, fill=255)

        y = block_top + ar_h + gap
        for line in lines:
            bbox_en = d_alpha.textbbox((0, 0), line, font=en_font)
            lw = bbox_en[2] - bbox_en[0]
            assert_fits("english", i, lw, center_x, width)
            tx_en = center_x - (bbox_en[2] + bbox_en[0]) // 2
            ty_en = y + (en_line_h - (bbox_en[3] - bbox_en[1])) // 2 - bbox_en[1]
            d_alpha.text((tx_en, ty_en), line, font=en_font, fill=255)
            y += en_line_h

        # 2. Dark Drop Shadow (Black #000000, Opacity 65%, Blur 8px for crisp contrast)
        shadow_a = alpha_mask.filter(ImageFilter.GaussianBlur(8.0)).point(lambda p: int(p * 0.65))
        shadow_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shadow_img.putalpha(shadow_a)

        # 3. Soft Balanced Outer Glow (#FFFFFF, Opacity 45%, Blur 12px)
        dilated = alpha_mask.filter(ImageFilter.MaxFilter(5))
        glow_a = dilated.filter(ImageFilter.GaussianBlur(12.0)).point(lambda p: min(255, int(p * 1.2)))
        glow_a = glow_a.point(lambda p: int(p * 0.45))
        glow_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        glow_img.putalpha(glow_a)

        # 4. Main Crisp Text layer (#F8F5EE)
        main_img = Image.new("RGBA", (width, height), TEXT_COLOR + (0,))
        main_img.putalpha(alpha_mask)

        # Composite order: Soft Glow -> Dark Drop Shadow -> Main Crisp Text
        card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        card = Image.alpha_composite(card, glow_img)
        card = Image.alpha_composite(card, shadow_img)
        card = Image.alpha_composite(card, main_img)

        cropped, ox, oy = trim_to_ink(card)
        path = os.path.join(png_dir, "%03d.png" % i)
        cropped.save(path)
        overlays.append({"path": path, "start": ph["start"], "end": ph["end"], "x": ox, "y": oy})

    if cfg["signature"]:
        sig_size = max(12, round(height * SIGNATURE_SIZE_FRAC))
        sig_font = ImageFont.truetype(en_font_path, sig_size)
        sig_text = cfg["signature"]
        
        alpha_mask = Image.new("L", (width, height), 0)
        d_alpha = ImageDraw.Draw(alpha_mask)
        bbox_sig = d_alpha.textbbox((0, 0), sig_text, font=sig_font)
        tx_sig = width // 2 - (bbox_sig[2] + bbox_sig[0]) // 2
        ty_sig = int(height * SIGNATURE_Y_FRAC) - (bbox_sig[3] - bbox_sig[1]) // 2 + int(cfg["signature_offset"]) - bbox_sig[1]
        d_alpha.text((tx_sig, ty_sig), sig_text, font=sig_font, fill=255)

        # 1. Dark Drop Shadow (Black #000000, Opacity 65%, Blur 8px)
        shadow_a = alpha_mask.filter(ImageFilter.GaussianBlur(8.0)).point(lambda p: int(p * 0.65))
        shadow_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shadow_img.putalpha(shadow_a)

        # 2. Soft Balanced Outer Glow (#FFFFFF, Opacity 45%, Blur 12px)
        dilated = alpha_mask.filter(ImageFilter.MaxFilter(5))
        glow_a = dilated.filter(ImageFilter.GaussianBlur(12.0)).point(lambda p: min(255, int(p * 1.2)))
        glow_a = glow_a.point(lambda p: int(p * 0.45))
        glow_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        glow_img.putalpha(glow_a)

        # 3. Main Crisp Text layer (#F8F5EE)
        main_img = Image.new("RGBA", (width, height), TEXT_COLOR + (0,))
        main_img.putalpha(alpha_mask)

        # Composite order: Soft Glow -> Dark Drop Shadow -> Main Crisp Text
        card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        card = Image.alpha_composite(card, glow_img)
        card = Image.alpha_composite(card, shadow_img)
        card = Image.alpha_composite(card, main_img)

        cropped, ox, oy = trim_to_ink(card)
        sig_path = os.path.join(png_dir, "signature.png")
        cropped.save(sig_path)
        overlays.append({"path": sig_path, "x": ox, "y": oy, "full": True})

    composite(processed_bg_path, overlays, plan["out"], plan["src"], info["duration"], size=(width, height))
