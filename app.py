"""
Crowd-Free Miracle Shot — AI people-removal app (single file).

Upload a garden photo and remove the strangers in the BACKGROUND while keeping
YOU (the main subject). The visitor picks which detected people to keep, so it
always works — no guessing.

AI engine (fully local, no API key):
  1) YOLOv8-seg  — detect & segment every person -> list of people (mask+box)
  2) UI          — show each person numbered; you tick the ones to KEEP
  3) LaMa        — inpaint the masked area, rebuilding the background
  4) feather-blend — merge back onto the full-res original

Models auto-download on first use (yolov8n-seg.pt + big-lama.pt, ~200 MB).

Deploy on Streamlit Community Cloud (free): set app.py as the main file.
"""

import os
import re
import time
import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import phonenumbers
import streamlit as st
from PIL import Image


# ============================================================================
#  AI PIPELINE  (inlined so the whole app is one file)
# ============================================================================

MODELS_DIR = Path(os.environ.get("CROWDFREE_MODELS", Path(__file__).resolve().parent / "models"))
MAX_DIM = int(os.environ.get("CROWDFREE_MAX_DIM", "1024"))
_yolo, _lama = None, None


def _load_yolo():
    global _yolo
    if _yolo is None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR / "yolo_config"))
        from ultralytics import YOLO
        w = MODELS_DIR / "yolov8n-seg.pt"
        _yolo = YOLO(str(w) if w.exists() else "yolov8n-seg.pt")
    return _yolo


def _load_lama():
    global _lama
    if _lama is None:
        w = MODELS_DIR / "big-lama.pt"
        if w.exists():
            os.environ["LAMA_MODEL"] = str(w)
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
    return _lama


def detect_people(img_bgr, conf=0.35):
    """Return list of detected persons: {mask(0-255), box(x1,y1,x2,y2), area, cx, cy}."""
    model = _load_yolo()
    H, W = img_bgr.shape[:2]
    res = model.predict(img_bgr, conf=conf, verbose=False)[0]
    people = []
    if res.masks is not None and res.boxes is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        boxes = res.boxes.xyxy.cpu().numpy()
        mdata = res.masks.data.cpu().numpy()
        names = res.names
        for i, c in enumerate(cls):
            if names.get(c) == "person":
                x1, y1, x2, y2 = boxes[i]
                m = cv2.resize(mdata[i].astype(np.uint8) * 255, (W, H),
                               interpolation=cv2.INTER_LINEAR)
                people.append({
                    "mask": m,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "area": int((m > 0).sum()),
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                })
    return people


def draw_overlay(img_bgr, people, keep_idx):
    """Draw each person with a numbered box; green = keep, red = remove."""
    img = img_bgr.copy()
    for n, p in enumerate(people):
        x1, y1, x2, y2 = p["box"]
        color = (60, 220, 60) if n in keep_idx else (0, 60, 255)  # BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"P{n+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 12, y1), color, -1)
        cv2.putText(img, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
    return img


def _downscale(img, mask, max_dim):
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s >= 1.0:
        return img, mask
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    return img, mask


def _feather_blend(original, inpainted_upscaled, mask):
    mask = mask.astype(np.float32) / 255.0
    k = max(5, int(min(original.shape[:2]) * 0.008))
    mask = cv2.GaussianBlur(mask, (k | 1, k | 1), 0)[..., None]
    return np.clip(original.astype(np.float32) * (1 - mask)
                   + inpainted_upscaled.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def inpaint_people(img_bgr, people_to_remove, max_dim=None, dilate_px=8):
    """Inpaint the given people out of the image. Returns (cleaned_bgr, n_removed, mask_px)."""
    max_dim = max_dim or MAX_DIM
    H, W = img_bgr.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for p in people_to_remove:
        mask = np.maximum(mask, p["mask"])
    mask_px = int((mask > 0).sum())
    if mask_px == 0:
        return img_bgr.copy(), len(people_to_remove), 0

    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((max(3, dilate_px), max(3, dilate_px)), np.uint8))

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_s, mask_s = _downscale(rgb, mask, max_dim)
    t0 = time.time()
    out_s = np.array(_load_lama()(Image.fromarray(rgb_s), Image.fromarray(mask_s)))
    out_full = cv2.resize(cv2.cvtColor(out_s, cv2.COLOR_RGB2BGR), (W, H),
                          interpolation=cv2.INTER_CUBIC)
    cleaned = _feather_blend(img_bgr, out_full, mask)
    return cleaned, len(people_to_remove), mask_px


# ============================================================================
#  CONTACT / CONSENT HELPERS
# ============================================================================

@st.cache_resource
def data_path():
    p = Path("data") / "entries.csv"
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        p.write_text("id,timestamp,name,whatsapp,country_code,email,market,"
                     "opt_feature,opt_draw,people_removed\n")
    return str(p)


def append_entry(rec):
    with open(data_path(), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(rec.keys())).writerow(rec)


def normalize_phone(v):
    """Return E.164, or raise ValueError. Accepts UAE-style + full international."""
    v = v.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    if not v:
        raise ValueError("Please enter your phone number.")
    if v.startswith("+"):
        try:
            num = phonenumbers.parse(v, None)
        except phonenumbers.NumberParseException:
            raise ValueError("That number doesn't look right.")
        if not phonenumbers.is_valid_number(num):
            raise ValueError(f"Invalid number ({phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.INTERNATIONAL)}).")
        return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
    try:
        num = phonenumbers.parse(v, "AE")
    except phonenumbers.NumberParseException:
        raise ValueError("For non-UAE numbers include your country code, e.g. +44 7911 123456.")
    if phonenumbers.is_valid_number(num):
        return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
    raise ValueError("That number doesn't look right. For UAE use e.g. 050 123 4567; "
                     "otherwise include your country code (+...).")


# ============================================================================
#  APP
# ============================================================================

st.set_page_config(page_title="Crowd-Free Miracle Shot", page_icon="🌸", layout="centered")
st.markdown("""
<style>
  .big{font-size:1.4rem;font-weight:800;color:#0d3d22}
  .muted{color:#66736b;font-size:.9rem}
  .badge{display:inline-block;background:#e7f4ec;color:#14532d;border-radius:20px;padding:3px 10px;font-size:.78rem;font-weight:700;margin-right:6px}
  .stat{background:#f3f7f4;border:1px solid #e2eae4;border-radius:10px;padding:10px;text-align:center}
  .stat b{color:#14532d;font-size:1.1rem;display:block}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big">🌸 Crowd-Free Miracle Shot</div>', unsafe_allow_html=True)
st.markdown('<span class="muted">Erase the strangers in the <b>background</b> of your garden '
            'photo — <b>you</b> stay in the shot. You pick who to keep.</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Free self-hosted AI</span>'
            '<span class="badge">YOLOv8 + LaMa</span>'
            '<span class="badge">You choose who stays</span>', unsafe_allow_html=True)

uploaded = st.file_uploader("📷 Upload your garden photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ---- detect people once per uploaded photo ----
people = []
img_bgr = None
if uploaded is not None:
    upbytes = uploaded.getvalue()
    if st.session_state.get("_up") != upbytes:
        st.session_state["_up"] = upbytes
        img_bgr = cv2.imdecode(np.frombuffer(upbytes, np.uint8), cv2.IMREAD_COLOR)
        st.session_state["_img"] = img_bgr
        st.session_state["_people"] = detect_people(img_bgr) if img_bgr is not None else []
    img_bgr = st.session_state.get("_img")
    people = st.session_state.get("_people", [])

tab_v, tab_o = st.tabs(["Visitor", "Ops · captured data"])

with tab_v:
    if uploaded is None:
        st.info("Upload a photo to start.")
    elif img_bgr is None:
        st.error("Could not read that image. Try a JPG or PNG.")
    elif not people:
        st.warning("No people detected in this photo — nothing to remove. It may already be "
                   "crowd-free, or the people are too small.")
        st.image(img_bgr[:, :, ::-1], caption="Your photo", use_container_width=True)
    else:
        st.markdown("#### 1 · Who should stay?")
        st.caption("Green boxes stay, red boxes get removed. Pick who to KEEP.")
        # default: keep the largest person (most likely the subject)
        default_keep = [int(np.argmax([p["area"] for p in people]))]
        with st.form("capture"):
            keep_idx = st.multiselect(
                "People to KEEP (highlighted green) — keep yourself, remove the rest",
                options=list(range(len(people))),
                default=default_keep,
                format_func=lambda i: f"P{i+1}",
            )
            remove_all = st.checkbox("Remove ALL people (empty the scene)", value=False)

            c1, c2 = st.columns(2)
            name = c1.text_input("First name", placeholder="e.g. Aisha")
            market = c2.selectbox("Visiting from", ["UAE resident", "Saudi Arabia", "India",
                                                    "United Kingdom", "Russia / CIS", "Germany",
                                                    "US / Canada", "Morocco", "Other international"])
            ph = st.text_input("Phone", placeholder="e.g. 050 123 4567  or  +44 7911 123456")
            email = st.text_input("Email (optional)", placeholder="you@email.com")
            st.markdown("#### 2 · Optional — how would you like to join in?")
            opt_feature = st.checkbox("🌸 Yes — you may feature my photo on the official garden "
                                      "social channels.")
            opt_draw = st.checkbox("🏆 Yes — enter my contact details into the 19:00 Daily "
                                   "Bloom Draw for a chance to win.")
            go = st.form_submit_button("✨ Remove the background people", type="primary",
                                       use_container_width=True)

        keep_idx = set(keep_idx)
        overlay = draw_overlay(img_bgr, people, keep_idx)
        st.image(overlay[:, :, ::-1], caption="Green = keep · Red = remove", use_container_width=True)

        if go:
            errs = []
            if not name.strip():
                errs.append("Please add your first name.")
            try:
                wa = normalize_phone(ph)
            except ValueError as ex:
                errs.append(str(ex))
            if email.strip() and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
                errs.append("That email doesn't look valid (you can leave it blank).")
            if not remove_all and not keep_idx:
                errs.append("You ticked no one to keep — tick the person who should stay "
                            "(or choose 'Remove all').")
            if errs:
                for e in errs:
                    st.error(e)
            else:
                # who to remove
                to_remove = [p for n, p in enumerate(people) if remove_all or (n not in keep_idx)]
                if not to_remove:
                    st.info("Everyone is kept — nothing to remove. Untick someone if you want "
                            "background people removed.")
                else:
                    with st.spinner("AI is removing the background people… (~10s on CPU, "
                                    "first run downloads models)"):
                        try:
                            cleaned, n_removed, _ = inpaint_people(img_bgr, to_remove)
                        except Exception as ex:
                            st.error(f"AI error: {ex}")
                            st.stop()
                    parsed = phonenumbers.parse(wa, None)
                    append_entry({
                        "id": str(int(time.time())),
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "name": name.strip(),
                        "whatsapp": wa,
                        "country_code": "+" + str(parsed.country_code),
                        "email": email.strip().lower(),
                        "market": market,
                        "opt_feature": "yes" if opt_feature else "no",
                        "opt_draw": "yes" if opt_draw else "no",
                        "people_removed": n_removed,
                    })
                    ca, cb = st.columns(2)
                    ca.image(img_bgr[:, :, ::-1], caption="Original", use_container_width=True)
                    cb.image(cleaned[:, :, ::-1], caption=f"Crowd-free ✨ ({n_removed} removed)",
                             use_container_width=True)
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f'<div class="stat"><b>{n_removed}</b>people removed</div>',
                                unsafe_allow_html=True)
                    m2.markdown(f'<div class="stat"><b>{len(to_remove)}</b>removed</div>',
                                unsafe_allow_html=True)
                    m3.markdown('<div class="stat"><b>$0</b>per photo</div>', unsafe_allow_html=True)
                    ok, buf = cv2.imencode(".jpg", cleaned, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        st.download_button("⬇ Download crowd-free photo", buf.tobytes(),
                                           file_name="my-crowdfree-miracle-shot.jpg",
                                           mime="image/jpeg", use_container_width=True)
                    st.success(f"✨ Done, {name.strip()}! Removed {n_removed} background "
                               f"person(s) and kept you/your group. Tag #MyMiracleMoment.")

with tab_o:
    st.caption("Contacts captured by this after-sale add-on — incl. who opted in to be "
               "featured on social (opt_feature) and who entered the draw (opt_draw).")
    try:
        import pandas as pd
        df = pd.read_csv(data_path())
        st.dataframe(df, use_container_width=True) if not df.empty else st.info(
            "No entries yet — submit a photo above.")
    except Exception as ex:
        st.info(f"Data store not ready yet: {ex}")
