"""
Crowd-Free Miracle Shot — AI people-removal app (single file).

Upload a garden photo with strangers in the background and the AI erases every
person, reconstructing the flower scenery behind them. Captures a consented
contact (name + international phone, optional email, market).

AI engine (fully local, no API key):
  1) YOLOv8-seg  — detect & segment every person -> binary person mask
  2) LaMa         — inpaint the masked area, rebuilding the background
  3) feather-blend — merge back onto the full-res original (keeps detail)

Models auto-download on first use (yolov8n-seg.pt + big-lama.pt, ~200 MB).

Deploy on Streamlit Community Cloud (free): set app.py as the main file.
"""

import io
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


def remove_people(img_bgr, max_dim=None, dilate_px=6):
    max_dim = max_dim or MAX_DIM
    t0 = time.time()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    model = _load_yolo()
    res = model.predict(rgb, conf=0.35, verbose=False)[0]
    person_mask = np.zeros((H, W), np.uint8)
    people = 0
    if res.masks is not None and res.boxes is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        mdata = res.masks.data.cpu().numpy()
        names = res.names
        for i, c in enumerate(cls):
            if names.get(c) == "person":
                people += 1
                m = cv2.resize(mdata[i].astype(np.uint8) * 255, (W, H),
                               interpolation=cv2.INTER_LINEAR)
                person_mask = np.maximum(person_mask, m)
    mask_px = int((person_mask > 0).sum())
    if mask_px > 0 and dilate_px > 0:
        k = max(3, int(dilate_px))
        person_mask = cv2.dilate(person_mask, np.ones((k, k), np.uint8))
    t_detect = time.time() - t0

    cleaned = img_bgr.copy()
    t_inpaint = 0.0
    if people > 0 and mask_px > 0:
        rgb_s, mask_s = _downscale(rgb, person_mask, max_dim)
        t1 = time.time()
        out_s = np.array(_load_lama()(Image.fromarray(rgb_s), Image.fromarray(mask_s)))
        t_inpaint = time.time() - t1
        out_full = cv2.resize(cv2.cvtColor(out_s, cv2.COLOR_RGB2BGR), (W, H),
                              interpolation=cv2.INTER_CUBIC)
        cleaned = _feather_blend(img_bgr, out_full, person_mask)

    return {"cleaned_bgr": cleaned, "people_count": people, "mask_px": mask_px,
            "timings": {"detect": round(t_detect, 2), "inpaint": round(t_inpaint, 2),
                        "total": round(time.time() - t0, 2)}}


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
st.markdown('<span class="muted">After-sale AI add-on — erase every background person '
            'and keep your perfect garden shot.</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Free self-hosted AI</span>'
            '<span class="badge">YOLOv8 + LaMa</span>'
            '<span class="badge">No API key</span>', unsafe_allow_html=True)


@st.cache_resource
def data_path():
    p = Path("data") / "entries.csv"
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        p.write_text("id,timestamp,name,whatsapp,country_code,email,market,consent,people_removed\n")
    return str(p)


def append_entry(rec):
    with open(data_path(), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(rec.keys())).writerow(rec)


def validate_phone(v):
    v = v.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if not v:
        return "Please enter your phone number."
    if not v.startswith("+"):
        return "Include your country code, e.g. +971 50 123 4567."
    try:
        num = phonenumbers.parse(v, None)
    except phonenumbers.NumberParseException:
        return "That number doesn't look right."
    if not phonenumbers.is_valid_number(num):
        return (f"Invalid number "
                f"({phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.INTERNATIONAL)}).")
    return None


uploaded = st.file_uploader("📷 Upload your garden photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
tab_v, tab_o = st.tabs(["Visitor", "Ops · captured data"])

with tab_v:
    with st.form("capture"):
        c1, c2 = st.columns(2)
        name = c1.text_input("First name", placeholder="e.g. Aisha")
        market = c2.selectbox("Visiting from", ["UAE resident", "Saudi Arabia", "India",
                                                "United Kingdom", "Russia / CIS", "Germany",
                                                "US / Canada", "Morocco", "Other international"])
        ph = st.text_input("Phone (international)", placeholder="+971 50 123 4567")
        email = st.text_input("Email (optional)", placeholder="you@email.com")
        consent = st.checkbox("I agree Dubai Miracle Garden may store my contact for the draw "
                              "and future seasons, and may feature my cleaned photo "
                              "(UAE PDPL consent; opt-out anytime).", value=True)
        go = st.form_submit_button("✨ Remove people from my photo", type="primary",
                                   use_container_width=True)

    if uploaded is not None:
        st.image(uploaded, caption="Original", use_container_width=True)

    if go:
        errs = []
        if not uploaded:
            errs.append("Please upload a photo first.")
        if not name.strip():
            errs.append("Please add your first name.")
        werr = validate_phone(ph)
        if werr:
            errs.append(werr)
        if email.strip() and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
            errs.append("That email doesn't look valid.")
        if errs:
            for e in errs:
                st.error(e)
        else:
            arr = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not read that image. Try a JPG or PNG.")
            else:
                parsed = phonenumbers.parse(ph.replace(" ", "").replace("-", ""), None)
                append_entry({
                    "id": str(int(time.time())),
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "name": name.strip(),
                    "whatsapp": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                    "country_code": "+" + str(parsed.country_code),
                    "email": email.strip().lower(),
                    "market": market,
                    "consent": "yes" if consent else "no",
                    "people_removed": 0,
                })
                with st.spinner("AI is removing the crowd… ~10s on CPU (first run downloads models)."):
                    try:
                        r = remove_people(img_bgr)
                    except Exception as ex:
                        st.error(f"AI error: {ex}")
                        st.stop()
                ca, cb = st.columns(2)
                ca.image(img_bgr[:, :, ::-1], caption="Original", use_container_width=True)
                cb.image(r["cleaned_bgr"][:, :, ::-1], caption=f"Crowd-free ✨ ({r['people_count']} removed)",
                         use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="stat"><b>{r["people_count"]}</b>people removed</div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="stat"><b>{r["timings"]["total"]}s</b>AI time</div>', unsafe_allow_html=True)
                m3.markdown('<div class="stat"><b>$0</b>per photo</div>', unsafe_allow_html=True)
                ok, buf = cv2.imencode(".jpg", r["cleaned_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    st.download_button("⬇ Download crowd-free photo", buf.tobytes(),
                                       file_name="my-crowdfree-miracle-shot.jpg",
                                       mime="image/jpeg", use_container_width=True)
                st.success(f"✨ Done, {name.strip()}! {r['people_count']} background person(s) "
                           f"removed. Tag #MyMiracleMoment and enter the Daily Bloom Draw at 19:00.")

with tab_o:
    st.caption("Consented contacts captured by this after-sale add-on.")
    try:
        import pandas as pd
        df = pd.read_csv(data_path())
        st.dataframe(df, use_container_width=True) if not df.empty else st.info(
            "No entries yet — submit a photo above.")
    except Exception as ex:
        st.info(f"Data store not ready yet: {ex}")
