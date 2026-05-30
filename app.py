import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# CONFIG
# ============================================================

SPREADSHEET_ID = "13XAwI8y9F6yox2yFdXWQ8kn80ep9E-uUA-xn8WI7b5Y"

DATA_FILE_PL = Path("hasil_hitung_fastcount_pl.csv")
DATA_FILE_BACTERI = Path("hasil_hitung_fastcount_bacteri.csv")
HEADER_LOGO = Path("logo_header.png")

# ============================================================
# GOOGLE SHEET
# ============================================================

def get_gsheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )
    return gspread.authorize(credentials)


def save_pl_to_google_sheet(row):
    client = get_gsheet_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    worksheet_name = "FastCount PL"

    try:
        sheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
        sheet.append_row([
            "tanggal",
            "unit",
            "tank",
            "umur_pl",
            "operator",
            "hasil_deteksi",
            "hasil_koreksi",
            "catatan",
        ])

    sheet.append_row([
        row["tanggal"],
        row["unit"],
        row["tank"],
        row["umur_pl"],
        row["operator"],
        row["hasil_deteksi"],
        row["hasil_koreksi"],
        row["catatan"],
    ])


def save_bacteri_to_google_sheet(row):
    client = get_gsheet_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    worksheet_name = "FastCount Bacteri"

    try:
        sheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
        sheet.append_row([
            "tanggal",
            "unit",
            "kode_sampel",
            "operator",
            "jenis_media",
            "pengenceran",
            "volume_tanam_ml",
            "koloni_kuning",
            "koloni_hijau",
            "koloni_lainnya",
            "total_koloni",
            "cfu_ml",
            "catatan",
        ])

    sheet.append_row([
        row["tanggal"],
        row["unit"],
        row["kode_sampel"],
        row["operator"],
        row["jenis_media"],
        row["pengenceran_label"],
        row["volume_tanam_ml"],
        row["koloni_kuning"],
        row["koloni_hijau"],
        row["koloni_lainnya"],
        row["total_koloni"],
        row["cfu_ml"],
        row["catatan"],
    ])


def safe_save_to_google(save_function, row):
    """
    Supaya aplikasi tidak crash bila Secrets/Google Sheets belum benar.
    Return: (success: bool, message: str)
    """
    try:
        save_function(row)
        return True, "Data berhasil tersimpan ke Google Sheets."
    except KeyError:
        return False, "Data tersimpan ke CSV. Streamlit Secrets Google Sheets belum ditemukan."
    except Exception as e:
        return False, f"Data tersimpan ke CSV. Google Sheets belum tersimpan: {type(e).__name__}"


def append_to_csv(path, row):
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)


# ============================================================
# DETECTION: FASTCOUNT PL
# ============================================================

def detect_pl(image_rgb, threshold, min_area, max_area, blur):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if blur % 2 == 0:
        blur += 1

    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = image_rgb.copy()
    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if min_area <= area <= max_area:
            count += 1
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 210, 170), 2)
            cv2.putText(
                result,
                str(count),
                (x, max(y - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 205, 45),
                2
            )

    return result, thresh, count


# ============================================================
# DETECTION: FASTCOUNT BACTERI
# ============================================================

def dilution_label_to_factor(label):
    mapping = {
        "10⁰ / tanpa pengenceran": 1,
        "10⁻¹": 10,
        "10⁻²": 100,
        "10⁻³": 1000,
        "10⁻⁴": 10000,
        "10⁻⁵": 100000,
        "10⁻⁶": 1000000,
    }
    return mapping.get(label, 1)


def classify_tcbs_colony(mean_rgb):
    """
    Klasifikasi sederhana koloni pada TCBS.
    Return:
    - yellow
    - green
    - None untuk objek tidak jelas/noise.
    Titik merah/unclassified tidak ditampilkan.
    """
    r, g, b = mean_rgb

    # Koloni kuning: red & green kuat, blue rendah.
    if r > 105 and g > 90 and b < 135 and r >= b * 1.12:
        return "yellow"

    # Koloni hijau: green relatif dominan.
    if g > 75 and g >= r * 0.78 and g >= b * 1.02:
        return "green"

    return None


def detect_bacteri_colonies(
    image_rgb,
    media_type="TCBS Vibrio",
    threshold=45,
    min_area=10,
    max_area=950,
    blur=5,
    petri_margin=0.90
):
    """
    Untuk TCBS Vibrio:
    - Yang ditampilkan hanya koloni kuning dan hijau.
    - Objek unclassified tidak ditampilkan.
    """
    result = image_rgb.copy()
    h, w = image_rgb.shape[:2]

    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * petri_margin / 2)

    petri_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(petri_mask, (cx, cy), radius, 255, -1)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if blur % 2 == 0:
        blur += 1

    gray_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    petri_pixels = gray_blur[petri_mask == 255]
    bg = np.percentile(petri_pixels, 70) if len(petri_pixels) else 180

    # Objek koloni: lebih gelap dari background atau memiliki saturasi warna.
    dark_mask = ((bg - gray_blur) > threshold).astype(np.uint8) * 255

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat_mask = (hsv[:, :, 1] > 35).astype(np.uint8) * 255

    combined = cv2.bitwise_or(dark_mask, sat_mask)
    combined = cv2.bitwise_and(combined, petri_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    yellow_count = 0
    green_count = 0
    other_count = 0
    total_count = 0
    number = 0

    # Lingkar area cawan.
    cv2.circle(result, (cx, cy), radius, (0, 220, 240), 3)

    for contour in contours:
        area = cv2.contourArea(contour)

        if not (min_area <= area <= max_area):
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        px = int(M["m10"] / M["m00"])
        py = int(M["m01"] / M["m00"])

        if (px - cx) ** 2 + (py - cy) ** 2 > (radius * 0.96) ** 2:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_rgb = cv2.mean(image_rgb, mask=mask)[:3]

        if media_type == "TCBS Vibrio":
            colony_type = classify_tcbs_colony(mean_rgb)

            # Tidak menampilkan titik merah/unclassified.
            if colony_type is None:
                continue

            if colony_type == "yellow":
                color = (255, 210, 45)
                yellow_count += 1
            else:
                color = (65, 220, 120)
                green_count += 1
        else:
            colony_type = "other"
            color = (70, 210, 245)
            other_count += 1

        number += 1
        total_count += 1

        cv2.circle(result, (px, py), 9, color, -1)
        cv2.circle(result, (px, py), 9, (255, 255, 255), 2)
        cv2.putText(
            result,
            str(number),
            (px + 10, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2
        )
        cv2.putText(
            result,
            str(number),
            (px + 10, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (5, 45, 80),
            1
        )

    counts = {
        "yellow": yellow_count,
        "green": green_count,
        "other": other_count,
        "total": total_count,
    }

    return result, combined, counts


def format_cfu(value):
    if value >= 100000:
        return f"{value:.2e}".replace("e+", " × 10^")
    return f"{value:,.0f}".replace(",", ".")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AquaCount",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK PREMIUM CSS
# ============================================================

st.markdown("""
<style>
:root {
    --bg0: #06111f;
    --bg1: #081827;
    --panel: #0b2035;
    --panel2: #0e2a44;
    --border: rgba(91, 198, 255, .18);
    --cyan: #37d9ff;
    --teal: #19c7b6;
    --gold: #f4b93f;
    --green: #57d886;
    --yellow: #ffe066;
    --text: #e8f6ff;
    --muted: #96b8d2;
    --danger: #ff5b73;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(55,217,255,.10), transparent 30%),
        radial-gradient(circle at top right, rgba(244,185,63,.08), transparent 24%),
        linear-gradient(135deg, #050d18 0%, #081827 48%, #071d22 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2rem;
    max-width: 1560px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #05101d 0%, #07213b 52%, #07393b 100%);
    border-right: 1px solid rgba(55,217,255,.18);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: var(--text) !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(91,198,255,.16);
    border-radius: 18px;
    padding: 10px 12px;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: rgba(6,17,31,.90) !important;
    border: 1px solid rgba(91,198,255,.22) !important;
    border-radius: 13px !important;
    color: #ffffff !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.02), 0 8px 20px rgba(0,0,0,.22);
}

section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stSlider p {
    color: var(--text) !important;
}

/* Header */
.header-card {
    background:
        linear-gradient(135deg, rgba(9,32,58,.92), rgba(6,17,31,.96));
    border: 1px solid rgba(55,217,255,.22);
    border-radius: 22px;
    padding: 12px;
    margin-bottom: 16px;
    box-shadow: 0 16px 42px rgba(0,0,0,.38);
}

.header-card img {
    border-radius: 18px;
    border: 1px solid rgba(55,217,255,.20);
}

/* Hero */
.dark-hero {
    background:
        linear-gradient(135deg, rgba(11,32,53,.96), rgba(6,17,31,.96));
    border: 1px solid rgba(91,198,255,.18);
    border-radius: 22px;
    padding: 20px 22px;
    margin: 4px 0 16px 0;
    box-shadow: 0 16px 38px rgba(0,0,0,.32);
}

.dark-hero h1 {
    color: #ffffff;
    font-size: 35px;
    font-weight: 900;
    margin: 0 0 6px;
    letter-spacing: -0.7px;
}

.dark-hero p {
    color: var(--muted);
    font-size: 16px;
    font-weight: 600;
    margin: 0;
}

.top-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}

.top-card {
    background:
        linear-gradient(135deg, rgba(14,42,68,.94), rgba(6,17,31,.94));
    border: 1px solid rgba(91,198,255,.18);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 12px 32px rgba(0,0,0,.25);
}

.top-card .label {
    color: var(--muted);
    font-size: 14px;
    font-weight: 700;
}

.top-card .value {
    color: var(--cyan);
    font-size: 31px;
    font-weight: 900;
    line-height: 1.15;
}

.top-card.yellow .value { color: var(--yellow); }
.top-card.green .value { color: var(--green); }

/* Image and upload */
.stFileUploader section {
    background: rgba(11,32,53,.78);
    border: 1.6px dashed rgba(55,217,255,.35);
    border-radius: 18px;
    padding: 15px;
}

.stImage img {
    border-radius: 17px;
    border: 1px solid rgba(91,198,255,.16);
    box-shadow: 0 18px 35px rgba(0,0,0,.32);
}

/* Result cards */
.result-box {
    background:
        linear-gradient(135deg, rgba(8,36,55,.94), rgba(5,13,24,.96));
    border: 1px solid rgba(55,217,255,.25);
    border-left: 7px solid var(--cyan);
    padding: 18px;
    border-radius: 18px;
    color: var(--cyan);
    font-weight: 900;
    font-size: 34px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 14px 30px rgba(0,0,0,.30);
}

.result-box-dark {
    background:
        linear-gradient(135deg, rgba(9,52,76,.96), rgba(5,13,24,.98));
    border: 1px solid rgba(55,217,255,.35);
    padding: 20px 18px;
    border-radius: 20px;
    color: var(--cyan);
    font-weight: 900;
    font-size: 39px;
    text-align: center;
    margin: 13px 0;
    box-shadow: 0 0 28px rgba(55,217,255,.12);
}

.metric-card {
    background:
        linear-gradient(135deg, rgba(11,32,53,.96), rgba(5,13,24,.96));
    border: 1px solid rgba(91,198,255,.18);
    border-radius: 18px;
    padding: 17px 10px;
    text-align: center;
    box-shadow: 0 12px 25px rgba(0,0,0,.26);
    margin-bottom: 12px;
}

.metric-card.yellow {
    border-color: rgba(255,224,102,.34);
}

.metric-card.green {
    border-color: rgba(87,216,134,.34);
}

.metric-card .value {
    font-size: 38px;
    font-weight: 900;
    color: var(--cyan);
    line-height: 1;
}

.metric-card.yellow .value {
    color: var(--yellow);
}

.metric-card.green .value {
    color: var(--green);
}

.metric-card .label {
    font-size: 12px;
    color: var(--muted);
    font-weight: 900;
    letter-spacing: .4px;
    margin-top: 8px;
}

/* Buttons */
.stButton > button {
    background:
        linear-gradient(90deg, #c58b20, #f4b93f, #b97810);
    color: #09111f;
    border-radius: 13px;
    border: 1px solid rgba(255,224,102,.45);
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 900;
    width: 100%;
    box-shadow: 0 12px 26px rgba(244,185,63,.22);
    transition: all .18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 34px rgba(244,185,63,.28);
    color: #050d18;
}

.success-box {
    background: rgba(25,199,182,.14);
    border: 1px solid rgba(25,199,182,.32);
    border-left: 7px solid var(--teal);
    padding: 13px;
    border-radius: 14px;
    color: #e8fffb;
    font-weight: 800;
    font-size: 15px;
}

.info-panel {
    background: rgba(11,32,53,.82);
    border: 1px solid rgba(91,198,255,.16);
    border-radius: 18px;
    padding: 14px;
    color: var(--muted);
    font-size: 14px;
    font-weight: 700;
    margin: 10px 0;
}

.small-note {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
}

h1, h2, h3 {
    color: var(--text) !important;
    font-weight: 900 !important;
}

div[data-testid="stExpander"] {
    background: rgba(11,32,53,.85);
    border: 1px solid rgba(91,198,255,.16);
    border-radius: 16px;
}

@media(max-width: 900px) {
    .top-strip {
        grid-template-columns: 1fr;
    }
    .dark-hero h1 {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="header-card">', unsafe_allow_html=True)

if HEADER_LOGO.exists():
    st.image(str(HEADER_LOGO), use_container_width=True)
else:
    st.markdown("""
    <div class="dark-hero">
        <h1>💧 AquaCount</h1>
        <p>FastCount PL & FastCount Bacteri</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 📌 Menu")
    menu = st.radio(
        "Pilih aplikasi",
        [
            "🦐 FastCount PL",
            "🧫 FastCount Bacteri",
        ]
    )


# ============================================================
# PAGE: FASTCOUNT PL
# ============================================================

if menu == "🦐 FastCount PL":
    with st.sidebar:
        st.markdown("## 📋 Input Sampel PL")
        unit = st.text_input("Unit Hatchery", "Makassar", key="pl_unit")
        tank = st.text_input("Nomor Tank", key="pl_tank")
        umur_pl = st.text_input("Umur PL", "PL10", key="pl_umur")
        operator = st.text_input("Operator", key="pl_operator")

        st.markdown("## ⚙️ Parameter PL")
        threshold = st.slider("Threshold", 0, 255, 120, key="pl_threshold")
        min_area = st.slider("Min Area", 1, 500, 15, key="pl_min_area")
        max_area = st.slider("Max Area", 10, 5000, 700, key="pl_max_area")
        blur = st.slider("Blur", 1, 21, 5, step=2, key="pl_blur")

    st.markdown("""
    <div class="dark-hero">
        <h1>🦐 FastCount PL</h1>
        <p>Analisa cepat jumlah PL dari foto sampel dengan deteksi otomatis, koreksi manual, dan penyimpanan data.</p>
    </div>
    <div class="top-strip">
        <div class="top-card"><div class="label">Mode Analisa</div><div class="value">PL Count</div></div>
        <div class="top-card"><div class="label">Metode</div><div class="value">Image Detection</div></div>
        <div class="top-card"><div class="label">Output</div><div class="value">CSV + Sheet</div></div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Foto PL",
        type=["jpg", "jpeg", "png"],
        key="upload_pl"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)

        result_img, thresh_img, count = detect_pl(
            image_rgb,
            threshold,
            min_area,
            max_area,
            blur
        )

        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 0.9])

        with col1:
            st.subheader("📷 1. Foto Asli")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("🎯 2. Hasil Deteksi")
            st.image(result_img, use_container_width=True)

        with col3:
            st.subheader("🧠 3. Mask")
            st.image(thresh_img, use_container_width=True)

        with col4:
            st.subheader("📊 4. Hasil")
            st.markdown(f"""
            <div class="result-box">
                {count}<br>
                <span style="font-size:14px;color:#96b8d2;">Jumlah Deteksi</span>
            </div>
            """, unsafe_allow_html=True)

            koreksi = st.number_input(
                "Koreksi Manual",
                min_value=0,
                value=int(count),
                key="pl_koreksi"
            )

            catatan = st.text_area("Catatan", height=90, key="pl_catatan")

            if st.button("💾 SIMPAN HASIL PL"):
                row = {
                    "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "unit": unit,
                    "tank": tank,
                    "umur_pl": umur_pl,
                    "operator": operator,
                    "hasil_deteksi": int(count),
                    "hasil_koreksi": int(koreksi),
                    "catatan": catatan
                }

                append_to_csv(DATA_FILE_PL, row)
                ok, msg = safe_save_to_google(save_pl_to_google_sheet, row)

                st.markdown(f"""
                <div class="success-box">
                    ✅ {msg}
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-panel">
            <b>📌 Cara Pakai FastCount PL</b><br>
            1. Isi data sampel di sidebar.<br>
            2. Upload foto PL.<br>
            3. Cek foto asli, hasil deteksi, dan mask.<br>
            4. Koreksi manual bila perlu.<br>
            5. Klik <b>SIMPAN HASIL PL</b>.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: FASTCOUNT BACTERI
# ============================================================

elif menu == "🧫 FastCount Bacteri":
    with st.sidebar:
        st.markdown("## 📋 Input Sampel")
        unit_b = st.text_input("Unit Hatchery", "Makassar", key="bact_unit")
        kode_sampel = st.text_input("Kode Sampel", key="bact_kode")
        operator_b = st.text_input("Operator", key="bact_operator")

        st.markdown("## 🧫 Parameter Bacteri")
        jenis_media = st.selectbox(
            "Jenis Media",
            ["TCBS Vibrio", "TPC / Umum"],
            key="bact_media"
        )
        pengenceran_label = st.selectbox(
            "Pengenceran",
            [
                "10⁰ / tanpa pengenceran",
                "10⁻¹",
                "10⁻²",
                "10⁻³",
                "10⁻⁴",
                "10⁻⁵",
                "10⁻⁶",
            ],
            index=3,
            key="bact_dilution"
        )
        volume_tanam = st.number_input(
            "Volume Tanam (mL)",
            min_value=0.01,
            value=0.10,
            step=0.01,
            key="bact_volume"
        )

        st.markdown("## ⚙️ Setting Deteksi")
        bact_threshold = st.slider("Ambang Deteksi", 5, 120, 45, key="bact_threshold")
        bact_min_area = st.slider("Ukuran Min", 3, 300, 10, key="bact_min_area")
        bact_max_area = st.slider("Ukuran Maks", 40, 5000, 950, key="bact_max_area")
        bact_blur = st.slider("Blur", 1, 21, 5, step=2, key="bact_blur")
        petri_margin = st.slider("Area Cawan", 0.50, 1.00, 0.90, step=0.01, key="petri_margin")

    st.markdown("""
    <div class="dark-hero">
        <h1>🧫 FastCount Bacteri</h1>
        <p>Hitung koloni bakteri secara otomatis dan konversi ke nilai CFU/mL.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_bacteria = st.file_uploader(
        "📤 Upload Foto Cawan Petri",
        type=["jpg", "jpeg", "png"],
        key="upload_bacteria"
    )

    if uploaded_bacteria:
        image = Image.open(uploaded_bacteria).convert("RGB")
        image_rgb = np.array(image)

        result_img, mask_img, counts = detect_bacteri_colonies(
            image_rgb=image_rgb,
            media_type=jenis_media,
            threshold=bact_threshold,
            min_area=bact_min_area,
            max_area=bact_max_area,
            blur=bact_blur,
            petri_margin=petri_margin
        )

        total_colony = int(counts["total"])
        dilution_factor = dilution_label_to_factor(pengenceran_label)
        cfu_ml = total_colony * dilution_factor / float(volume_tanam)

        st.markdown(f"""
        <div class="top-strip">
            <div class="top-card yellow">
                <div class="label">Koloni Kuning</div>
                <div class="value">{counts["yellow"]}</div>
            </div>
            <div class="top-card green">
                <div class="label">Koloni Hijau</div>
                <div class="value">{counts["green"]}</div>
            </div>
            <div class="top-card">
                <div class="label">Estimasi CFU/mL</div>
                <div class="value">{format_cfu(cfu_ml)} CFU/mL</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1.15, 1.15, 0.9])

        with col1:
            st.subheader("📷 1. Foto Asli")
            st.image(image_rgb, use_container_width=True)
            st.markdown(f"""
            <div class="info-panel">
                File: <b>{uploaded_bacteria.name}</b>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("🎯 2. Hasil Deteksi Koloni")
            st.image(result_img, use_container_width=True)
            st.markdown(f"""
            <div class="info-panel">
                🟡 Kuning ({counts["yellow"]}) &nbsp;&nbsp; 🟢 Hijau ({counts["green"]})
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.subheader("📊 3. Hasil Hitung")

            m1, m2 = st.columns(2)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{total_colony}</div>
                    <div class="label">TOTAL KOLONI</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card yellow">
                    <div class="value">{counts["yellow"]}</div>
                    <div class="label">KUNING</div>
                </div>
                """, unsafe_allow_html=True)

            m3, m4 = st.columns(2)

            with m3:
                st.markdown(f"""
                <div class="metric-card green">
                    <div class="value">{counts["green"]}</div>
                    <div class="label">HIJAU</div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                other_label = counts["other"] if jenis_media == "TPC / Umum" else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{other_label}</div>
                    <div class="label">LAINNYA</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-box-dark">
                <span style="font-size:16px;color:#96b8d2;">Estimasi CFU/mL</span><br>
                {format_cfu(cfu_ml)} CFU/mL
            </div>
            """, unsafe_allow_html=True)

            if total_colony < 30:
                st.warning("Koloni <30: hasil sebaiknya dicatat sebagai estimasi.")
            elif total_colony <= 300:
                st.success("Status validasi: rentang ideal 30–300 koloni.")
            else:
                st.error("Koloni >300: berisiko TNTC/confluent.")

            koreksi_bakteri = st.number_input(
                "Koreksi Manual Total Koloni",
                min_value=0,
                value=int(total_colony),
                key="bact_koreksi"
            )

            cfu_koreksi = int(koreksi_bakteri) * dilution_factor / float(volume_tanam)
            st.info(f"Hasil koreksi: {format_cfu(cfu_koreksi)} CFU/mL")

            catatan_b = st.text_area("Catatan", height=90, key="bact_catatan")

            if st.button("💾 SIMPAN HASIL BACTERI"):
                row = {
                    "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "unit": unit_b,
                    "kode_sampel": kode_sampel,
                    "operator": operator_b,
                    "jenis_media": jenis_media,
                    "pengenceran_label": pengenceran_label,
                    "volume_tanam_ml": float(volume_tanam),
                    "koloni_kuning": int(counts["yellow"]),
                    "koloni_hijau": int(counts["green"]),
                    "koloni_lainnya": int(other_label),
                    "total_koloni": int(koreksi_bakteri),
                    "cfu_ml": float(cfu_koreksi),
                    "catatan": catatan_b,
                }

                append_to_csv(DATA_FILE_BACTERI, row)
                ok, msg = safe_save_to_google(save_bacteri_to_google_sheet, row)

                st.markdown(f"""
                <div class="success-box">
                    ✅ {msg}
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel">
            <b>Kode Sampel:</b> {kode_sampel or "-"} &nbsp; | &nbsp;
            <b>Media:</b> {jenis_media} &nbsp; | &nbsp;
            <b>Pengenceran:</b> {pengenceran_label} &nbsp; | &nbsp;
            <b>Volume Tanam:</b> {volume_tanam} mL &nbsp; | &nbsp;
            <b>Ambang Deteksi:</b> {bact_threshold} &nbsp; | &nbsp;
            <b>Ukuran Min:</b> {bact_min_area}px &nbsp; | &nbsp;
            <b>Ukuran Maks:</b> {bact_max_area}px
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🧠 Lihat Mask Deteksi"):
            st.image(mask_img, use_container_width=True)

    else:
        st.markdown("""
        <div class="info-panel">
            <b>📌 Cara Pakai FastCount Bacteri</b><br>
            1. Isi unit, kode sampel, operator, media, pengenceran, dan volume tanam.<br>
            2. Upload foto cawan petri.<br>
            3. Cek hasil deteksi koloni kuning dan hijau.<br>
            4. Koreksi manual bila perlu.<br>
            5. Klik <b>SIMPAN HASIL BACTERI</b>.
        </div>
        """, unsafe_allow_html=True)
