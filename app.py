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
DATA_FILE_BENUR = Path("hasil_hitung_aquacount.csv")
DATA_FILE_BAKTERI = Path("hasil_hitung_bakteri.csv")
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

def save_benur_to_google_sheet(row):
    client = get_gsheet_client()
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1

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

def save_bacteria_to_google_sheet(row):
    """
    Menyimpan data bacteri.
    Catatan:
    - Fungsi ini mencoba menyimpan ke worksheet bernama 'FastCount Bacteri'.
    - Jika worksheet belum ada, otomatis dibuat.
    """
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
            "jenis_media",
            "pengenceran",
            "volume_tanam_ml",
            "koloni_kuning",
            "koloni_hijau",
            "total_koloni",
            "cfu_ml",
            "operator",
            "catatan",
        ])

    sheet.append_row([
        row["tanggal"],
        row["unit"],
        row["kode_sampel"],
        row["jenis_media"],
        row["pengenceran_label"],
        row["volume_tanam_ml"],
        row["koloni_kuning"],
        row["koloni_hijau"],
        row["total_koloni"],
        row["cfu_ml"],
        row["operator"],
        row["catatan"],
    ])

# ============================================================
# BENUR COUNTER
# ============================================================

def detect_benur(image_rgb, threshold, min_area, max_area, blur):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
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

            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 180, 120), 2)
            cv2.putText(
                result,
                str(count),
                (x, max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 90, 255),
                1
            )

    return result, thresh, count

# ============================================================
# BACTERIA / COLONY COUNTER
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

def classify_tcbs_colony(mean_bgr):
    """
    Klasifikasi sederhana koloni TCBS.
    Return:
    - yellow
    - green
    - None untuk objek tidak jelas/noise. None tidak ditampilkan.
    """
    b, g, r = mean_bgr

    # Kuning TCBS: R dan G kuat, B rendah
    if r > 105 and g > 90 and b < 120 and r >= b * 1.15:
        return "yellow"

    # Hijau TCBS: G relatif dominan, R tidak terlalu tinggi
    if g > 80 and g >= r * 0.85 and g >= b * 1.05:
        return "green"

    return None

def detect_bacteria_colonies(
    image_rgb,
    media_type="TCBS Vibrio",
    threshold=45,
    min_area=10,
    max_area=950,
    blur=5,
    petri_margin=0.90
):
    """
    Deteksi koloni bakteri berbasis kontras dan warna.
    Untuk TCBS Vibrio:
    - Hanya titik kuning dan hijau yang ditampilkan/dihitung.
    - Titik merah/unclassified tidak ditampilkan agar tidak membingungkan.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = image_rgb.copy()

    h, w = image_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * petri_margin / 2)

    # Mask area cawan petri berbentuk lingkaran di tengah foto.
    petri_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(petri_mask, (cx, cy), radius, 255, -1)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if blur % 2 == 0:
        blur += 1
    gray_blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    # Estimasi background dari area cawan.
    petri_pixels = gray_blur[petri_mask == 255]
    bg = np.percentile(petri_pixels, 70) if len(petri_pixels) else 180

    # Koloni biasanya lebih gelap atau lebih kontras dari background.
    dark_mask = ((bg - gray_blur) > threshold).astype(np.uint8) * 255

    # Saturation mask untuk koloni berwarna.
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

    # Gambar lingkaran area cawan
    cv2.circle(result, (cx, cy), radius, (0, 210, 210), 3)

    number = 0

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
        mean_bgr = cv2.mean(image_bgr, mask=mask)[:3]

        if media_type == "TCBS Vibrio":
            colony_type = classify_tcbs_colony(mean_bgr)

            # Objek tidak jelas tidak ditampilkan dan tidak dihitung.
            if colony_type is None:
                continue

            if colony_type == "yellow":
                color = (255, 205, 0)       # RGB kuning
                yellow_count += 1
            else:
                color = (20, 180, 90)       # RGB hijau
                green_count += 1

        else:
            colony_type = "other"
            color = (0, 180, 255)           # RGB biru
            other_count += 1

        number += 1
        total_count += 1

        cv2.circle(result, (px, py), 9, color, -1)
        cv2.circle(result, (px, py), 9, (255, 255, 255), 2)
        cv2.putText(
            result,
            str(number),
            (px + 9, py - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )
        cv2.putText(
            result,
            str(number),
            (px + 9, py - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (5, 60, 90),
            1
        )

    return result, combined, {
        "yellow": yellow_count,
        "green": green_count,
        "other": other_count,
        "total": total_count,
    }

def format_cfu(value):
    if value >= 100000:
        return f"{value:.2e}".replace("e+", " × 10^")
    return f"{value:,.0f}".replace(",", ".")

# ============================================================
# PAGE CONFIG + CSS
# ============================================================

st.set_page_config(
    page_title="AquaCount",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    max-width: 100%;
}

.stApp {
    background: linear-gradient(135deg, #f5fbff 0%, #ffffff 45%, #eafffb 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0646b8 0%, #078ed1 55%, #10bfae 100%);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p {
    color: white !important;
}

.logo-card {
    background: white;
    padding: 18px;
    border-radius: 22px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.12);
    margin-bottom: 18px;
}

.mini-card {
    background: white;
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.07);
    height: 100%;
}

.result-box {
    background: #eafffb;
    border-left: 7px solid #11d3c5;
    padding: 14px;
    border-radius: 14px;
    color: #0646b8;
    font-weight: 800;
    font-size: 28px;
    text-align: center;
    margin-bottom: 10px;
}

.result-box-dark {
    background: linear-gradient(135deg, #0646b8, #078ed1);
    padding: 18px;
    border-radius: 18px;
    color: white;
    font-weight: 900;
    font-size: 32px;
    text-align: center;
    margin-bottom: 10px;
}

.success-box {
    background: #e6fff9;
    border-left: 7px solid #10bfae;
    padding: 12px;
    border-radius: 12px;
    color: #0646b8;
    font-weight: 700;
    font-size: 15px;
}

.metric-card {
    background: white;
    border: 1px solid #d8edf7;
    border-radius: 18px;
    padding: 14px;
    text-align: center;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
}

.metric-card .value {
    font-size: 34px;
    font-weight: 900;
    color: #0646b8;
}

.metric-card .label {
    font-size: 13px;
    color: #4a6680;
    font-weight: 800;
}

.stButton > button {
    background-color: #10bfae;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
}

.small-note {
    color: #557089;
    font-size: 13px;
    line-height: 1.45;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="logo-card">', unsafe_allow_html=True)

if HEADER_LOGO.exists():
    st.image(str(HEADER_LOGO), use_container_width=True)
else:
    st.title("💧 AquaCount")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:
    st.header("📌 Menu")
    menu = st.radio(
        "Pilih aplikasi",
        [
            "🦐 FastCount PL",
            "🧫 FastCount Bacteri",
        ]
    )

# ============================================================
# PAGE: FASTCOUNT BENUR
# ============================================================

if menu == "🦐 FastCount PL":
    with st.sidebar:
        st.header("📋 Data Sampling")
        unit = st.text_input("Unit Hatchery", "Makassar", key="benur_unit")
        tank = st.text_input("Nomor Tank", key="benur_tank")
        umur_pl = st.text_input("Umur PL", "PL10", key="benur_umur")
        operator = st.text_input("Operator", key="benur_operator")

        st.header("⚙️ Parameter Benur")
        threshold = st.slider("Threshold", 0, 255, 120, key="benur_threshold")
        min_area = st.slider("Min Area", 1, 500, 15, key="benur_min_area")
        max_area = st.slider("Max Area", 10, 5000, 700, key="benur_max_area")
        blur = st.slider("Blur", 1, 21, 5, step=2, key="benur_blur")

    st.subheader("🦐 FastCount PL")

    uploaded_file = st.file_uploader(
        "📤 Upload Foto Benur",
        type=["jpg", "jpeg", "png"],
        key="upload_benur"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)

        result_img, thresh_img, count = detect_benur(
            image_rgb,
            threshold,
            min_area,
            max_area,
            blur
        )

        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 0.9])

        with col1:
            st.subheader("📷 Foto Asli")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("🎯 Hasil Deteksi")
            st.image(result_img, use_container_width=True)

        with col3:
            st.subheader("🧠 Mask")
            st.image(thresh_img, use_container_width=True)

        with col4:
            st.markdown(f"""
            <div class="result-box">
                Jumlah<br>{count}
            </div>
            """, unsafe_allow_html=True)

            koreksi = st.number_input(
                "Koreksi Manual",
                min_value=0,
                value=int(count),
                key="benur_koreksi"
            )

            catatan = st.text_area("Catatan", height=90, key="benur_catatan")

            if st.button("💾 Simpan Data PL"):
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

                if DATA_FILE_BENUR.exists():
                    df = pd.read_csv(DATA_FILE_BENUR)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])

                df.to_csv(DATA_FILE_BENUR, index=False)
                save_benur_to_google_sheet(row)

                st.markdown("""
                <div class="success-box">
                    ✅ Data PL tersimpan ke Google Sheets
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="mini-card">
            <b>📌 Cara Pakai FastCount PL:</b><br>
            1. Isi data sampling di sidebar.<br>
            2. Upload foto benur.<br>
            3. Cek hasil deteksi.<br>
            4. Koreksi manual bila perlu.<br>
            5. Klik Simpan Data PL.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE: BACTERIA COUNTER
# ============================================================

elif menu == "🧫 FastCount Bacteri":
    with st.sidebar:
        st.header("📋 Data Sampling Bakteri")
        unit_b = st.text_input("Unit Hatchery", "Makassar", key="bact_unit")
        kode_sampel = st.text_input("Kode Sampel", key="bact_kode")
        operator_b = st.text_input("Operator", key="bact_operator")

        st.header("🧫 Parameter Bakteri")
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
            "Volume tanam (mL)",
            min_value=0.01,
            value=0.10,
            step=0.01,
            key="bact_volume"
        )

        st.header("⚙️ Setting Deteksi")
        bact_threshold = st.slider("Ambang Deteksi", 5, 120, 45, key="bact_threshold")
        bact_min_area = st.slider("Ukuran Koloni Minimum", 3, 300, 10, key="bact_min_area")
        bact_max_area = st.slider("Ukuran Koloni Maksimum", 40, 5000, 950, key="bact_max_area")
        bact_blur = st.slider("Blur", 1, 21, 5, step=2, key="bact_blur")
        petri_margin = st.slider("Area Cawan", 0.50, 1.00, 0.90, step=0.01, key="petri_margin")

    st.subheader("🧫 FastCount Bacteri")

    uploaded_bacteria = st.file_uploader(
        "📤 Upload Foto Cawan Petri",
        type=["jpg", "jpeg", "png"],
        key="upload_bacteria"
    )

    if uploaded_bacteria:
        image = Image.open(uploaded_bacteria).convert("RGB")
        image_rgb = np.array(image)

        result_img, mask_img, counts = detect_bacteria_colonies(
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

        col1, col2, col3 = st.columns([1.25, 1.25, 0.9])

        with col1:
            st.subheader("📷 Foto Asli")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("🎯 Hasil Deteksi Koloni")
            st.image(result_img, use_container_width=True)

        with col3:
            st.subheader("📊 Hasil Hitung")

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
                <div class="metric-card">
                    <div class="value">{counts["yellow"]}</div>
                    <div class="label">KUNING</div>
                </div>
                """, unsafe_allow_html=True)

            m3, m4 = st.columns(2)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{counts["green"]}</div>
                    <div class="label">HIJAU</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                if jenis_media == "TPC / Umum":
                    other_label = counts["other"]
                else:
                    other_label = 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{other_label}</div>
                    <div class="label">LAINNYA</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-box-dark">
                {format_cfu(cfu_ml)}<br>
                <span style="font-size:18px;">CFU/mL</span>
            </div>
            """, unsafe_allow_html=True)

            if total_colony < 30:
                st.warning("Koloni <30: hasil sebaiknya dicatat sebagai estimasi atau gunakan pengenceran lebih rendah.")
            elif total_colony <= 300:
                st.success("Rentang ideal 30–300 koloni.")
            else:
                st.error("Koloni >300: berisiko TNTC/confluent, sebaiknya pengenceran ulang.")

            koreksi_bakteri = st.number_input(
                "Koreksi Manual Total Koloni",
                min_value=0,
                value=int(total_colony),
                key="bact_koreksi"
            )

            cfu_koreksi = int(koreksi_bakteri) * dilution_factor / float(volume_tanam)
            st.info(f"Hasil koreksi: {format_cfu(cfu_koreksi)} CFU/mL")

            catatan_b = st.text_area("Catatan", height=90, key="bact_catatan")

            if st.button("💾 Simpan Data Bacteri"):
                row = {
                    "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "unit": unit_b,
                    "kode_sampel": kode_sampel,
                    "jenis_media": jenis_media,
                    "pengenceran_label": pengenceran_label,
                    "volume_tanam_ml": float(volume_tanam),
                    "koloni_kuning": int(counts["yellow"]),
                    "koloni_hijau": int(counts["green"]),
                    "total_koloni": int(koreksi_bakteri),
                    "cfu_ml": float(cfu_koreksi),
                    "operator": operator_b,
                    "catatan": catatan_b,
                }

                if DATA_FILE_BAKTERI.exists():
                    df = pd.read_csv(DATA_FILE_BAKTERI)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])

                df.to_csv(DATA_FILE_BAKTERI, index=False)
                save_bacteria_to_google_sheet(row)

                st.markdown("""
                <div class="success-box">
                    ✅ Data bacteri tersimpan ke Google Sheets
                </div>
                """, unsafe_allow_html=True)

        with st.expander("🧠 Lihat Mask Deteksi"):
            st.image(mask_img, use_container_width=True)

    else:
        st.markdown("""
        <div class="mini-card">
            <b>📌 Cara Pakai FastCount Bacteri:</b><br>
            1. Pilih menu <b>FastCount Bacteri</b> di sidebar.<br>
            2. Isi unit, kode sampel, operator, media, pengenceran, dan volume tanam.<br>
            3. Upload foto cawan petri.<br>
            4. Cek hasil deteksi koloni kuning dan hijau.<br>
            5. Koreksi manual bila perlu.<br>
            6. Klik Simpan Data Bacteri.
        </div>
        """, unsafe_allow_html=True)
