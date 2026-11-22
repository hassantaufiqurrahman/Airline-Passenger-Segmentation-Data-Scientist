# Import Library
from datetime import date, datetime
import os
import pickle
import tempfile

from fpdf import FPDF
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Airline Passenger Segmentation",
    page_icon="✈️",
    layout="wide",
)

# Nilai tetap LOAD_TIME
LOAD_TIME_FIXED = date(2014, 3, 31)

# Menentukan Nilai Minimum dan Maksimum untuk Masing-masing Feature Input Data
FFP_DATE_MIN = date(2004, 11, 1)
FFP_DATE_MAX = date(2013, 3, 31)

FIRST_FLIGHT_MIN = date(1905, 12, 31)
FIRST_FLIGHT_MAX = date(2015, 5, 30)

LAST_FLIGHT_MIN = date(2012, 4, 1)
LAST_FLIGHT_MAX = date(2014, 3, 31)

FLIGHT_COUNT_MIN = 2
FLIGHT_COUNT_MAX = 213

LAST_TO_END_MIN = 1.0
LAST_TO_END_MAX = 731.0

SEG_KM_SUM_MIN = 368.0
SEG_KM_SUM_MAX = 580717.0

AVG_DISCOUNT_MIN = 0.0
AVG_DISCOUNT_MAX = 1.5

FFP_TIER_OPTIONS = [4, 5, 6]

CLUSTER_PROFILES = {
    0: {
        "name": "Senior Low Frequency",
        "description": (
            "Masa keanggotaan terpanjang, namun frekuensi penerbangan paling rendah "
            "dengan jarak antar penerbangan cukup renggang."
        ),
    },
    1: {
        "name": "Dormant High Frequency",
        "description": (
            "Frekuensi penerbangan sangat tinggi dengan interval antar penerbangan tersingkat "
            "dan total jarak tempuh relatif pendek, tetapi sudah lama tidak terbang dengan "
            "penurunan nilai transaksi pada Tahun ke-2."
        ),
    },
    2: {
        "name": "VIP Flyers",
        "description": (
            "Klaster bernilai tertinggi dengan total jarak tempuh terjauh, koefisien diskon tertinggi, "
            "aktivitas terbang paling baru, serta nilai transaksi terbesar dan terus bertumbuh."
        ),
    },
    3: {
        "name": "Churn-Risk",
        "description": (
            "Klaster dengan populasi terbesar yang memiliki masa keanggotaan relatif singkat, "
            "jarak tempuh rendah, interval antar penerbangan paling lama, serta keterlibatan "
            "dan nilai transaksi yang menurun."
        ),
    },
    4: {
        "name": "Loyal Active Growth Flyers",
        "description": (
            "Pelanggan aktif yang baru saja terbang dengan masa keanggotaan menengah, "
            "frekuensi dan jarak tempuh menengah tinggi, serta pertumbuhan nilai transaksi "
            "yang konsisten dari Tahun ke-1 ke Tahun ke-2."
        ),
    },
}

EMPTY_INPUTS = {
    "customer_name": "",
    "ffp_date": None,
    "first_flight_date": None,
    "last_flight_date": None,
    "flight_count": None,
    "last_to_end": None,
    "seg_km_sum": None,
    "avg_discount": None,
    "ffp_tier": None,
}


def get_cluster_info(cluster_id: int) -> dict:
    return CLUSTER_PROFILES.get(
        int(cluster_id),
        {
            "name": f"Cluster {cluster_id}",
            "description": "Karakteristik klaster hasil segmentasi K-Means.",
        },
    )


@st.cache_resource
def load_artifacts(pickle_dir: str = "pickle"):
    with open(os.path.join(pickle_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(pickle_dir, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    with open(os.path.join(pickle_dir, "profiling_data.pkl"), "rb") as f:
        profiling_data = pickle.load(f)
    return scaler, kmeans, profiling_data


def init_state():
    if "form_values" not in st.session_state:
        st.session_state.form_values = EMPTY_INPUTS.copy()
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
    if "is_editing" not in st.session_state:
        st.session_state.is_editing = True
    if "predicted_cluster" not in st.session_state:
        st.session_state.predicted_cluster = None
    if "submitted_inputs" not in st.session_state:
        st.session_state.submitted_inputs = {}
    if "submitted_display" not in st.session_state:
        st.session_state.submitted_display = {}
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "pdf_bytes_cache" not in st.session_state:
        st.session_state.pdf_bytes_cache = None
    if "pdf_error_msg" not in st.session_state:
        st.session_state.pdf_error_msg = ""


def reset_all_data():
    st.session_state.form_values = EMPTY_INPUTS.copy()
    st.session_state.has_prediction = False
    st.session_state.is_editing = True
    st.session_state.predicted_cluster = None
    st.session_state.submitted_inputs = {}
    st.session_state.submitted_display = {}
    st.session_state.customer_name = ""
    st.session_state.pdf_bytes_cache = None
    st.session_state.pdf_error_msg = ""


def normalize_crosstab_row(row_series: pd.Series) -> pd.Series:
    s = row_series.copy().astype(float)
    if (s < 0).any():
        s = s + 100.0
    return s.round(2)


# Bagian Input Data Pelanggan
def render_input_section():
    st.subheader("📝 Input Data Pelanggan")

    with st.form("customer_input_form"):
        fv = st.session_state.form_values

        customer_name_val = st.text_input(
            "Nama Pelanggan",
            value=fv.get("customer_name", ""),
            placeholder="Contoh: Budi Santoso",
        )

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            ffp_date_val = st.date_input(
                "Tanggal Bergabung Member Frequent Flayer",
                value=fv.get("ffp_date"),
                min_value=FFP_DATE_MIN,
                max_value=FFP_DATE_MAX,
                format="YYYY-MM-DD",
                help=f"Pilih tanggal antara {FFP_DATE_MIN} s.d. {FFP_DATE_MAX}. Contoh: 2010-12-17",
            )
            first_flight_val = st.date_input(
                "Tanggal Penerbangan Pertama",
                value=fv.get("first_flight_date"),
                min_value=FIRST_FLIGHT_MIN,
                max_value=FIRST_FLIGHT_MAX,
                format="YYYY-MM-DD",
                help=f"Pilih tanggal antara {FIRST_FLIGHT_MIN} s.d. {FIRST_FLIGHT_MAX}. Contoh: 2011-01-10",
            )
            last_flight_val = st.date_input(
                "Tanggal Penerbangan Terakhir",
                value=fv.get("last_flight_date"),
                min_value=LAST_FLIGHT_MIN,
                max_value=LAST_FLIGHT_MAX,
                format="YYYY-MM-DD",
                help=f"Pilih tanggal antara {LAST_FLIGHT_MIN} s.d. {LAST_FLIGHT_MAX}. Contoh: 2014-03-01",
            )

        with c2:
            flight_count_val = st.number_input(
                "Jumlah Penerbangan",
                min_value=FLIGHT_COUNT_MIN,
                max_value=FLIGHT_COUNT_MAX,
                value=fv.get("flight_count"),
                step=1,
                placeholder="Contoh: 25",
            )
            last_to_end_val = st.number_input(
                "Jarak Tanggal Penerbangan Terakhir dengan Tanggal Penarikan Data (Hari)",
                min_value=LAST_TO_END_MIN,
                max_value=LAST_TO_END_MAX,
                value=fv.get("last_to_end"),
                step=1.0,
                placeholder="Contoh: 30",
            )
            seg_km_sum_val = st.number_input(
                "Total Jarak Penerbangan (km)",
                min_value=SEG_KM_SUM_MIN,
                max_value=SEG_KM_SUM_MAX,
                value=fv.get("seg_km_sum"),
                step=500.0,
                placeholder="Contoh: 15000",
            )

        with c3:
            avg_discount_val = st.number_input(
                "Rata-rata Discount (%)",
                min_value=AVG_DISCOUNT_MIN,
                max_value=AVG_DISCOUNT_MAX,
                value=fv.get("avg_discount"),
                step=0.01,
                format="%.4f",
                placeholder="Contoh: 0.72",
            )
            current_tier = fv.get("ffp_tier")
            tier_idx = (
                FFP_TIER_OPTIONS.index(int(current_tier))
                if current_tier in FFP_TIER_OPTIONS
                else None
            )
            ffp_tier_val = st.selectbox(
                "Tingkat Member Frequent Flyer",
                options=FFP_TIER_OPTIONS,
                index=tier_idx,
                placeholder="Contoh: 4",
            )

        submit_btn = st.form_submit_button(
            "🔍 Prediksi Cluster", use_container_width=True, type="primary"
        )

    if submit_btn:
        load_time_dt = LOAD_TIME_FIXED

        missing_fields = []
        if not customer_name_val or not customer_name_val.strip():
            missing_fields.append("Nama Pelanggan")
        if ffp_date_val is None:
            missing_fields.append("Tanggal Bergabung Member Frequent Flayer")
        if first_flight_val is None:
            missing_fields.append("Tanggal Penerbangan Pertama")
        if last_flight_val is None:
            missing_fields.append("Tanggal Penerbangan Terakhir")
        if flight_count_val is None:
            missing_fields.append("Jumlah Penerbangan")
        if last_to_end_val is None:
            missing_fields.append(
                "Jarak Tanggal Penerbangan Terakhir dengan Tanggal Penarikan Data"
            )
        if seg_km_sum_val is None:
            missing_fields.append("Total Jarak Penerbangan")
        if avg_discount_val is None:
            missing_fields.append("Rata-rata Discount")
        if ffp_tier_val is None:
            missing_fields.append("Tingkat Member Frequent Flyer")

        if missing_fields:
            st.warning(
                f"Mohon lengkapi seluruh kolom berikut sebelum prediksi: **{', '.join(missing_fields)}**."
            )
            return

        if load_time_dt < ffp_date_val:
            st.warning(
                f"Tanggal bergabung member (`{ffp_date_val}`) tidak boleh lebih akhir dari tanggal penarikan data (`{load_time_dt}`)."
            )
            return

        if last_flight_val < first_flight_val:
            st.warning(
                "Tanggal penerbangan terakhir tidak boleh lebih awal dari tanggal penerbangan pertama."
            )
            return

        length_val = float((load_time_dt - ffp_date_val).days)
        flight_duration_days = (last_flight_val - first_flight_val).days
        flight_duration_years = max(flight_duration_days, 1) / 365.25
        frequency_val = float(flight_count_val / flight_duration_years)
        recency_val = float(last_to_end_val)
        mileage_val = float(seg_km_sum_val)
        coefficient_val = float(avg_discount_val)
        tier_int = int(ffp_tier_val)
        clean_name = customer_name_val.strip()

        st.session_state.form_values.update(
            {
                "customer_name": clean_name,
                "ffp_date": ffp_date_val,
                "first_flight_date": first_flight_val,
                "last_flight_date": last_flight_val,
                "flight_count": int(flight_count_val),
                "last_to_end": recency_val,
                "seg_km_sum": mileage_val,
                "avg_discount": coefficient_val,
                "ffp_tier": tier_int,
            }
        )

        st.session_state.customer_name = clean_name
        st.session_state.submitted_inputs = {
            "Length": length_val,
            "Recency": recency_val,
            "Frequency": frequency_val,
            "Mileage": mileage_val,
            "Coefficient": coefficient_val,
            "FFP_TIER": tier_int,
        }
        st.session_state.submitted_display = {
            "Nama Pelanggan": clean_name,
            "Tanggal Penarikan Data (Otomatis)": str(load_time_dt),
            "Tanggal Bergabung Member": str(ffp_date_val),
            "Tanggal Penerbangan Pertama": str(first_flight_val),
            "Tanggal Penerbangan Terakhir": str(last_flight_val),
            "Jumlah Penerbangan": f"{int(flight_count_val):,}",
            "Length (Hari)": f"{int(round(length_val)):,}",
            "Recency (Hari)": f"{int(round(recency_val)):,}",
            "Frequency (/Tahun)": f"{int(round(frequency_val)):,}",
            "Mileage (Km)": f"{mileage_val:,.2f}",
            "Coefficient": f"{coefficient_val:.4f}",
            "Tingkat Member": str(tier_int),
        }
        st.session_state.has_prediction = True
        st.session_state.is_editing = False
        st.rerun()


# Bagian Hasil Prediksi Klaster
def render_prediction_section(scaler, kmeans, cluster_summary) -> int:
    st.subheader("🎯 Hasil Prediksi Cluster")

    inputs = st.session_state.submitted_inputs
    df_input = pd.DataFrame([inputs])[
        ["Length", "Recency", "Frequency", "Mileage", "Coefficient", "FFP_TIER"]
    ]

    skewed_cols = ["Frequency", "Mileage", "Recency"]
    for col in skewed_cols:
        df_input[col] = np.log1p(df_input[col])

    scaled_input = scaler.transform(df_input)
    cluster_id = int(kmeans.predict(scaled_input)[0])
    st.session_state.predicted_cluster = cluster_id

    cluster_stats = cluster_summary.loc[cluster_id]
    cluster_info = get_cluster_info(cluster_id)
    customer_name = st.session_state.customer_name

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
            border-left: 7px solid #1D4ED8;
            border-radius: 12px;
            padding: 20px 24px;
            margin: 8px 0 14px 0;
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.08);
        ">
            <div style="font-size: 0.95rem; font-weight: 600; color: #1E40AF; margin-bottom: 4px;">
                Pelanggan: <span style="color: #0F172A;">{customer_name}</span>
            </div>
            <div style="font-size: 2.05rem; font-weight: 800; color: #0F172A; line-height: 1.25; margin-bottom: 10px;">
                Cluster {cluster_id} — <span style="color: #1D4ED8;">{cluster_info["name"]}</span>
            </div>
            <div style="font-size: 1rem; color: #1E293B; line-height: 1.55; margin-bottom: 10px;">
                {cluster_info["description"]}
            </div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #334155;">
                Populasi klaster: {int(cluster_stats["jumlah_pelanggan"]):,} pelanggan
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Lihat Ringkasan Data Input & Hasil Perhitungan LRFMC", expanded=False):
        items = list(st.session_state.submitted_display.items())
        for row_start in range(0, len(items), 4):
            cols = st.columns(4)
            for idx, (label, val) in enumerate(items[row_start : row_start + 4]):
                cols[idx].metric(label, val)

    return cluster_id


# Bagian Rata-rata Profil Klaster
def render_mean_metrics_section(cluster_id: int, mean_profile):
    cluster_info = get_cluster_info(cluster_id)
    st.subheader(
        f"📊 Rata-rata (Mean) Profil Klaster ({cluster_info['name']})"
    )

    mean_row = mean_profile.loc[cluster_id]
    length_rounded = int(round(float(mean_row["Length"])))
    recency_rounded = int(round(float(mean_row["Recency"])))
    frequency_rounded = int(round(float(mean_row["Frequency"])))

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Length", f"{length_rounded:,} hari")
    m2.metric("Recency", f"{recency_rounded:,} hari")
    m3.metric("Frequency", f"{frequency_rounded:,} /tahun")
    m4.metric("Mileage", f"{mean_row['Mileage']:,.2f} km")
    m5.metric("Coefficient", f"{mean_row['Coefficient']:.2f}")
    m6.metric("Average Interval", f"{mean_row['AVG_INTERVAL']:,.2f}")


# Bagian Visualisasi Profil Klaster
def render_visualizations_section(cluster_id: int, mean_profile, profiling_data):
    cluster_info = get_cluster_info(cluster_id)
    st.subheader(
        f"📈 Visualisasi Profil Klaster ({cluster_info['name']})"
    )

    mean_row = mean_profile.loc[cluster_id]
    ct_tier_all = profiling_data["crosstab_ffp_tier"]
    ct_gender_all = profiling_data["crosstab_gender"]

    tier_series = normalize_crosstab_row(ct_tier_all.loc[cluster_id])
    raw_tier_labels = [f"Tier {t}" for t in tier_series.index]
    tier_palette = ["#2563EB", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444"]
    tier_color_map = {
        label: tier_palette[idx % len(tier_palette)]
        for idx, label in enumerate(raw_tier_labels)
    }
    df_tier = (
        pd.DataFrame(
            {
                "FFP_TIER": raw_tier_labels,
                "Persentase (%)": tier_series.values,
            }
        )
        .sort_values("Persentase (%)", ascending=False)
        .reset_index(drop=True)
    )
    tier_top_to_bottom = df_tier["FFP_TIER"].tolist()
    fig_tier = px.bar(
        df_tier,
        x="Persentase (%)",
        y="FFP_TIER",
        orientation="h",
        text="Persentase (%)",
        color="FFP_TIER",
        color_discrete_map=tier_color_map,
        category_orders={"FFP_TIER": tier_top_to_bottom},
        title="Distribusi FFP_TIER",
    )
    fig_tier.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_tier.update_layout(
        xaxis_range=[0, max(float(df_tier["Persentase (%)"].max()) * 1.22, 10)],
        showlegend=False,
        xaxis_title="Persentase (%)",
        yaxis_title="FFP Tier",
        height=380,
        margin=dict(t=55, b=40, l=40, r=45),
    )

    gender_series = normalize_crosstab_row(ct_gender_all.loc[cluster_id])
    gender_labels = []
    for g in gender_series.index:
        g_norm = str(g).strip().upper()
        if g_norm in ("M", "MALE", "LAKI-LAKI", "LAKI LAKI"):
            gender_labels.append("Laki-laki")
        elif g_norm in ("F", "FEMALE", "PEREMPUAN", "WANITA"):
            gender_labels.append("Perempuan")
        else:
            gender_labels.append(str(g).strip())

    df_gender = pd.DataFrame(
        {
            "Jenis Kelamin": gender_labels,
            "Persentase (%)": gender_series.values,
        }
    )
    gender_color_map = {
        "Laki-laki": "#2563EB",
        "Perempuan": "#EC4899",
    }
    fig_gender = px.pie(
        df_gender,
        names="Jenis Kelamin",
        values="Persentase (%)",
        color="Jenis Kelamin",
        color_discrete_map=gender_color_map,
        title="Distribusi Jenis Kelamin Pelanggan",
        hole=0.35,
    )
    fig_gender.update_traces(textinfo="label+percent", textposition="inside")
    fig_gender.update_layout(
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=55, b=55, l=20, r=20),
    )

    sum_yr_labels = ["Total Pendapatan Tahun Pertama", "Total Pendapatan Tahun Kedua"]
    df_sum_yr = (
        pd.DataFrame(
            {
                "Periode Pendapatan": sum_yr_labels,
                "Rata-rata Pendapatan": [
                    float(mean_row["SUM_YR_1"]),
                    float(mean_row["SUM_YR_2"]),
                ],
            }
        )
        .sort_values("Rata-rata Pendapatan", ascending=False)
        .reset_index(drop=True)
    )
    sum_yr_top_to_bottom = df_sum_yr["Periode Pendapatan"].tolist()
    sum_yr_color_map = {
        "Total Pendapatan Tahun Pertama": "#2563EB",
        "Total Pendapatan Tahun Kedua": "#10B981",
    }
    fig_sum_yr = px.bar(
        df_sum_yr,
        x="Rata-rata Pendapatan",
        y="Periode Pendapatan",
        orientation="h",
        text="Rata-rata Pendapatan",
        color="Periode Pendapatan",
        color_discrete_map=sum_yr_color_map,
        category_orders={"Periode Pendapatan": sum_yr_top_to_bottom},
        title="Rata-rata Total Pendapatan Tahun 1 & 2",
    )
    fig_sum_yr.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    fig_sum_yr.update_layout(
        xaxis_range=[0, max(float(df_sum_yr["Rata-rata Pendapatan"].max()) * 1.25, 10)],
        showlegend=False,
        xaxis_title="Rata-rata Nilai Pendapatan",
        yaxis_title="",
        height=380,
        margin=dict(t=55, b=40, l=40, r=55),
    )

    v_col1, v_col2, v_col3 = st.columns(3, gap="medium")
    with v_col1:
        st.plotly_chart(fig_tier, use_container_width=True)
    with v_col2:
        st.plotly_chart(fig_gender, use_container_width=True)
    with v_col3:
        st.plotly_chart(fig_sum_yr, use_container_width=True)

    return fig_tier, fig_gender, fig_sum_yr


# Bagian Download PDF & Edit / Reset Input Data
def _pdf_safe_text(text) -> str:
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_output_to_bytes(pdf: FPDF) -> bytes:
    raw = pdf.output(dest="S")
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="replace")
    return bytes(raw)


def _save_plotly_png(fig: go.Figure, filepath: str, width: int, height: int) -> bool:
    try:
        fig.write_image(filepath, width=width, height=height, scale=2, engine="kaleido")
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
    except Exception:
        try:
            fig.write_image(filepath, width=width, height=height, scale=1)
            return os.path.exists(filepath) and os.path.getsize(filepath) > 0
        except Exception:
            return False


def build_prediction_pdf(
    input_data: dict,
    cluster_id: int,
    cluster_stats: pd.Series,
    mean_row: pd.Series,
    fig_tier: go.Figure,
    fig_gender: go.Figure,
    fig_sum_yr: go.Figure,
) -> bytes:
    cluster_info = get_cluster_info(cluster_id)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        usable_w,
        10,
        _pdf_safe_text("Laporan Hasil Segmentasi Pelanggan Maskapai"),
        ln=True,
    )
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        usable_w,
        6,
        _pdf_safe_text(f"Dicetak pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        ln=True,
    )
    pdf.ln(3)

    pop_count = int(cluster_stats.get("jumlah_pelanggan", 0))

    pdf.set_fill_color(235, 245, 255)
    pdf.set_text_color(15, 55, 110)
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_w,
        7,
        _pdf_safe_text(
            f"Hasil Prediksi: Klaster {cluster_info['name']} "
            f"(Populasi: {pop_count:,} pelanggan"
        ),
        border=0,
        fill=True,
    )
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_w,
        5.5,
        _pdf_safe_text(f"Karakteristik Utama: {cluster_info['description']}"),
    )
    pdf.ln(3)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        usable_w,
        7,
        _pdf_safe_text("1. Identitas & Data Input Pelanggan"),
        ln=True,
    )
    pdf.set_font("Helvetica", "", 8.8)

    col_w = usable_w / 2.0
    items = list(input_data.items())
    for i in range(0, len(items), 2):
        pdf.set_x(pdf.l_margin)
        k1, v1 = items[i]
        line1 = _pdf_safe_text(f"- {k1}: {v1}")
        pdf.cell(col_w, 5.5, line1, border=0)
        if i + 1 < len(items):
            k2, v2 = items[i + 1]
            line2 = _pdf_safe_text(f"- {k2}: {v2}")
            pdf.cell(col_w, 5.5, line2, border=0, ln=True)
        else:
            pdf.ln(5.5)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        usable_w,
        7,
        _pdf_safe_text(f"2. Rata-rata (Mean) Karakteristik Klaster"),
        ln=True,
    )

    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_x(pdf.l_margin)
    pdf.cell(col_w, 7, _pdf_safe_text("Metrik / Fitur"), border=1, fill=True)
    pdf.cell(
        col_w,
        7,
        _pdf_safe_text(f"Nilai Mean (Klaster {cluster_id})"),
        border=1,
        fill=True,
        ln=True,
    )

    pdf.set_font("Helvetica", "", 9.5)
    metrics_to_show = [
        ("Length (Hari)", "Length", True),
        ("Recency (Hari)", "Recency", True),
        ("Frequency (Tahun)", "Frequency", True),
        ("Mileage (km)", "Mileage", False),
        ("Coefficient", "Coefficient", False),
        ("Average Interval", "AVG_INTERVAL", False),
    ]
    for label, col_key, is_int_round in metrics_to_show:
        val = mean_row.get(col_key, np.nan)
        if pd.isna(val):
            val_str = "-"
        elif is_int_round:
            val_str = f"{int(round(float(val))):,}"
        else:
            val_str = f"{float(val):,.2f}"
        pdf.set_x(pdf.l_margin)
        pdf.cell(col_w, 6.5, _pdf_safe_text(f"  {label}"), border=1)
        pdf.cell(col_w, 6.5, _pdf_safe_text(f"  {val_str}"), border=1, ln=True)

    pdf.ln(4)

    if pdf.get_y() > 195:
        pdf.add_page()

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        usable_w,
        7,
        _pdf_safe_text(f"Visualisasi Profil Klaster"),
        ln=True,
    )
    pdf.ln(2)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_tier = os.path.join(tmpdir, "tier.png")
        img_gender = os.path.join(tmpdir, "gender.png")
        img_sum = os.path.join(tmpdir, "sum_yr.png")

        ok_tier = _save_plotly_png(fig_tier, img_tier, width=650, height=380)
        ok_gender = _save_plotly_png(fig_gender, img_gender, width=650, height=380)
        ok_sum = _save_plotly_png(fig_sum_yr, img_sum, width=650, height=380)

        if ok_tier and ok_gender and ok_sum:
            if pdf.get_y() > 215:
                pdf.add_page()
            y_now = pdf.get_y()
            img_w = 61
            img_h = 36
            pdf.image(img_tier, x=10, y=y_now, w=img_w, h=img_h)
            pdf.image(img_gender, x=74.5, y=y_now, w=img_w, h=img_h)
            pdf.image(img_sum, x=139, y=y_now, w=img_w, h=img_h)
            pdf.set_y(y_now + img_h + 4)
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                usable_w,
                5.5,
                _pdf_safe_text(
                    "Catatan: Grafik tidak dapat dirender ke gambar pada lingkungan ini "
                    "(pastikan paket `kaleido` terpasang jika ingin menyertakan chart di PDF)."
                ),
            )

    return _pdf_output_to_bytes(pdf)


def render_export_and_actions_section(
    cluster_id: int,
    cluster_stats: pd.Series,
    mean_row: pd.Series,
    fig_tier: go.Figure,
    fig_gender: go.Figure,
    fig_sum_yr: go.Figure,
):
    st.divider()
    st.subheader("💾 Download PDF & Kelola Input Data")

    raw_name = st.session_state.get("customer_name", "")
    safe_name = (
        "".join(
            c if c.isalnum() or c in (" ", "_", "-") else "_"
            for c in raw_name
        )
        .strip()
        .replace(" ", "_")
    )
    pdf_filename = (
        f"Hasil_Segmentasi_{safe_name}_Cluster_{cluster_id}.pdf"
        if safe_name
        else f"Hasil_Segmentasi_Cluster_{cluster_id}.pdf"
    )

    if st.session_state.get("pdf_bytes_cache") is None and not st.session_state.get("pdf_error_msg"):
        try:
            st.session_state.pdf_bytes_cache = build_prediction_pdf(
                input_data=st.session_state.get("submitted_display", {}),
                cluster_id=cluster_id,
                cluster_stats=cluster_stats,
                mean_row=mean_row,
                fig_tier=fig_tier,
                fig_gender=fig_gender,
                fig_sum_yr=fig_sum_yr,
            )
        except Exception as e:
            st.session_state.pdf_error_msg = str(e)

    col_pdf, col_edit, col_reset, _ = st.columns([1.8, 1.3, 1.4, 2.5])

    with col_pdf:
        pdf_bytes = st.session_state.get("pdf_bytes_cache")
        if pdf_bytes:
            st.download_button(
                label="📥 Download Hasil (PDF)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="download_pdf_btn",
            )
        else:
            st.warning(
                f"PDF belum dapat dibuat: {st.session_state.get('pdf_error_msg', 'Terjadi kesalahan.')}"
            )

    with col_edit:
        if st.button("✏️ Edit Input Data", use_container_width=True, key="edit_input_btn"):
            st.session_state.is_editing = True
            st.rerun()

    with col_reset:
        if st.button("🗑️ Hapus Input Data", use_container_width=True, key="reset_input_btn"):
            reset_all_data()
            st.rerun()


# Entrypoint Utama
def main():
    init_state()

    st.title("✈️ Airline Passenger Segmentation")
    st.markdown(
        "Masukkan identitas dan data penerbangan pelanggan untuk memprediksi **Klaster** "
        "beserta rata-rata karakteristik LRFMC dan visualisasi profil klaster."
    )

    try:
        scaler, kmeans, profiling_data = load_artifacts("pickle")
    except Exception as e:
        st.error(
            f"Gagal memuat file dari folder `pickle/`: {e}. "
            "Pastikan `scaler.pkl`, `kmeans_model.pkl`, dan `profiling_data.pkl` tersedia."
        )
        return

    cluster_summary = profiling_data["cluster_summary"]
    mean_profile = profiling_data["mean_profile"]

    if not st.session_state.has_prediction or st.session_state.is_editing:
        render_input_section()

    if st.session_state.has_prediction:
        st.divider()
        cluster_id = render_prediction_section(scaler, kmeans, cluster_summary)

        st.divider()
        render_mean_metrics_section(cluster_id, mean_profile)

        st.divider()
        fig_tier, fig_gender, fig_sum_yr = render_visualizations_section(
            cluster_id, mean_profile, profiling_data
        )

        render_export_and_actions_section(
            cluster_id=cluster_id,
            cluster_stats=cluster_summary.loc[cluster_id],
            mean_row=mean_profile.loc[cluster_id],
            fig_tier=fig_tier,
            fig_gender=fig_gender,
            fig_sum_yr=fig_sum_yr,
        )


if __name__ == "__main__":
    main()