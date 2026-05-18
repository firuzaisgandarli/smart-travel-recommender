from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Travel Insights",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# UNIVERSAL COLOR SYSTEM
# =========================================================
PRIMARY_SCALE = [
    [0.00, "#dbeafe"],
    [0.15, "#bfdbfe"],
    [0.30, "#93c5fd"],
    [0.45, "#60a5fa"],
    [0.60, "#3b82f6"],
    [0.75, "#2563eb"],
    [0.90, "#1d4ed8"],
    [1.00, "#1e40af"]
]

PRIMARY_DISCRETE = [
    "#1e40af",
    "#1d4ed8",
    "#2563eb",
    "#3b82f6",
    "#60a5fa",
    "#93c5fd",
    "#bfdbfe",
    "#dbeafe"
]

PRIMARY_LINE = ["#3b82f6"]

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1728 100%) !important;
}

.main {
    background: linear-gradient(180deg, #07111f 0%, #0b1728 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

.hero-box {
    padding: 1.6rem 1.8rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(16,185,129,0.14));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 14px 36px rgba(0,0,0,0.18);
    margin-bottom: 1.1rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
    color: white;
    letter-spacing: -0.03em;
}

.hero-sub {
    font-size: 1.02rem;
    color: rgba(255,255,255,0.82);
    margin-top: 0.5rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.6rem;
}

.glass-card {
    border-radius: 20px;
    padding: 1rem 1.1rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 25px rgba(0,0,0,0.18);
}

.insight-card {
    border-radius: 20px;
    padding: 1rem 1.1rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    min-height: 135px;
}

.insight-title {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.76);
    margin-bottom: 0.4rem;
}

.insight-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.35rem;
}

.insight-desc {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.78);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px 16px;
    border-radius: 18px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
}

div[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.72) !important;
}

div[data-testid="stMetricValue"] {
    color: white !important;
}

h1, h2, h3, label, p, div {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "final_df_main.csv", low_memory=False)

df = load_data()

# =========================================================
# CLEAN
# =========================================================
for col in ["engagement", "duration_sec", "year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in [
    "is_food", "is_budget", "is_luxury", "is_nature",
    "is_city", "is_history", "is_adventure",
    "is_vlog_style", "has_person"
]:
    if col not in df.columns:
        df[col] = 0

if "country_model" not in df.columns:
    df["country_model"] = "unknown"

if "city" not in df.columns:
    df["city"] = "unknown"

clean_df = df.copy()
clean_df = clean_df[
    ~clean_df["country_model"].astype(str).str.lower().isin(["unknown", "other", "", "nan"])
].copy()

clean_df["country_model"] = clean_df["country_model"].astype(str).str.title()
clean_df["city"] = clean_df["city"].fillna("unknown").astype(str).str.strip()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class='hero-box'>
    <div class='hero-title'>📊 Travel Insights Dashboard</div>
    <div class='hero-sub'>
        Explore patterns in the travel dataset: interests, engagement, countries, cities, content style, and video behavior.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPIs
# =========================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Videos", f"{len(clean_df):,}")
k2.metric("Unique Countries", f"{clean_df['country_model'].nunique():,}")
k3.metric("Average Engagement", f"{clean_df['engagement'].mean():.3f}" if "engagement" in clean_df.columns else "N/A")
k4.metric("Average Duration", f"{clean_df['duration_sec'].mean()/60:.1f} min" if "duration_sec" in clean_df.columns else "N/A")

st.markdown("")

# =========================================================
# PREP DATA
# =========================================================
interest_df = pd.DataFrame({
    "Interest": ["Food", "Budget", "Luxury", "Nature", "City", "History", "Adventure"],
    "Share": [
        clean_df["is_food"].mean(),
        clean_df["is_budget"].mean(),
        clean_df["is_luxury"].mean(),
        clean_df["is_nature"].mean(),
        clean_df["is_city"].mean(),
        clean_df["is_history"].mean(),
        clean_df["is_adventure"].mean(),
    ]
}).sort_values("Share", ascending=False)

country_perf = (
    clean_df.groupby("country_model", as_index=False)
    .agg(
        avg_engagement=("engagement", "mean"),
        video_count=("country_model", "count"),
        avg_duration=("duration_sec", "mean")
    )
    .sort_values(["avg_engagement", "video_count"], ascending=[False, False])
    .head(12)
    .copy()
)

city_df = clean_df.copy()
city_df["city_clean"] = city_df["city"].astype(str).str.strip().str.lower()
city_df = city_df[~city_df["city_clean"].isin(["unknown", "other", "", "nan"])]

top_cities = (
    city_df.groupby(["city_clean", "country_model"], as_index=False)
    .agg(
        avg_engagement=("engagement", "mean"),
        video_count=("city_clean", "count")
    )
    .sort_values(["avg_engagement", "video_count"], ascending=[False, False])
    .head(12)
    .copy()
)

top_cities["destination"] = top_cities["city_clean"].str.title() + ", " + top_cities["country_model"]

top_interest = interest_df.iloc[0]["Interest"] if len(interest_df) else "N/A"
top_country = country_perf.iloc[0]["country_model"] if len(country_perf) else "N/A"
top_city = top_cities.iloc[0]["destination"] if len(top_cities) else "N/A"

vlog_yes = clean_df[clean_df["is_vlog_style"] == 1]["engagement"].mean()
vlog_no = clean_df[clean_df["is_vlog_style"] == 0]["engagement"].mean()
vlog_better = "Vlog-style" if vlog_yes > vlog_no else "Non-vlog"

# =========================================================
# INSIGHT CARDS
# =========================================================
i1, i2, i3, i4 = st.columns(4)

with i1:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Top Content Theme</div>
        <div class='insight-value'>{top_interest}</div>
        <div class='insight-desc'>This is the strongest recurring travel signal in the dataset.</div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Best Country</div>
        <div class='insight-value'>{top_country}</div>
        <div class='insight-desc'>Highest average engagement among top country groups.</div>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Top City</div>
        <div class='insight-value'>{top_city}</div>
        <div class='insight-desc'>Strong city-level performance based on average engagement.</div>
    </div>
    """, unsafe_allow_html=True)

with i4:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Best Video Style</div>
        <div class='insight-value'>{vlog_better}</div>
        <div class='insight-desc'>This style currently performs better in engagement terms.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# INTEREST MIX
# =========================================================
st.markdown("<div class='section-title'>🎯 What travel themes appear most often?</div>", unsafe_allow_html=True)

fig_interest = px.bar(
    interest_df,
    x="Interest",
    y="Share",
    text="Share",
    title="Content Theme Mix",
    color="Share",
    color_continuous_scale=PRIMARY_SCALE
)

fig_interest.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_interest.update_layout(
    height=430,
    title_font_size=20,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    coloraxis_showscale=False
)

st.plotly_chart(fig_interest, use_container_width=True)

# =========================================================
# TOP COUNTRIES / TOP CITIES
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='section-title'>🌍 Top countries by average engagement</div>", unsafe_allow_html=True)

    fig_country = px.bar(
        country_perf,
        x="avg_engagement",
        y="country_model",
        orientation="h",
        text="avg_engagement",
        color="avg_engagement",
        color_continuous_scale=PRIMARY_SCALE
    )

    fig_country.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_country.update_layout(
        title="Country Performance",
        height=500,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig_country, use_container_width=True)

with c2:
    st.markdown("<div class='section-title'>🏙️ Top cities by average engagement</div>", unsafe_allow_html=True)

    fig_city = px.bar(
        top_cities,
        x="avg_engagement",
        y="destination",
        orientation="h",
        text="avg_engagement",
        color="avg_engagement",
        color_continuous_scale=PRIMARY_SCALE
    )

    fig_city.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_city.update_layout(
        title="City Performance",
        height=500,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig_city, use_container_width=True)

# =========================================================
# TREND / SCATTER
# =========================================================
c3, c4 = st.columns(2)

with c3:
    st.markdown("<div class='section-title'>📈 Engagement trend over time</div>", unsafe_allow_html=True)

    if "year" in clean_df.columns and clean_df["year"].notna().any():
        year_trend = (
            clean_df.groupby("year", as_index=False)
            .agg(
                avg_engagement=("engagement", "mean"),
                videos=("year", "count")
            )
            .sort_values("year")
        )

        fig_year = px.line(
            year_trend,
            x="year",
            y="avg_engagement",
            markers=True,
            title="Average Engagement by Year",
            color_discrete_sequence=PRIMARY_LINE
        )

        fig_year.update_layout(
            height=420,
            title_font_size=18,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig_year, use_container_width=True)

with c4:
    st.markdown("<div class='section-title'>⏱️ Duration vs engagement</div>", unsafe_allow_html=True)

    scatter_df = clean_df[["duration_sec", "engagement", "country_model"]].dropna().head(700)

    fig_scatter = px.scatter(
        scatter_df,
        x="duration_sec",
        y="engagement",
        color="country_model",
        opacity=0.72,
        title="Duration vs Engagement",
        color_discrete_sequence=PRIMARY_DISCRETE
    )

    fig_scatter.update_layout(
        height=420,
        showlegend=False,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# STYLE COMPARISONS
# =========================================================
c5, c6 = st.columns(2)

with c5:
    st.markdown("<div class='section-title'>🎥 Vlog-style vs non-vlog</div>", unsafe_allow_html=True)

    vlog_df = (
        clean_df.groupby("is_vlog_style", as_index=False)
        .agg(
            avg_engagement=("engagement", "mean"),
            avg_duration=("duration_sec", "mean"),
            videos=("is_vlog_style", "count")
        )
    )

    vlog_df["Type"] = vlog_df["is_vlog_style"].map({
        0: "Non-vlog / Scene-based",
        1: "Vlog-style"
    })

    fig_vlog = px.bar(
        vlog_df,
        x="Type",
        y="avg_engagement",
        text="avg_engagement",
        title="Does vlog style perform better?",
        color="avg_engagement",
        color_continuous_scale=PRIMARY_SCALE
    )

    fig_vlog.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_vlog.update_layout(
        height=420,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_vlog, use_container_width=True)

with c6:
    st.markdown("<div class='section-title'>👤 Human presence vs no human presence</div>", unsafe_allow_html=True)

    person_df = (
        clean_df.groupby("has_person", as_index=False)
        .agg(
            avg_engagement=("engagement", "mean"),
            videos=("has_person", "count")
        )
    )

    person_df["Type"] = person_df["has_person"].map({
        0: "No person visible",
        1: "Person visible"
    })

    fig_person = px.bar(
        person_df,
        x="Type",
        y="avg_engagement",
        text="avg_engagement",
        title="Does visible human presence help?",
        color="avg_engagement",
        color_continuous_scale=PRIMARY_SCALE
    )

    fig_person.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_person.update_layout(
        height=420,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_person, use_container_width=True)

# =========================================================
# FINAL SUMMARY
# =========================================================
st.markdown("<div class='section-title'>🧠 Key Insights</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class='glass-card'>
    <p>• <b>Most dominant content theme:</b> {top_interest}</p>
    <p>• <b>Best-performing country:</b> {top_country}</p>
    <p>• <b>Best-performing city:</b> {top_city}</p>
    <p>• <b>Best video style:</b> {vlog_better}</p>
    <p>• <b>This page helps explain overall user preference signals before using the recommendation page.</b></p>
</div>
""", unsafe_allow_html=True)

with st.expander("See summarized tables"):
    st.dataframe(country_perf, use_container_width=True, hide_index=True)
    st.dataframe(top_cities, use_container_width=True, hide_index=True)
