from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Comparisons",
    page_icon="⚖️",
    layout="wide"
)

# =========================================================
# COLORS
# soft green / mint comparison palette
# =========================================================
BAR_PALETTE = ["#86efac", "#a7f3d0", "#6ee7b7", "#bbf7d0", "#99f6e4", "#d9f99d", "#c7d2fe"]
COMPARE_PALETTE = px.colors.qualitative.Set3
SCATTER_PALETTE = px.colors.qualitative.Pastel
LINE_COLOR = ["#22c55e"]

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
    background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(16,185,129,0.12), rgba(59,130,246,0.10));
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

df = df[
    ~df["country_model"].astype(str).str.lower().isin(["unknown", "other", "", "nan"])
].copy()

df["country_model"] = df["country_model"].astype(str).str.title()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class='hero-box'>
    <div class='hero-title'>⚖️ Comparative Travel Analysis</div>
    <div class='hero-sub'>
        Compare travel interests, content styles, countries, and destination-level behavior patterns across the dataset.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# COUNTRY FILTER
# =========================================================
country_options = sorted(df["country_model"].dropna().unique().tolist())
selected_country = st.selectbox("Choose a country for detailed comparison", country_options)

country_df = df[df["country_model"] == selected_country].copy()

# =========================================================
# KPIs
# =========================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Selected Country", selected_country)
k2.metric("Videos in Country", f"{len(country_df):,}")
k3.metric("Avg Engagement", f"{country_df['engagement'].mean():.3f}" if len(country_df) else "N/A")
k4.metric("Avg Duration", f"{country_df['duration_sec'].mean()/60:.1f} min" if len(country_df) else "N/A")

st.markdown("")

# =========================================================
# COUNTRY PROFILE
# =========================================================
interest_compare = pd.DataFrame({
    "Interest": ["Food", "Budget", "Luxury", "Nature", "City", "History", "Adventure"],
    "Share": [
        country_df["is_food"].mean(),
        country_df["is_budget"].mean(),
        country_df["is_luxury"].mean(),
        country_df["is_nature"].mean(),
        country_df["is_city"].mean(),
        country_df["is_history"].mean(),
        country_df["is_adventure"].mean(),
    ]
}).sort_values("Share", ascending=False)

best_theme = interest_compare.iloc[0]["Interest"] if len(interest_compare) else "N/A"

# =========================================================
# INSIGHT CARDS
# =========================================================
budget_mean = df[df["is_budget"] == 1]["engagement"].mean()
luxury_mean = df[df["is_luxury"] == 1]["engagement"].mean()
food_mean = df[df["is_food"] == 1]["engagement"].mean()
nature_mean = df[df["is_nature"] == 1]["engagement"].mean()
vlog_mean = df[df["is_vlog_style"] == 1]["engagement"].mean()
non_vlog_mean = df[df["is_vlog_style"] == 0]["engagement"].mean()

budget_vs_luxury = "Budget" if budget_mean > luxury_mean else "Luxury"
food_vs_nature = "Food" if food_mean > nature_mean else "Nature"
vlog_vs_non = "Vlog-style" if vlog_mean > non_vlog_mean else "Non-vlog"

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Strongest Theme</div>
        <div class='insight-value'>{best_theme}</div>
        <div class='insight-desc'>This is the top signal in {selected_country}.</div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Budget vs Luxury</div>
        <div class='insight-value'>{budget_vs_luxury}</div>
        <div class='insight-desc'>This travel style performs better overall.</div>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Food vs Nature</div>
        <div class='insight-value'>{food_vs_nature}</div>
        <div class='insight-desc'>This content category has stronger engagement.</div>
    </div>
    """, unsafe_allow_html=True)

with i4:
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-title'>Winning Video Style</div>
        <div class='insight-value'>{vlog_vs_non}</div>
        <div class='insight-desc'>This style currently performs better in the dataset.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# COUNTRY INTEREST PROFILE
# =========================================================
st.markdown("<div class='section-title'>🌍 Interest profile for selected country</div>", unsafe_allow_html=True)

fig_interest = px.bar(
    interest_compare,
    x="Interest",
    y="Share",
    text="Share",
    title=f"Interest Profile: {selected_country}",
    color="Interest",
    color_discrete_sequence=BAR_PALETTE
)
fig_interest.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_interest.update_layout(
    height=430,
    title_font_size=20,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white"
)
st.plotly_chart(fig_interest, use_container_width=True)

# =========================================================
# HEAD TO HEAD COMPARISONS
# =========================================================
c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='section-title'>💸 Budget vs Luxury</div>", unsafe_allow_html=True)

    budget_luxury_df = pd.DataFrame({
        "Type": ["Budget", "Luxury"],
        "Average Engagement": [budget_mean, luxury_mean]
    })

    fig_budget = px.bar(
        budget_luxury_df,
        x="Type",
        y="Average Engagement",
        text="Average Engagement",
        title="Budget vs Luxury Performance",
        color="Type",
        color_discrete_sequence=["#86efac", "#c4b5fd"]
    )
    fig_budget.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_budget.update_layout(
        height=420,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_budget, use_container_width=True)

with c2:
    st.markdown("<div class='section-title'>🍜 Food vs Nature</div>", unsafe_allow_html=True)

    food_nature_df = pd.DataFrame({
        "Type": ["Food", "Nature"],
        "Average Engagement": [food_mean, nature_mean]
    })

    fig_fn = px.bar(
        food_nature_df,
        x="Type",
        y="Average Engagement",
        text="Average Engagement",
        title="Food vs Nature Performance",
        color="Type",
        color_discrete_sequence=["#fde68a", "#86efac"]
    )
    fig_fn.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fn.update_layout(
        height=420,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_fn, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.markdown("<div class='section-title'>🎥 Vlog vs Non-vlog</div>", unsafe_allow_html=True)

    vlog_df = (
        df.groupby("is_vlog_style", as_index=False)
        .agg(
            avg_engagement=("engagement", "mean"),
            videos=("is_vlog_style", "count")
        )
    )
    vlog_df["Type"] = vlog_df["is_vlog_style"].map({0: "Non-vlog", 1: "Vlog-style"})

    fig_vlog = px.bar(
        vlog_df,
        x="Type",
        y="avg_engagement",
        text="avg_engagement",
        title="Vlog-style Comparison",
        color="Type",
        color_discrete_sequence=["#93c5fd", "#86efac"]
    )
    fig_vlog.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_vlog.update_layout(
        height=420,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_vlog, use_container_width=True)

with c4:
    st.markdown("<div class='section-title'>👤 Human visible vs not visible</div>", unsafe_allow_html=True)

    person_df = (
        df.groupby("has_person", as_index=False)
        .agg(
            avg_engagement=("engagement", "mean"),
            videos=("has_person", "count")
        )
    )
    person_df["Type"] = person_df["has_person"].map({0: "No person visible", 1: "Person visible"})

    fig_person = px.bar(
        person_df,
        x="Type",
        y="avg_engagement",
        text="avg_engagement",
        title="Human Presence Comparison",
        color="Type",
        color_discrete_sequence=["#c7d2fe", "#a7f3d0"]
    )
    fig_person.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_person.update_layout(
        height=420,
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_person, use_container_width=True)

# =========================================================
# COUNTRY LEAGUE TABLE
# =========================================================
st.markdown("<div class='section-title'>🏆 Country comparison table</div>", unsafe_allow_html=True)

country_compare = (
    df.groupby("country_model", as_index=False)
    .agg(
        avg_engagement=("engagement", "mean"),
        avg_duration=("duration_sec", "mean"),
        videos=("country_model", "count"),
        food_share=("is_food", "mean"),
        budget_share=("is_budget", "mean"),
        luxury_share=("is_luxury", "mean"),
        nature_share=("is_nature", "mean"),
        history_share=("is_history", "mean"),
    )
    .sort_values(["avg_engagement", "videos"], ascending=[False, False])
)

country_compare["avg_duration"] = (country_compare["avg_duration"] / 60).round(1)

st.dataframe(country_compare, use_container_width=True, hide_index=True)

# =========================================================
# EXTRA SUMMARY
# =========================================================
top_country = country_compare.iloc[0]["country_model"] if len(country_compare) else "N/A"

st.markdown("<div class='section-title'>🧠 Comparative Insight Summary</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class='glass-card'>
    <p>• <b>Selected country:</b> {selected_country}</p>
    <p>• <b>Strongest travel signal in this country:</b> {best_theme}</p>
    <p>• <b>Best overall country by engagement:</b> {top_country}</p>
    <p>• <b>Winning global comparison:</b> {budget_vs_luxury} in budget-vs-luxury, {food_vs_nature} in food-vs-nature, and {vlog_vs_non} in style comparison.</b></p>
    <p>• <b>This page helps compare what type of travel content tends to perform better and how preferences differ across markets.</b></p>
</div>
""", unsafe_allow_html=True)

with st.expander("See detailed comparison tables"):
    st.dataframe(country_compare, use_container_width=True, hide_index=True)
    st.dataframe(interest_compare, use_container_width=True, hide_index=True)
