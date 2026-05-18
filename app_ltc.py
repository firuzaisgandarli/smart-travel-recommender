import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent

MAIN_BAR = ["#93c5fd"]
MAIN_MULTI = px.colors.sequential.Blues_r
MAIN_CAT = px.colors.qualitative.Pastel1
MAIN_LINE = px.colors.qualitative.Pastel1

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Travel Recommender",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
MODEL_FEATURES = [
    "is_food",
    "is_budget",
    "is_luxury",
    "is_nature",
    "is_city",
    "is_history",
    "is_adventure",
    "has_best",
    "has_top",
    "has_cheap",
    "has_luxury_word",
    "luxury_middleeast",
    "title_len",
    "desc_len",
    "has_tags",
    "duration_sec",
    "is_short_video",
    "is_long_video",
    "month",
    "year",
    "dayofweek",
    "country_model",
]

DEFAULT_ASIA = [
    "japan", "thailand", "vietnam", "indonesia", "singapore", "south korea",
    "taiwan", "india", "philippines", "malaysia", "china", "cambodia"
]
DEFAULT_EUROPE = [
    "france", "italy", "spain", "uk", "netherlands", "portugal",
    "greece", "switzerland", "sweden", "norway", "austria", "romania", "croatia"
]
DEFAULT_MIDDLE_EAST = ["uae", "qatar", "saudi arabia", "lebanon", "oman"]
DEFAULT_AMERICAS = [
    "usa", "canada", "mexico", "colombia", "peru", "guatemala", "panama",
    "argentina", "brazil", "chile"
]

REGION_MAP = {
    "Any": None,
    "Asia": DEFAULT_ASIA,
    "Europe": DEFAULT_EUROPE,
    "Middle East": DEFAULT_MIDDLE_EAST,
    "Americas": DEFAULT_AMERICAS,
}

INTEREST_TO_COLUMN = {
    "Food": "is_food",
    "Budget": "is_budget",
    "Luxury": "is_luxury",
    "Nature": "is_nature",
    "City": "is_city",
    "History": "is_history",
    "Adventure": "is_adventure",
}


def score_to_label(score: float) -> str:
    if score >= 0.75:
        return "Excellent"
    if score >= 0.60:
        return "Strong"
    if score >= 0.45:
        return "Good"
    return "Moderate"


def format_seconds(seconds: float) -> str:
    if pd.isna(seconds):
        return "N/A"
    seconds = int(seconds)
    mins = seconds // 60
    hrs = mins // 60
    mins = mins % 60
    if hrs > 0:
        return f"{hrs}h {mins}m"
    return f"{mins} min"


def minmax_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    smin, smax = s.min(), s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - smin) / (smax - smin)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "title": "",
        "description": "",
        "title_clean": "",
        "country": "unknown",
        "country_model": "unknown",
        "city": "unknown",
        "thumbnail_url": "",
        "channel_title": "",
        "engagement": 0.0,
        "duration_sec": 0.0,
        "month": 0,
        "year": 0,
        "dayofweek": 0,
        "title_len": 0,
        "desc_len": 0,
        "has_tags": 0,
        "is_short_video": 0,
        "is_long_video": 0,
        "has_best": 0,
        "has_top": 0,
        "has_cheap": 0,
        "has_luxury_word": 0,
        "luxury_middleeast": 0,
        "is_food": 0,
        "is_budget": 0,
        "is_luxury": 0,
        "is_nature": 0,
        "is_city": 0,
        "is_history": 0,
        "is_adventure": 0,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "final_df_main.csv", low_memory=False)
    df = ensure_columns(df)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)
    df["engagement"] = pd.to_numeric(df["engagement"], errors="coerce").fillna(0)

    return df


def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_resource(show_spinner=False)
def load_model_and_preprocessor():
    model = None
    preprocessor = None
    return model, preprocessor


def apply_filters(
    df: pd.DataFrame,
    budget_choice: str,
    interests: list[str],
    region_choice: str,
    min_duration: int,
    max_duration: int,
    year_range: tuple[int, int],
    min_videos_country: int,
) -> pd.DataFrame:
    temp = df.copy()

    temp = temp[(temp["duration_sec"] >= min_duration) & (temp["duration_sec"] <= max_duration)]
    temp = temp[(temp["year"] >= year_range[0]) & (temp["year"] <= year_range[1])]

    if budget_choice == "Budget":
        temp = temp[temp["is_budget"] == 1]
    elif budget_choice == "Luxury":
        temp = temp[temp["is_luxury"] == 1]

    if interests:
        masks = [temp[INTEREST_TO_COLUMN[i]] == 1 for i in interests if i in INTEREST_TO_COLUMN]
        if masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            temp = temp[combined]

    region_countries = REGION_MAP.get(region_choice)
    if region_countries is not None:
        temp = temp[temp["country_model"].isin(region_countries)]

    country_counts = temp["country_model"].value_counts()
    valid_countries = country_counts[country_counts >= min_videos_country].index
    temp = temp[temp["country_model"].isin(valid_countries)]

    return temp.reset_index(drop=True)


def add_model_scores(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    temp = df.copy()

    if model is None or preprocessor is None:
        temp["model_probability"] = minmax_scale(temp["engagement"]).clip(0, 1)
        return temp

    X = temp[MODEL_FEATURES].copy()
    X_prepared = preprocessor.transform(X)
    temp["model_probability"] = model.predict_proba(X_prepared)[:, 1]
    return temp


def add_ranking_score(df: pd.DataFrame, budget_choice: str, interests: list[str], region_choice: str) -> pd.DataFrame:
    temp = df.copy()

    temp["engagement_norm"] = minmax_scale(temp["engagement"])
    temp["duration_norm"] = minmax_scale(temp["duration_sec"])
    temp["preference_score"] = 0.0

    if budget_choice == "Budget":
        temp["preference_score"] += 0.15 * temp["is_budget"]
    elif budget_choice == "Luxury":
        temp["preference_score"] += 0.15 * temp["is_luxury"]

    for interest in interests:
        col = INTEREST_TO_COLUMN.get(interest)
        if col:
            temp["preference_score"] += 0.12 * temp[col]

    region_countries = REGION_MAP.get(region_choice)
    if region_countries is not None:
        temp["preference_score"] += 0.10 * temp["country_model"].isin(region_countries).astype(int)

    temp["final_score"] = (
        0.62 * temp["model_probability"]
        + 0.18 * temp["engagement_norm"]
        + 0.10 * temp["preference_score"]
        + 0.10 * temp["duration_norm"]
    )

    temp["final_score"] = temp["final_score"].clip(0, 1)
    return temp


def aggregate_destinations(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    temp["city_clean"] = temp["city"].fillna("unknown").astype(str).str.strip().str.lower()
    temp["country_clean"] = temp["country_model"].fillna("unknown").astype(str).str.strip().str.lower()

    temp["destination"] = np.where(
        temp["city_clean"].isin(["", "unknown", "other", "nan"]),
        temp["country_clean"].str.title(),
        temp["city_clean"].str.title() + ", " + temp["country_clean"].str.title(),
    )

    grouped = (
        temp.groupby(["destination", "country_clean"], as_index=False)
        .agg(
            recommendation_score=("final_score", "mean"),
            model_score=("model_probability", "mean"),
            avg_engagement=("engagement", "mean"),
            avg_duration=("duration_sec", "mean"),
            video_count=("destination", "count"),
            food_share=("is_food", "mean"),
            budget_share=("is_budget", "mean"),
            luxury_share=("is_luxury", "mean"),
            nature_share=("is_nature", "mean"),
            history_share=("is_history", "mean"),
        )
        .sort_values(["recommendation_score", "video_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return grouped


def top_video_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    # burada url column-u əlavə olunub
    cols = [
        c for c in [
            "title",
            "channel_title",
            "country_model",
            "city",
            "url",   # datasetdə link sütunu budursa işləyəcək
            "final_score",
            "model_probability",
            "engagement",
            "duration_sec"
        ] if c in df.columns
    ]

    temp = df.sort_values("final_score", ascending=False)[cols].head(n).copy()

    if "duration_sec" in temp.columns:
        temp["duration_sec"] = temp["duration_sec"].apply(format_seconds)
        temp = temp.rename(columns={"duration_sec": "duration"})

    return temp


def build_insight_text(row: pd.Series) -> str:
    reasons = []

    if row.get("food_share", 0) >= 0.50:
        reasons.append("This destination strongly matches food-related travel interest.")
    if row.get("budget_share", 0) >= 0.50:
        reasons.append("It includes many budget-friendly travel videos.")
    if row.get("luxury_share", 0) >= 0.50:
        reasons.append("It has strong luxury travel content.")
    if row.get("nature_share", 0) >= 0.50:
        reasons.append("It is supported by nature-focused videos.")
    if row.get("history_share", 0) >= 0.50:
        reasons.append("It shows strong history and culture interest.")
    if row.get("video_count", 0) >= 5:
        reasons.append("It is backed by enough supporting videos.")
    if row.get("recommendation_score", 0) >= 0.70:
        reasons.append("It has a high recommendation score.")

    if not reasons:
        return "This destination shows balanced travel signals across the filtered dataset."

    return " ".join(reasons[:3])


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(16,185,129,0.08));
        border: 1px solid rgba(15,23,42,0.06);
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
        margin-bottom: 1.2rem;
    }

    .metric-card {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(15,23,42,0.06);
        box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    }

    .dest-card {
        border-radius: 22px;
        padding: 1.1rem 1.15rem;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.06);
        box-shadow: 0 10px 28px rgba(15,23,42,0.06);
        min-height: 220px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .dest-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 34px rgba(15,23,42,0.10);
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .soft-box {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.80);
        border: 1px solid rgba(15,23,42,0.06);
        box-shadow: 0 6px 18px rgba(15,23,42,0.04);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(15,23,42,0.06);
        padding: 14px 16px;
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(15,23,42,0.04);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    div[data-testid="stSidebar"] {
        background: #f3f6fb;
        border-right: 1px solid rgba(15,23,42,0.06);
    }

    .stDownloadButton button {
        border-radius: 14px;
        border: 1px solid rgba(15,23,42,0.08);
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load assets
# ------------------------------------------------------------
try:
    df = load_data()
    model, preprocessor = load_model_and_preprocessor()
except Exception as e:
    st.error(f"Setup error: {e}")
    st.stop()

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

budget_choice = st.sidebar.selectbox("Budget style", ["Any", "Budget", "Luxury"])
interests = st.sidebar.multiselect(
    "Travel interests",
    ["Food", "Nature", "Luxury", "City", "History", "Adventure", "Budget"],
    default=["Food", "Nature"],
)
region_choice = st.sidebar.selectbox("Region", ["Any", "Asia", "Europe", "Middle East", "Americas"])

max_duration_value = int(max(60, df["duration_sec"].fillna(0).max()))
min_duration, max_duration = st.sidebar.slider(
    "Video duration range (seconds)",
    min_value=0,
    max_value=max_duration_value,
    value=(30, min(2400, max_duration_value)),
    step=30,
)

min_year = int(df["year"].replace(0, np.nan).dropna().min()) if (df["year"] > 0).any() else 2020
max_year = int(df["year"].max()) if (df["year"] > 0).any() else 2026
year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(max(min_year, max_year - 4), max_year))

min_videos_country = st.sidebar.slider("Minimum videos per country", 1, 10, 2)
top_n = st.sidebar.slider("Top destinations", 3, 12, 5)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class='hero'>
        <h1 style='margin-bottom:0.35rem;'>🌍 Smart Travel Recommender</h1>
        <p style='margin:0; font-size:1.03rem;'>A YouTube-based travel recommendation dashboard using metadata, NLP features, engagement signals, and a trained ML model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------
run_btn = st.sidebar.button(
    "Generate Recommendations",
    type="primary",
    use_container_width=True,
    key="generate_main_btn"
)

if not run_btn:
    st.info("Choose your filters on the left, then click *Generate Recommendations*.")

if "run_recommender" not in st.session_state:
    st.session_state.run_recommender = False

if run_btn:
    st.session_state.run_recommender = True

if st.session_state.run_recommender:
    filtered = apply_filters(
        df=df,
        budget_choice=budget_choice,
        interests=interests,
        region_choice=region_choice,
        min_duration=min_duration,
        max_duration=max_duration,
        year_range=year_range,
        min_videos_country=min_videos_country,
    )

    filtered = filtered[
        ~filtered["country_model"].astype(str).str.lower().isin(
            ["unknown", "other", "", "nan"]
        )
    ]

    if filtered.empty:
        st.warning("No rows matched these filters. Try widening the duration, year range, or region.")
        st.stop()

    scored = add_model_scores(filtered, model, preprocessor)
    mode_text = "Fallback scoring active" if model is None or preprocessor is None else "ML model active"
    st.info(f"Scoring mode: {mode_text}")

    scored = add_ranking_score(scored, budget_choice, interests, region_choice)
    dest_df = aggregate_destinations(scored).head(top_n)
    top_videos = top_video_table(scored, n=min(10, len(scored)))

    # KPIs
    k1, k2, k3, k4 = st.columns(4)

    top_avg = dest_df["recommendation_score"].head(5).mean()
    model_avg = scored["model_probability"].mean()

    k1.metric("Filtered Videos", f"{len(scored):,}")
    k2.metric("Unique Destinations", f"{dest_df['destination'].nunique():,}")
    k3.metric("Top Destination Score", f"{top_avg*100:.0f}%")
    k4.metric("Average Match Score", f"{model_avg*100:.0f}%")

    st.markdown("### 📍 Destination Detail")

    detail_df = scored.copy()
    detail_df["city_clean"] = detail_df["city"].fillna("unknown").astype(str).str.strip().str.lower()
    detail_df["country_clean"] = detail_df["country_model"].fillna("unknown").astype(str).str.strip().str.lower()

    detail_df["destination"] = np.where(
        detail_df["city_clean"].isin(["", "unknown", "other", "nan"]),
        detail_df["country_clean"].str.title(),
        detail_df["city_clean"].str.title() + ", " + detail_df["country_clean"].str.title(),
    )

    selected_destination = st.selectbox(
        "Choose a destination",
        dest_df["destination"].tolist(),
        key="destination_detail_select"
    )

    selected_rows = detail_df[detail_df["destination"] == selected_destination]

    d1, d2, d3 = st.columns(3)
    d1.metric("Videos", len(selected_rows))
    d2.metric("Avg Score", f"{selected_rows['final_score'].mean():.2f}")
    d3.metric("Avg Engagement", f"{selected_rows['engagement'].mean():.3f}")

    st.markdown("### 🏆 Top Destination Recommendations")
    c1, c2, c3 = st.columns(3)
    top_cards = [c1, c2, c3]

    for idx, (_, row) in enumerate(dest_df.head(3).iterrows()):
        with top_cards[idx]:
            st.markdown(
                f"""
                <div class='dest-card'>
                    <h3 style='margin-top:0;'>#{idx+1} {row['destination']}</h3>
                    <p><b>Score:</b> {row['recommendation_score']:.2f} {"🟢 Excellent" if row['recommendation_score'] >= 0.80 else "🟡 Good" if row['recommendation_score'] >= 0.55 else "🟠 Fair" if row['recommendation_score'] >= 0.35 else "🔴 Low"}</p>
                    <p><b>Support:</b> {int(row['video_count'])} videos</p>
                    <p><b>Avg duration:</b> {format_seconds(row['avg_duration'])}</p>
                    <p><b>Why:</b> {build_insight_text(row)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 🤖 Why This Recommendation?")

    best_row = dest_df.iloc[0]
    reasons = []

    if best_row["food_share"] > 0.4:
        reasons.append("🍽 Strong food-related travel content")
    if best_row["nature_share"] > 0.4:
        reasons.append("🌿 Nature-focused destination appeal")
    if best_row["budget_share"] > 0.4:
        reasons.append("💰 Good budget-friendly options")
    if best_row["luxury_share"] > 0.4:
        reasons.append("✨ Premium luxury travel interest")
    if best_row["video_count"] >= 5:
        reasons.append("🎥 Backed by multiple relevant videos")
    if best_row["recommendation_score"] > 0.75:
        reasons.append("📈 High recommendation confidence")

    st.success(f"*{best_row['destination']}* is currently your best match.")
    for r in reasons[:4]:
        st.write(r)

    st.markdown("### 📋 Destination Ranking Table")
    show_df = dest_df.copy()
    show_df["recommendation_score"] = show_df["recommendation_score"].round(3)
    show_df["model_score"] = show_df["model_score"].round(3)
    show_df["avg_engagement"] = show_df["avg_engagement"].round(4)
    show_df["avg_duration"] = show_df["avg_duration"].apply(format_seconds)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="📥 Download Destination Ranking",
        data=convert_df_to_csv(show_df),
        file_name="destination_ranking.csv",
        mime="text/csv"
    )

    # Charts row 1
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("### 📈 Recommendation Score by Destination")
        fig = px.bar(
            dest_df,
            x="recommendation_score",
            y="destination",
            orientation="h",
            text="recommendation_score",
            color_discrete_sequence=MAIN_BAR
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(height=430, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("### 🌐 Country Distribution")
        country_counts = (
            scored["country_model"].value_counts().head(12).rename_axis("country_model").reset_index(name="count")
        )
        fig2 = px.pie(
            country_counts,
            names="country_model",
            values="count",
            hole=0.45,
            color_discrete_sequence=MAIN_MULTI
        )
        fig2.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Charts row 2
    ch3, ch4 = st.columns(2)
    with ch3:
        st.markdown("### 🎯 Content Type Mix")
        content_mix = pd.DataFrame({
            "Feature": ["Food", "Budget", "Luxury", "Nature", "City", "History", "Adventure"],
            "Share": [
                scored["is_food"].mean(),
                scored["is_budget"].mean(),
                scored["is_luxury"].mean(),
                scored["is_nature"].mean(),
                scored["is_city"].mean(),
                scored["is_history"].mean(),
                scored["is_adventure"].mean(),
            ]
        })
        fig3 = px.bar(
            content_mix,
            x="Feature",
            y="Share",
            text="Share",
            color="Feature",
            color_discrete_sequence=MAIN_BAR
        )
        fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig3.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        st.markdown("### ⏱️ Score vs Duration")
        scatter_df = scored[["duration_sec", "final_score", "country_model"]].copy().head(500)
        fig4 = px.scatter(
            scatter_df,
            x="duration_sec",
            y="final_score",
            color="country_model",
            opacity=0.7,
            color_discrete_sequence=MAIN_CAT
        )
        fig4.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🌍 Global Recommendation Map")

    map_df = scored.groupby("country_model", as_index=False).agg(
        avg_score=("final_score", "mean"),
        videos=("country_model", "count")
    )

    map_df["country_model"] = map_df["country_model"].str.title()

    fig_map = px.choropleth(
        map_df,
        locations="country_model",
        locationmode="country names",
        color="avg_score",
        hover_name="country_model",
        hover_data={"videos": True, "avg_score": ':.2f'},
        color_continuous_scale="Viridis"
    )
    fig_map.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### 📈 Destination Trend Over Years")

    scored["city_clean"] = scored["city"].fillna("unknown").astype(str).str.strip().str.lower()
    scored["country_clean"] = scored["country_model"].fillna("unknown").astype(str).str.strip().str.lower()

    scored["destination"] = np.where(
        scored["city_clean"].isin(["", "unknown", "other", "nan"]),
        scored["country_clean"].str.title(),
        scored["city_clean"].str.title() + ", " + scored["country_clean"].str.title(),
    )

    trend_df = scored.groupby(["year", "destination"], as_index=False).agg(
        avg_score=("final_score", "mean")
    )

    top_destinations = dest_df["destination"].head(5).tolist()
    trend_df = trend_df[trend_df["destination"].isin(top_destinations)]

    fig_trend = px.line(
        trend_df,
        x="year",
        y="avg_score",
        color="destination",
        markers=True,
        title="",
        color_discrete_sequence=MAIN_LINE
    )
    fig_trend.update_layout(
        height=450,
        xaxis_title="Year",
        yaxis_title="Average Score",
        legend_title="Destination"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Analysis section
    st.markdown("### 🔍 Recommendation Analysis")
    a1, a2 = st.columns([1.1, 0.9])

    with a1:
        st.markdown("#### Top Matching Videos")

        # burada link column-u göstərilir
        if "url" in top_videos.columns:
            st.data_editor(
                top_videos,
                use_container_width=True,
                hide_index=True,
                disabled=True,
                column_config={
                    "url": st.column_config.LinkColumn(
                        "Watch Video",
                        display_text="Open ▶"
                    )
                }
            )
        else:
            st.dataframe(top_videos, use_container_width=True, hide_index=True)

    with a2:
        st.markdown("#### Quick Insights")
        st.markdown(
            f"""
            - *Best destination right now:* *{dest_df.iloc[0]['destination']}*
            - *Average model probability:* *{scored['model_probability'].mean():.2f}*
            - *Most common country in filtered set:* *{scored['country_model'].mode().iloc[0].title()}*
            - *Most common travel signal:* *{max([(scored['is_food'].mean(),'Food'),(scored['is_budget'].mean(),'Budget'),(scored['is_luxury'].mean(),'Luxury'),(scored['is_nature'].mean(),'Nature'),(scored['is_city'].mean(),'City')])[1]}*
            - *Threshold note:* ranking uses model probability plus engagement and preference signals.
            """
        )

    with st.expander("See filtered raw dataset"):
        raw_cols = [
            c for c in [
                "title", "thumbnail_url", "country_model", "city",
                "channel_title", "engagement", "final_score"
            ] if c in scored.columns
        ]
        raw_df = scored[raw_cols].copy()
        st.dataframe(
            raw_df.sort_values("final_score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Tip: place app_main.py, final_df_main.csv, and model files in the same folder, then run streamlit run app_main.py"
)