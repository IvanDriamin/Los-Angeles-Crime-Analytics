import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import pydeck as pdk

import joblib

from pathlib import Path

# PAGE CONFIG

st.set_page_config(page_title="LA Crime Analytics", layout="wide")

# CSS


def load_css():
    with open("assets/style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# DATA


@st.cache_data
def load_data():
    df = pd.read_csv("data/Crime_Data_from_2020_to_2024.csv")
    return df


df = load_data()

# MODELS


@st.cache_resource
def load_models():
    crime_model = joblib.load("models/crime_type_model.pkl")
    risk_model = joblib.load("models/area_risk_model.pkl")
    hotspot_model = joblib.load("models/hotspot_model.pkl")
    return (crime_model, risk_model, hotspot_model)


crime_model, risk_model, hotspot_model = load_models()

# FEATURE ENGINEERING

df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
df["YEAR"] = df["DATE OCC"].dt.year
df["MONTH"] = df["DATE OCC"].dt.month
df["DAY"] = df["DATE OCC"].dt.day
df["WEEKDAY"] = df["DATE OCC"].dt.dayofweek
df["HOUR"] = df["TIME OCC"].fillna(0).astype(str).str.zfill(4).str[:2].astype(int)

df["DAY_OF_WEEK"] = df["DATE OCC"].dt.dayofweek
df["WEEK_OF_YEAR"] = df["DATE OCC"].dt.isocalendar().week.astype(int)
df["QUARTER"] = df["DATE OCC"].dt.quarter
df["MINUTE"] = (
    df["TIME OCC"].fillna(0).astype(int).astype(str).str.zfill(4).str[2:].astype(int)
)


def get_period(hour):

    if hour < 6:
        return "Night"

    elif hour < 12:
        return "Morning"

    elif hour < 18:
        return "Day"

    return "Evening"


df["PERIOD"] = df["HOUR"].apply(get_period)

df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)

df["REPORT_DELAY"] = (
    pd.to_datetime(df["Date Rptd"]) - pd.to_datetime(df["DATE OCC"])
).dt.days

df["REPORT_DELAY"] = df["REPORT_DELAY"].clip(lower=0)

# HEADER

st.title("Los Angeles Crime Analytics")
st.markdown(
    """Интерактивная аналитическая система для исследования преступности Лос-Анджелеса за 2020-2024 годы."""
)

# KPI

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Всего преступлений", f"{len(df):,}")

with col2:
    st.metric("Типов преступлений", df["Crm Cd Desc"].nunique())

with col3:
    st.metric("Районов", df["AREA NAME"].nunique())

with col4:
    st.metric("Период", "2020-2024")

# TOP 10 CRIMES

st.header("Топ-10 самых частых преступлений")
top_crimes = df["Crm Cd Desc"].value_counts().head(10).reset_index()
top_crimes.columns = ["Crime", "Count"]
fig = px.bar(top_crimes, x="Crime", y="Count", text="Count")

fig.update_layout(
    template="simple_white",
    height=550,
    showlegend=False,
    xaxis_title="",
    yaxis_title="Количество",
    margin=dict(l=20, r=20, t=40, b=20),
)

fig.update_traces(marker_color="black")
st.plotly_chart(fig, use_container_width=True)

# TOP 20 AREAS

st.header("20 самых криминальных районов")
area_table = df["AREA NAME"].value_counts().reset_index()
area_table.columns = ["Район", "Количество преступлений"]
area_table.index += 1
area_table.index.name = "Место"
st.dataframe(area_table.head(20), use_container_width=True, height=700)

# CRIME TYPE PREDICTION

st.header("Прогноз типа преступления")

with st.form("crime_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Район", sorted(df["AREA NAME"].dropna().unique()))
        victim_age = st.slider("Возраст жертвы", 0, 100, 30)
        victim_sex = st.selectbox("Пол", sorted(df["Vict Sex"].dropna().unique()))
        victim_descent = st.selectbox(
            "Этническая принадлежность", sorted(df["Vict Descent"].dropna().unique())
        )

    with col2:

        premis = st.selectbox(
            "Место преступления", sorted(df["Premis Desc"].dropna().unique())
        )
        weapon = st.selectbox("Оружие", sorted(df["Weapon Desc"].dropna().unique()))
        hour = st.slider("Час", 0, 23, 12)
    predict_crime = st.form_submit_button("Предсказать")

if predict_crime:
    row = pd.DataFrame(
        [
            {
                "AREA NAME": area,
                "Vict Age": victim_age,
                "Vict Sex": victim_sex,
                "Vict Descent": victim_descent,
                "Premis Desc": premis,
                "Weapon Desc": weapon,
                "LAT": df["LAT"].median(),
                "LON": df["LON"].median(),
                "YEAR": 2024,
                "MONTH": 6,
                "DAY": 1,
                "DAY_OF_WEEK": 5,
                "WEEK_OF_YEAR": 22,
                "QUARTER": 2,
                "HOUR": hour,
                "MINUTE": 0,
                "PERIOD": get_period(hour),
                "IS_WEEKEND": 1,
                "REPORT_DELAY": 0,
            }
        ]
    )

    prediction = crime_model.predict(row)[0]
    st.success(f"Предсказанный тип преступления:\n\n**{prediction}**")

    try:
        probs = crime_model.predict_proba(row)[0]
        labels = crime_model.classes_
        probs_df = pd.DataFrame({"Crime": labels, "Probability": probs})
        probs_df = probs_df.sort_values("Probability", ascending=False).head(5)
        fig = px.bar(probs_df, x="Crime", y="Probability", text="Probability")
        fig.update_layout(template="simple_white", showlegend=False)
        fig.update_traces(marker_color="black")
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass

# AREA RISK

st.header("Прогноз криминального риска района")

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_risk = st.selectbox("Район", sorted(df["AREA NAME"].dropna().unique()))
    with col2:
        hour_risk = st.slider("Час", 0, 23, 12)
    predict_risk = st.form_submit_button("Оценить риск")

if predict_risk:
    row = pd.DataFrame(
        [
            {
                "AREA NAME": area,
                "Vict Age": victim_age,
                "Vict Sex": victim_sex,
                "Vict Descent": victim_descent,
                "Premis Desc": premis,
                "Weapon Desc": weapon,
                "LAT": df["LAT"].median(),
                "LON": df["LON"].median(),
                "YEAR": 2024,
                "MONTH": 6,
                "DAY": 1,
                "DAY_OF_WEEK": 5,
                "WEEK_OF_YEAR": 22,
                "QUARTER": 2,
                "HOUR": hour,
                "MINUTE": 0,
                "PERIOD": get_period(hour),
                "IS_WEEKEND": 1,
                "REPORT_DELAY": 0,
            }
        ]
    )

    risk = risk_model.predict(row)[0]

    if risk == "High":
        st.error(f"Риск: {risk}")

    elif risk == "Medium":
        st.warning(f"Риск: {risk}")

    else:
        st.success(f"Риск: {risk}")

# HOTSPOT MAP

st.header("Прогнозирование очагов преступности")
st.markdown(
    """Карта показывает районы, где наблюдается наибольшая концентрация преступлений. Чем больше круг — тем чаще происходят преступления в этой точке."""
)

hotspots = df.groupby(["LAT", "LON"]).size().reset_index(name="count")
hotspots = hotspots[hotspots["LAT"] != 0]
hotspots = hotspots[hotspots["LON"] != 0]
hotspots = hotspots.sort_values("count", ascending=False)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=hotspots,
    get_position="[LON,LAT]",
    get_radius="count*5",
    get_fill_color=[0, 0, 0, 100],
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(latitude=34.0522, longitude=-118.2437, zoom=10, pitch=35)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Количество преступлений: {count}"},
)

st.pydeck_chart(deck)
st.success("LA Crime Analytics успешно загружен")
