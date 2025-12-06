import streamlit as st
import pandas as pd
from facet_schema import FACET_SCHEMA

st.set_page_config(page_title="Airbnb Faceted Search", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("airbnb_final.parquet")

df = load_data()

st.title("Airbnb Faceted Search Prototype")

# ------------------------------
# Location facets (initial UI)
# ------------------------------

# Country filter
countries = sorted(df["Country"].dropna().unique())
selected_country = st.sidebar.selectbox("Country", ["Any"] + countries)

# State filter
if selected_country != "Any":
    state_df = df[df["Country"] == selected_country]
    states = sorted(state_df["State"].dropna().unique())
else:
    states = sorted(df["State"].dropna().unique())

selected_state = st.sidebar.selectbox("State", ["Any"] + list(states))

# City filter
city_df = df.copy()
if selected_country != "Any":
    city_df = city_df[city_df["Country"] == selected_country]
if selected_state != "Any":
    city_df = city_df[city_df["State"] == selected_state]

cities = sorted(city_df["City"].dropna().unique())
selected_city = st.sidebar.selectbox("City", ["Any"] + list(cities))

# Neighborhood filter
neigh_df = city_df.copy()
if selected_city != "Any":
    neigh_df = neigh_df[neigh_df["City"] == selected_city]

neighbourhoods = sorted(neigh_df["Neighbourhood Cleansed"].dropna().unique())
selected_neighbourhood = st.sidebar.selectbox("Neighbourhood", ["Any"] + list(neighbourhoods))

# --------------------------------
# Basic display for now
# --------------------------------

st.subheader("Data Preview")
st.dataframe(df.head(50))
