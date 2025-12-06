import streamlit as st
import pandas as pd
from facet_schema import FACET_SCHEMA

st.set_page_config(page_title="Airbnb Faceted Search", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("airbnb_final.parquet")

df = load_data()

st.title("Airbnb Faceted Search Prototype")


# LOCATION FILTERS

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

neighbourhoods = sorted(neigh_df["Neighbourhood"].dropna().unique())
selected_neighbourhood = st.sidebar.selectbox("Neighbourhood", ["Any"] + list(neighbourhoods))

# PROPERTY TYPE / ROOM TYPE FILTERS

st.sidebar.markdown("---")

prop_types = sorted(df["Property Type Normalized"].dropna().unique())
selected_prop_types = st.sidebar.multiselect(
    "Property type",
    options=prop_types,
    default=prop_types
)

room_types = sorted(df["Room Type"].dropna().unique())
selected_room_types = st.sidebar.multiselect(
    "Room type",
    options=room_types,
    default=room_types
)

# NUMERIC FILTERS

st.sidebar.markdown("---")

min_price, max_price = float(df["Price"].min()), float(df["Price"].max())
price_range = st.sidebar.slider(
    "Price per night",
    min_value=int(min_price),
    max_value=int(max_price),
    value=(int(min_price), int(max_price))
)

min_guests, max_guests = int(df["Accommodates"].min()), int(df["Accommodates"].max())
guests_min = st.sidebar.slider(
    "Minimum guests",
    min_value=min_guests,
    max_value=max_guests,
    value=min_guests
)

min_beds, max_beds = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
bedrooms_min = st.sidebar.slider(
    "Minimum bedrooms",
    min_value=min_beds,
    max_value=max_beds,
    value=min_beds
)

# rating_stars should be 0–5 (or close)
min_rating, max_rating = float(df["rating_stars"].min()), float(df["rating_stars"].max())
rating_min = st.sidebar.slider(
    "Minimum rating (stars)",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)

# AMENITIES FILTERS

st.sidebar.markdown("---")
st.sidebar.subheader("Amenities")

amenity_checks = {}

with st.sidebar.expander("Basics", expanded=False):
    amenity_checks["amenity_wifi"] = st.checkbox("Wifi")
    amenity_checks["amenity_TV"] = st.checkbox("TV")
    amenity_checks["amenity_kitchen"] = st.checkbox("Kitchen")
    amenity_checks["amenity_heating"] = st.checkbox("Heating")
    amenity_checks["amenity_air_conditioning"] = st.checkbox("Air conditioning")

with st.sidebar.expander("Laundry", expanded=False):
    amenity_checks["amenity_washer"] = st.checkbox("Washer")
    amenity_checks["amenity_dryer"] = st.checkbox("Dryer")

with st.sidebar.expander("Parking", expanded=False):
    amenity_checks["amenity_free_parking"] = st.checkbox("Free parking")

with st.sidebar.expander("Leisure", expanded=False):
    amenity_checks["amenity_pool"] = st.checkbox("Pool")
    amenity_checks["amenity_hot_tub"] = st.checkbox("Hot tub")

with st.sidebar.expander("Pets", expanded=False):
    amenity_checks["amenity_pet_friendly"] = st.checkbox("Pet-friendly")

# APPLY FILTERS

filtered = df.copy()

# location
if selected_country != "Any":
    filtered = filtered[filtered["Country"] == selected_country]
if selected_state != "Any":
    filtered = filtered[filtered["State"] == selected_state]
if selected_city != "Any":
    filtered = filtered[filtered["City"] == selected_city]
if selected_neighbourhood != "Any":
    filtered = filtered[filtered["Neighbourhood"] == selected_neighbourhood]

# property / room type
filtered = filtered[filtered["Property Type Normalized"].isin(selected_prop_types)]
filtered = filtered[filtered["Room Type"].isin(selected_room_types)]

# numeric
filtered = filtered[
    (filtered["Price"] >= price_range[0]) &
    (filtered["Price"] <= price_range[1]) &
    (filtered["Accommodates"] >= guests_min) &
    (filtered["Bedrooms"] >= bedrooms_min)
]

if rating_min > 0:
    filtered = filtered[ (filtered["rating_stars"] >= rating_min) ]

# amenities (AND logic: must have all checked amenities)
for col, must_have in amenity_checks.items():
    if must_have:
        if col in filtered.columns:
            filtered = filtered[filtered[col] == True]

# SHOW RESULTS

st.subheader(f"Results ({len(filtered)} listings)")

cols_to_show = [
    "ID", "Listing Url", "Neighbourhood", "City", "State",
    "Property Type Normalized", "Property Type",
    "Room Type", "Accommodates", "Bedrooms", "Price", "rating_stars"
]

available_cols = [c for c in cols_to_show if c in filtered.columns]
st.dataframe(filtered[available_cols])