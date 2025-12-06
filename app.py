import streamlit as st
import pandas as pd
from facet_schema import FACET_SCHEMA

st.set_page_config(page_title="Airbnb Faceted Search", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("airbnb_final.parquet")

df = load_data()

st.title("Airbnb Faceted Search Prototype")

st.markdown("### What are you looking for?")
user_query = st.text_input(
    "Natural language query",
    placeholder="e.g. cheap 2 bedroom in Los Angeles with pool and wifi",
)

apply_query = st.button("Interpret query")

# LOCATION FILTERS

# Country filter
countries = sorted(df["Country"].dropna().unique())
country_options = ["Any"] + countries

# Check for pending country update from query extraction
if "_pending_country_update" in st.session_state:
    pending_country = st.session_state["_pending_country_update"]
    if pending_country in country_options:
        st.session_state["country_filter"] = pending_country
    del st.session_state["_pending_country_update"]

# Determine the index for the selectbox based on session state or default to 0
country_index = 0
if "country_filter" in st.session_state:
    country_value = st.session_state["country_filter"]
    if country_value in country_options:
        country_index = country_options.index(country_value)

selected_country = st.sidebar.selectbox("Country", country_options, index=country_index, key="country_filter")

# State filter
if selected_country != "Any":
    state_df = df[df["Country"] == selected_country]
    states = sorted(state_df["State"].dropna().unique())
else:
    states = sorted(df["State"].dropna().unique())

state_options = ["Any"] + list(states)

# Check for pending state update from query extraction
if "_pending_state_update" in st.session_state:
    pending_state = st.session_state["_pending_state_update"]
    if pending_state in state_options:
        st.session_state["state_filter"] = pending_state
    del st.session_state["_pending_state_update"]

# Determine the index for the selectbox based on session state or default to 0
state_index = 0
if "state_filter" in st.session_state:
    state_value = st.session_state["state_filter"]
    if state_value in state_options:
        state_index = state_options.index(state_value)

selected_state = st.sidebar.selectbox("State", state_options, index=state_index, key="state_filter")

# City filter
city_df = df.copy()
if selected_country != "Any":
    city_df = city_df[city_df["Country"] == selected_country]
if selected_state != "Any":
    city_df = city_df[city_df["State"] == selected_state]

cities = sorted(city_df["City"].dropna().unique())
city_options = ["Any"] + list(cities)

# Check for pending city update from query extraction
if "_pending_city_update" in st.session_state:
    pending_city = st.session_state["_pending_city_update"]
    if pending_city in city_options:
        # Set the city filter value before creating the widget
        st.session_state["city_filter"] = pending_city
    # Clear the pending update
    del st.session_state["_pending_city_update"]

# Determine the index for the selectbox based on session state or default to 0
city_index = 0
if "city_filter" in st.session_state:
    city_value = st.session_state["city_filter"]
    if city_value in city_options:
        city_index = city_options.index(city_value)
    # If the stored value is not in current options, just use index 0 (don't modify session_state)

selected_city = st.sidebar.selectbox("City", city_options, index=city_index, key="city_filter")

# Neighborhood filter
neigh_df = city_df.copy()
if selected_city != "Any":
    neigh_df = neigh_df[neigh_df["City"] == selected_city]

neighbourhoods = sorted(neigh_df["Neighbourhood"].dropna().unique())
neighbourhood_options = ["Any"] + list(neighbourhoods)

# Check for pending neighbourhood update from query extraction
if "_pending_neighbourhood_update" in st.session_state:
    pending_neighbourhood = st.session_state["_pending_neighbourhood_update"]
    if pending_neighbourhood in neighbourhood_options:
        st.session_state["neighbourhood_filter"] = pending_neighbourhood
    del st.session_state["_pending_neighbourhood_update"]

# Determine the index for the selectbox based on session state or default to 0
neighbourhood_index = 0
if "neighbourhood_filter" in st.session_state:
    neighbourhood_value = st.session_state["neighbourhood_filter"]
    if neighbourhood_value in neighbourhood_options:
        neighbourhood_index = neighbourhood_options.index(neighbourhood_value)

selected_neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhood_options, index=neighbourhood_index, key="neighbourhood_filter")

# PROPERTY TYPE / ROOM TYPE FILTERS

st.sidebar.markdown("---")

prop_types = sorted(df["Property Type Normalized"].dropna().unique())

# Check for pending property type update from query extraction
if "_pending_prop_type_update" in st.session_state:
    pending_prop_types = st.session_state["_pending_prop_type_update"]
    # Filter to only include valid property types
    valid_prop_types = [pt for pt in pending_prop_types if pt in prop_types]
    if valid_prop_types:
        st.session_state["prop_type_filter"] = valid_prop_types
    del st.session_state["_pending_prop_type_update"]

# Get initial property types from session state or use defaults
if "prop_type_filter" in st.session_state:
    prop_type_value = st.session_state["prop_type_filter"]
else:
    prop_type_value = prop_types

selected_prop_types = st.sidebar.multiselect(
    "Property type",
    options=prop_types,
    default=prop_type_value,
    key="prop_type_filter"
)

room_types = sorted(df["Room Type"].dropna().unique())

# Check for pending room type update from query extraction
if "_pending_room_type_update" in st.session_state:
    pending_room_types = st.session_state["_pending_room_type_update"]
    # Filter to only include valid room types
    valid_room_types = [rt for rt in pending_room_types if rt in room_types]
    if valid_room_types:
        st.session_state["room_type_filter"] = valid_room_types
    del st.session_state["_pending_room_type_update"]

# Get initial room types from session state or use defaults
if "room_type_filter" in st.session_state:
    room_type_value = st.session_state["room_type_filter"]
else:
    room_type_value = room_types

selected_room_types = st.sidebar.multiselect(
    "Room type",
    options=room_types,
    default=room_type_value,
    key="room_type_filter"
)

# NUMERIC FILTERS

st.sidebar.markdown("---")

min_price, max_price = float(df["Price"].min()), float(df["Price"].max())

# Check for pending price update from query extraction
if "_pending_price_update" in st.session_state:
    pending_price_high = st.session_state["_pending_price_update"]
    if "price_filter" in st.session_state:
        low, high = st.session_state["price_filter"]
        new_high = min(high, pending_price_high)
        st.session_state["price_filter"] = (low, new_high)
    else:
        st.session_state["price_filter"] = (int(min_price), min(int(max_price), pending_price_high))
    del st.session_state["_pending_price_update"]

# Get initial price range from session state or use defaults
if "price_filter" in st.session_state:
    price_value = st.session_state["price_filter"]
else:
    price_value = (int(min_price), int(max_price))

price_range = st.sidebar.slider(
    "Price per night",
    min_value=int(min_price),
    max_value=int(max_price),
    value=price_value,
    key="price_filter"
)

min_guests, max_guests = int(df["Accommodates"].min()), int(df["Accommodates"].max())

# Check for pending guests update from query extraction
if "_pending_guests_update" in st.session_state:
    pending_guests = st.session_state["_pending_guests_update"]
    if min_guests <= pending_guests <= max_guests:
        st.session_state["guests_filter"] = pending_guests
    del st.session_state["_pending_guests_update"]

# Get initial guests value from session state or use default
if "guests_filter" in st.session_state:
    guests_value = st.session_state["guests_filter"]
else:
    guests_value = min_guests

guests_min = st.sidebar.slider(
    "Minimum guests",
    min_value=min_guests,
    max_value=max_guests,
    value=guests_value,
    key="guests_filter"
)

min_beds, max_beds = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())

# Check for pending bedrooms update from query extraction
if "_pending_bedrooms_update" in st.session_state:
    pending_bedrooms = st.session_state["_pending_bedrooms_update"]
    if min_beds <= pending_bedrooms <= max_beds:
        st.session_state["bedrooms_filter"] = pending_bedrooms
    del st.session_state["_pending_bedrooms_update"]

# Get initial bedrooms value from session state or use default
if "bedrooms_filter" in st.session_state:
    bedrooms_value = st.session_state["bedrooms_filter"]
else:
    bedrooms_value = min_beds

bedrooms_min = st.sidebar.slider(
    "Minimum bedrooms",
    min_value=min_beds,
    max_value=max_beds,
    value=bedrooms_value,
    key="bedrooms_filter"
)

# rating_stars should be 0–5 (or close)
min_rating, max_rating = float(df["rating_stars"].min()), float(df["rating_stars"].max())

# Check for pending rating update from query extraction
if "_pending_rating_update" in st.session_state:
    pending_rating = st.session_state["_pending_rating_update"]
    if 0.0 <= pending_rating <= 5.0:
        st.session_state["rating_filter"] = pending_rating
    del st.session_state["_pending_rating_update"]

# Get initial rating value from session state or use default
if "rating_filter" in st.session_state:
    rating_value = st.session_state["rating_filter"]
else:
    rating_value = 0.0

rating_min = st.sidebar.slider(
    "Minimum rating (stars)",
    min_value=0.0,
    max_value=5.0,
    value=rating_value,
    step=0.5,
    key="rating_filter"
)

# AMENITIES FILTERS

st.sidebar.markdown("---")
st.sidebar.subheader("Amenities")

# Check for pending amenity updates from query extraction
if "_pending_amenity_wifi_update" in st.session_state:
    st.session_state["amenity_wifi_filter"] = st.session_state["_pending_amenity_wifi_update"]
    del st.session_state["_pending_amenity_wifi_update"]

if "_pending_amenity_pool_update" in st.session_state:
    st.session_state["amenity_pool_filter"] = st.session_state["_pending_amenity_pool_update"]
    del st.session_state["_pending_amenity_pool_update"]

amenity_checks = {}

with st.sidebar.expander("Basics", expanded=False):
    amenity_checks["amenity_wifi"] = st.checkbox("Wifi", key="amenity_wifi_filter")
    amenity_checks["amenity_TV"] = st.checkbox("TV", key="amenity_TV_filter")
    amenity_checks["amenity_kitchen"] = st.checkbox("Kitchen", key="amenity_kitchen_filter")
    amenity_checks["amenity_heating"] = st.checkbox("Heating", key="amenity_heating_filter")
    amenity_checks["amenity_air_conditioning"] = st.checkbox("Air conditioning", key="amenity_air_conditioning_filter")

with st.sidebar.expander("Laundry", expanded=False):
    amenity_checks["amenity_washer"] = st.checkbox("Washer", key="amenity_washer_filter")
    amenity_checks["amenity_dryer"] = st.checkbox("Dryer", key="amenity_dryer_filter")

with st.sidebar.expander("Parking", expanded=False):
    amenity_checks["amenity_free_parking"] = st.checkbox("Free parking", key="amenity_free_parking_filter")

with st.sidebar.expander("Leisure", expanded=False):
    amenity_checks["amenity_pool"] = st.checkbox("Pool", key="amenity_pool_filter")
    amenity_checks["amenity_hot_tub"] = st.checkbox("Hot tub", key="amenity_hot_tub_filter")

with st.sidebar.expander("Pets", expanded=False):
    amenity_checks["amenity_pet_friendly"] = st.checkbox("Pet-friendly", key="amenity_pet_friendly_filter" )

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


# temporary stub to extract facets from query
def extract_facets_from_query(query: str) -> dict:
    
    q = query.lower()
    facets = {
        "country": None,
        "state": None,
        "city": None,
        "neighbourhood": None,
        "property_types": [],
        "room_types": [],
        "price_max": None,
        "guests_min": None,
        "bedrooms_min": None,
        "rating_min": None,
        "amenities": []
    }

    # Location filters
    if "los angeles" in q:
        facets["city"] = "Los Angeles"
    if "san francisco" in q:
        facets["city"] = "San Francisco"
   
    # Price
    if "cheap" in q or "under 150" in q:
        facets["price_max"] = 150

    # Guests/Accommodates

    # Bedrooms
    if "2 bedroom" in q or "two bedroom" in q:
        facets["bedrooms_min"] = 2

    # Amenities
    if "wifi" in q:
        facets["amenities"].append("Wifi")
    if "pool" in q:
        facets["amenities"].append("Pool")

    return facets

if apply_query and user_query:
    extracted = extract_facets_from_query(user_query)

    # country -> country_filter
    if extracted.get("country") is not None:
        # Check if the country exists in the available countries
        available_countries = sorted(df["Country"].dropna().unique())
        if extracted["country"] in available_countries:
            st.session_state["_pending_country_update"] = extracted["country"]

    # state -> state_filter
    if extracted.get("state") is not None:
        # Check if the state exists in the available states
        state_df_check = df.copy()
        if selected_country != "Any":
            state_df_check = state_df_check[state_df_check["Country"] == selected_country]
        available_states = sorted(state_df_check["State"].dropna().unique())
        if extracted["state"] in available_states:
            st.session_state["_pending_state_update"] = extracted["state"]

    # city -> city_filter
    if extracted.get("city") is not None:
        # Check if the city exists in the available cities
        city_df_check = df.copy()
        if selected_country != "Any":
            city_df_check = city_df_check[city_df_check["Country"] == selected_country]
        if selected_state != "Any":
            city_df_check = city_df_check[city_df_check["State"] == selected_state]
        available_cities = sorted(city_df_check["City"].dropna().unique())
        if extracted["city"] in available_cities:
            st.session_state["_pending_city_update"] = extracted["city"]

    # neighbourhood -> neighbourhood_filter
    if extracted.get("neighbourhood") is not None:
        # Check if the neighbourhood exists in the available neighbourhoods
        neigh_df_check = df.copy()
        if selected_country != "Any":
            neigh_df_check = neigh_df_check[neigh_df_check["Country"] == selected_country]
        if selected_state != "Any":
            neigh_df_check = neigh_df_check[neigh_df_check["State"] == selected_state]
        if selected_city != "Any":
            neigh_df_check = neigh_df_check[neigh_df_check["City"] == selected_city]
        available_neighbourhoods = sorted(neigh_df_check["Neighbourhood"].dropna().unique())
        if extracted["neighbourhood"] in available_neighbourhoods:
            st.session_state["_pending_neighbourhood_update"] = extracted["neighbourhood"]

    # property_types -> prop_type_filter
    property_types_list = extracted.get("property_types", [])
    if property_types_list is not None and len(property_types_list) > 0:
        # Filter to only include valid property types
        valid_prop_types = sorted(df["Property Type Normalized"].dropna().unique())
        filtered_prop_types = [pt for pt in property_types_list if pt in valid_prop_types]
        if filtered_prop_types:
            st.session_state["_pending_prop_type_update"] = filtered_prop_types

    # room_types -> room_type_filter
    room_types_list = extracted.get("room_types", [])
    if room_types_list is not None and len(room_types_list) > 0:
        # Filter to only include valid room types
        valid_room_types = sorted(df["Room Type"].dropna().unique())
        filtered_room_types = [rt for rt in room_types_list if rt in valid_room_types]
        if filtered_room_types:
            st.session_state["_pending_room_type_update"] = filtered_room_types

    # price_max -> upper end of price slider
    if extracted.get("price_max") is not None:
        st.session_state["_pending_price_update"] = extracted["price_max"]

    # guests_min -> guests_filter
    if extracted.get("guests_min") is not None:
        st.session_state["_pending_guests_update"] = extracted["guests_min"]

    # bedrooms_min -> bedrooms_filter
    if extracted.get("bedrooms_min") is not None:
        st.session_state["_pending_bedrooms_update"] = extracted["bedrooms_min"]

    # rating_min -> rating_filter
    if extracted.get("rating_min") is not None:
        st.session_state["_pending_rating_update"] = extracted["rating_min"]

    # amenities -> checkboxes
    amenities_list = extracted.get("amenities", [])
    if amenities_list is not None and len(amenities_list) > 0:
        if "Wifi" in amenities_list:
            st.session_state["_pending_amenity_wifi_update"] = True
        if "Pool" in amenities_list:
            st.session_state["_pending_amenity_pool_update"] = True

    # force Streamlit to rerun with updated widget values
    st.rerun()



# SHOW RESULTS

st.subheader(f"Results ({len(filtered)} listings)")

cols_to_show = [
    "ID", "Listing Url", "Neighbourhood", "City", "State",
    "Property Type Normalized", "Property Type",
    "Room Type", "Accommodates", "Bedrooms", "Price", "rating_stars"
]

available_cols = [c for c in cols_to_show if c in filtered.columns]
st.dataframe(filtered[available_cols])