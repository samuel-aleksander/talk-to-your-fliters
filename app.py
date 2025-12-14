import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from facet_schema import FACET_SCHEMA

 
# To use LLM extraction, set your Anthropic API key in one of these ways:
# 1. Streamlit secrets: Create .streamlit/secrets.toml with: ANTHROPIC_API_KEY = "your-key-here"
# 2. Environment variable: export ANTHROPIC_API_KEY="your-key-here"
# 
# If the anthropic library is not installed or the API key is not found,
# the app will return empty facets (no filters will be applied from the query).
try:
    from anthropic import Anthropic  # type: ignore
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

st.set_page_config(page_title="Talk to Your Filters", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("airbnb_final.parquet")

df = load_data()

# Clear query input if flag is set (from manual filter changes in previous run)
# This must happen before any widgets are created
if st.session_state.get("_clear_query_input", False):
    if "user_query_input" in st.session_state:
        st.session_state["user_query_input"] = ""
    del st.session_state["_clear_query_input"]

st.title("Talk to Your Filters")

st.markdown("## Natural-Language Faceted Search for Airbnb")

st.markdown("### Tell us what you are looking for")

st.markdown(
    "Describe your needs in plain language. The system will interpret your description and apply the corresponding filters."
    " You can always review, change, or remove these filters using the sidebar on the left."
)

# Initialize query input in session state if not exists
if "user_query_input" not in st.session_state:
    st.session_state["user_query_input"] = ""

# Initialize applied filters text in session state if not exists
if "_applied_filters_text" not in st.session_state:
    st.session_state["_applied_filters_text"] = ""

user_query = st.text_input(
    "We will automatically apply filters based on your description",
    placeholder="e.g. cheap 2 bedroom in Los Angeles with pool and wifi",
    key="user_query_input"
)

apply_query = st.button("Interpret query")

# Small utility: normalize a value into a list[str]
def _normalize_string_list(value: Any) -> List[str]:
    """Convert a string or list of strings into a list[str]. Non-strings -> []"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []

# Function to format actually applied filters from session_state
def format_actually_applied_filters(df: pd.DataFrame) -> str:
    """Format actually applied filters from session_state into a human-readable string.
    
    This function reads from the actual filter state values in session_state,
    not from the raw LLM output, ensuring the display matches what's actually applied.
    """
    filters = []
    
    # Location filters
    country_filter = st.session_state.get("country_filter", [])
    if country_filter:
        filters.append(f"Country: {', '.join(country_filter)}")
    
    state_filter = st.session_state.get("state_filter", [])
    if state_filter:
        filters.append(f"State: {', '.join(state_filter)}")
    
    city_filter = st.session_state.get("city_filter", [])
    if city_filter:
        filters.append(f"City: {', '.join(city_filter)}")
    
    neighborhood_filter = st.session_state.get("neighborhood_filter", [])
    if neighborhood_filter:
        filters.append(f"Neighborhood: {', '.join(neighborhood_filter)}")
    
    # Property and room types
    prop_type_filter = st.session_state.get("prop_type_filter", [])
    if prop_type_filter:
        filters.append(f"Property Type: {', '.join(prop_type_filter)}")
    
    room_type_filter = st.session_state.get("room_type_filter", [])
    if room_type_filter:
        filters.append(f"Room Type: {', '.join(room_type_filter)}")
    
    # Numeric filters - only show if they differ from defaults
    min_price, max_price = float(df["Price"].min()), float(df["Price"].max())
    price_filter = st.session_state.get("price_filter", (int(min_price), int(max_price)))
    if price_filter and price_filter[1] < max_price:
        filters.append(f"Max Price: ${price_filter[1]:.0f}")
    
    min_guests = int(df["Accommodates"].min())
    guests_filter = st.session_state.get("guests_filter", min_guests)
    if guests_filter > min_guests:
        filters.append(f"Min Guests: {guests_filter}")
    
    min_beds = int(df["Bedrooms"].min())
    bedrooms_filter = st.session_state.get("bedrooms_filter", min_beds)
    if bedrooms_filter > min_beds:
        filters.append(f"Min Bedrooms: {bedrooms_filter}")
    
    rating_filter = st.session_state.get("rating_filter", 0.0)
    if rating_filter > 0.0:
        filters.append(f"Min Rating: {rating_filter:.1f} stars")
    
    # Amenities - only show checked ones
    amenity_labels = []
    amenity_mapping = {
        "amenity_wifi_filter": "Wifi",
        "amenity_TV_filter": "TV",
        "amenity_kitchen_filter": "Kitchen",
        "amenity_heating_filter": "Heating",
        "amenity_air_conditioning_filter": "Air conditioning",
        "amenity_washer_filter": "Washer",
        "amenity_dryer_filter": "Dryer",
        "amenity_free_parking_filter": "Free parking",
        "amenity_pool_filter": "Pool",
        "amenity_hot_tub_filter": "Hot tub",
        "amenity_pet_friendly_filter": "Pet-friendly"
    }
    
    for key, label in amenity_mapping.items():
        if st.session_state.get(key, False):
            amenity_labels.append(label)
    
    if amenity_labels:
        filters.append(f"Amenities: {', '.join(amenity_labels)}")
    
    return " | ".join(filters) if filters else ""

# Function to format and display applied filters
def format_applied_filters(extracted: Dict[str, Any]) -> str:
    """Format extracted filters into a human-readable string."""
    filters = []
    
    # Location filters
    if extracted.get("country"):
        country_val = extracted["country"]
        if isinstance(country_val, list):
            filters.append(f"Country: {', '.join(country_val)}")
        else:
            filters.append(f"Country: {country_val}")
    
    if extracted.get("state"):
        state_val = extracted["state"]
        if isinstance(state_val, list):
            filters.append(f"State: {', '.join(state_val)}")
        else:
            filters.append(f"State: {state_val}")
    
    if extracted.get("city"):
        city_val = extracted["city"]
        if isinstance(city_val, list):
            filters.append(f"City: {', '.join(city_val)}")
        else:
            filters.append(f"City: {city_val}")
    
    if extracted.get("neighborhood"):
        filters.append(f"Neighborhood: {extracted['neighborhood']}")
    
    # Property and room types
    if extracted.get("property_types") and len(extracted["property_types"]) > 0:
        filters.append(f"Property Type: {', '.join(extracted['property_types'])}")
    
    if extracted.get("room_types") and len(extracted["room_types"]) > 0:
        filters.append(f"Room Type: {', '.join(extracted['room_types'])}")
    
    # Numeric filters
    if extracted.get("price_max") is not None:
        filters.append(f"Max Price: ${extracted['price_max']:.0f}")
    
    if extracted.get("guests_min") is not None:
        filters.append(f"Min Guests: {extracted['guests_min']}")
    
    if extracted.get("bedrooms_min") is not None:
        filters.append(f"Min Bedrooms: {extracted['bedrooms_min']}")
    
    if extracted.get("rating_min") is not None:
        filters.append(f"Min Rating: {extracted['rating_min']:.1f} stars")
    
    # Amenities
    if extracted.get("amenities") and len(extracted["amenities"]) > 0:
        filters.append(f"Amenities: {', '.join(extracted['amenities'])}")
    
    return " | ".join(filters) if filters else ""

# LOCATION FILTERS - Hierarchical multi-select

st.sidebar.subheader("Location")

# Create a placeholder for the applied filters display (early, for clearing purposes)
# The actual display will happen after Results section
applied_filters_placeholder = st.empty()

# Clear the placeholder if we're in the process of applying a new query
# This prevents showing stale filters during the rerun
if st.session_state.get("_applying_filters_from_query", False):
    applied_filters_placeholder.empty()

# Initialize as lists for multi-select
if "country_filter" not in st.session_state:
    st.session_state["country_filter"] = []
if "state_filter" not in st.session_state:
    st.session_state["state_filter"] = []
if "city_filter" not in st.session_state:
    st.session_state["city_filter"] = []
if "neighborhood_filter" not in st.session_state:
    st.session_state["neighborhood_filter"] = []

# Handle pending updates BEFORE creating widgets
# For location filters, replace (not append) when coming from a new query
if "_pending_country_update" in st.session_state:
    pending_country = st.session_state["_pending_country_update"]
    country_list = _normalize_string_list(pending_country)
    
    # Replace the filter if it's from a new query (indicated by _replace_filters flag)
    if st.session_state.get("_replace_location_filters", False):
        st.session_state["country_filter"] = country_list
    elif country_list:
        # Append new countries that aren't already in the filter
        for country in country_list:
            if country not in st.session_state["country_filter"]:
                st.session_state["country_filter"].append(country)
    elif pending_country is None and st.session_state.get("_replace_location_filters", False):
        st.session_state["country_filter"] = []
    del st.session_state["_pending_country_update"]

if "_pending_state_update" in st.session_state:
    pending_state = st.session_state["_pending_state_update"]
    state_list = _normalize_string_list(pending_state)
    
    # Replace the filter if it's from a new query
    if st.session_state.get("_replace_location_filters", False):
        st.session_state["state_filter"] = state_list
    elif state_list:
        # Append new states that aren't already in the filter
        for state in state_list:
            if state not in st.session_state["state_filter"]:
                st.session_state["state_filter"].append(state)
    elif pending_state is None and st.session_state.get("_replace_location_filters", False):
        st.session_state["state_filter"] = []
    del st.session_state["_pending_state_update"]

if "_pending_city_update" in st.session_state:
    pending_city = st.session_state["_pending_city_update"]
    city_list = _normalize_string_list(pending_city)
    
    # Replace the filter if it's from a new query
    if st.session_state.get("_replace_location_filters", False):
        st.session_state["city_filter"] = city_list
    elif city_list:
        # Append new cities that aren't already in the filter
        for city in city_list:
            if city not in st.session_state["city_filter"]:
                st.session_state["city_filter"].append(city)
    elif pending_city is None and st.session_state.get("_replace_location_filters", False):
        st.session_state["city_filter"] = []
    del st.session_state["_pending_city_update"]

if "_pending_neighborhood_update" in st.session_state:
    pending_neighborhood = st.session_state["_pending_neighborhood_update"]
    neighborhood_list = _normalize_string_list(pending_neighborhood)
    
    # Replace the filter if it's from a new query
    if st.session_state.get("_replace_location_filters", False):
        st.session_state["neighborhood_filter"] = neighborhood_list
    elif neighborhood_list:
        # Append new neighborhoods that aren't already in the filter
        for neighborhood in neighborhood_list:
            if neighborhood not in st.session_state["neighborhood_filter"]:
                st.session_state["neighborhood_filter"].append(neighborhood)
    elif pending_neighborhood is None and st.session_state.get("_replace_location_filters", False):
        st.session_state["neighborhood_filter"] = []
    del st.session_state["_pending_neighborhood_update"]

# Clear the replace flag after processing all location filters
if "_replace_location_filters" in st.session_state:
    del st.session_state["_replace_location_filters"]

# Country filter - multi-select
countries = sorted(df["Country"].dropna().unique())
# Filter out countries that are no longer valid - update session_state directly
st.session_state["country_filter"] = [c for c in st.session_state["country_filter"] if c in countries]
# Disable country filter if state, city, or neighborhood is selected
country_disabled = (len(st.session_state.get("state_filter", [])) > 0 or 
                    len(st.session_state.get("city_filter", [])) > 0 or 
                    len(st.session_state.get("neighborhood_filter", [])) > 0)
country_help = "Disabled because a state, city, or neighborhood is selected."
selected_country = st.sidebar.multiselect(
    "Countries",
    options=countries,
    key="country_filter",
    placeholder="Auto-filtered" if country_disabled else "Any",
    disabled=country_disabled,
    help=country_help if country_disabled else None
)
# Don't modify session_state here - widget handles it automatically

# State filter - filtered by selected countries
if st.session_state.get("country_filter"):
    state_df = df[df["Country"].isin(st.session_state["country_filter"])]
    states = sorted(state_df["State"].dropna().unique())
else:
    states = sorted(df["State"].dropna().unique())

# Filter out states that are no longer valid - update session_state directly
st.session_state["state_filter"] = [s for s in st.session_state["state_filter"] if s in states]
# Disable state filter if city or neighborhood is selected
state_disabled = (len(st.session_state.get("city_filter", [])) > 0 or 
                  len(st.session_state.get("neighborhood_filter", [])) > 0)
state_help = "Disabled because a city or neighborhood is selected."
selected_state = st.sidebar.multiselect(
    "States",
    options=states,
    key="state_filter",
    placeholder="Auto-filtered" if state_disabled else "Any",
    disabled=state_disabled,
    help=state_help if state_disabled else None
)
# Don't modify session_state here - widget handles it automatically


# City filter - filtered by selected countries and states
city_df = df.copy()
if st.session_state.get("country_filter"):
    city_df = city_df[city_df["Country"].isin(st.session_state["country_filter"])]
if st.session_state.get("state_filter"):
    city_df = city_df[city_df["State"].isin(st.session_state["state_filter"])]

cities = sorted(city_df["City"].dropna().unique())

# Filter out cities that are no longer valid - update session_state directly
st.session_state["city_filter"] = [c for c in st.session_state["city_filter"] if c in cities]
# Disable city filter if neighborhood is selected
city_disabled = len(st.session_state.get("neighborhood_filter", [])) > 0
city_help = "Disabled because a neighborhood is selected."
selected_city = st.sidebar.multiselect(
    "Cities",
    options=cities,
    key="city_filter",
    placeholder="Auto-filtered" if city_disabled else "Any",
    disabled=city_disabled,
    help=city_help if city_disabled else None
)
# Don't modify session_state here - widget handles it automatically


# Neighborhood filter - filtered by selected countries, states, and cities
neigh_df = city_df.copy()
if st.session_state.get("city_filter"):
    neigh_df = neigh_df[neigh_df["City"].isin(st.session_state["city_filter"])]

neighborhoods = sorted(neigh_df["Neighborhood"].dropna().unique())

# Filter out neighborhoods that are no longer valid - update session_state directly
st.session_state["neighborhood_filter"] = [n for n in st.session_state["neighborhood_filter"] if n in neighborhoods]
selected_neighborhood = st.sidebar.multiselect(
    "Neighborhoods",
    options=neighborhoods,
    key="neighborhood_filter",
    placeholder="Any"
)
# Don't modify session_state here - widget handles it automatically


# PROPERTY TYPE / ROOM TYPE FILTERS

st.sidebar.markdown("---")
st.sidebar.subheader("Listing Type")

prop_types = sorted(df["Property Type Normalized"].dropna().unique())

# Initialize as empty list for multi-select
if "prop_type_filter" not in st.session_state:
    st.session_state["prop_type_filter"] = []

# Check for pending property type update from query extraction
if "_pending_prop_type_update" in st.session_state:
    pending_prop_types = st.session_state["_pending_prop_type_update"]
    # Filter to only include valid property types
    valid_prop_types = [pt for pt in pending_prop_types if pt in prop_types]
    if valid_prop_types:
        st.session_state["prop_type_filter"] = valid_prop_types
    del st.session_state["_pending_prop_type_update"]

selected_prop_types = st.sidebar.multiselect(
    "Property type",
    options=prop_types,
    key="prop_type_filter",
    placeholder="Any"
)

room_types = sorted(df["Room Type"].dropna().unique())

# Initialize as empty list for multi-select
if "room_type_filter" not in st.session_state:
    st.session_state["room_type_filter"] = []

# Check for pending room type update from query extraction
if "_pending_room_type_update" in st.session_state:
    pending_room_types = st.session_state["_pending_room_type_update"]
    # Filter to only include valid room types
    valid_room_types = [rt for rt in pending_room_types if rt in room_types]
    if valid_room_types:
        st.session_state["room_type_filter"] = valid_room_types
    del st.session_state["_pending_room_type_update"]

selected_room_types = st.sidebar.multiselect(
    "Room type",
    options=room_types,
    key="room_type_filter",
    placeholder="Any"
)

# NUMERIC FILTERS

st.sidebar.markdown("---")
st.sidebar.subheader("Requirements")


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
    "Price per night (USD)",
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
    step=0.1,
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

# Detect manual filter changes and clear query context if user manually adjusts filters
# Initialize previous filter states if not exists
if "_prev_filter_states" not in st.session_state:
    st.session_state["_prev_filter_states"] = {
        "country_filter": st.session_state.get("country_filter", []).copy(),
        "state_filter": st.session_state.get("state_filter", []).copy(),
        "city_filter": st.session_state.get("city_filter", []).copy(),
        "neighborhood_filter": st.session_state.get("neighborhood_filter", []).copy(),
        "prop_type_filter": st.session_state.get("prop_type_filter", []).copy(),
        "room_type_filter": st.session_state.get("room_type_filter", []).copy(),
        "price_filter": st.session_state.get("price_filter", (int(min_price), int(max_price))),
        "guests_filter": st.session_state.get("guests_filter", min_guests),
        "bedrooms_filter": st.session_state.get("bedrooms_filter", min_beds),
        "rating_filter": st.session_state.get("rating_filter", 0.0),
        "amenity_wifi_filter": st.session_state.get("amenity_wifi_filter", False),
        "amenity_TV_filter": st.session_state.get("amenity_TV_filter", False),
        "amenity_kitchen_filter": st.session_state.get("amenity_kitchen_filter", False),
        "amenity_heating_filter": st.session_state.get("amenity_heating_filter", False),
        "amenity_air_conditioning_filter": st.session_state.get("amenity_air_conditioning_filter", False),
        "amenity_washer_filter": st.session_state.get("amenity_washer_filter", False),
        "amenity_dryer_filter": st.session_state.get("amenity_dryer_filter", False),
        "amenity_free_parking_filter": st.session_state.get("amenity_free_parking_filter", False),
        "amenity_pool_filter": st.session_state.get("amenity_pool_filter", False),
        "amenity_hot_tub_filter": st.session_state.get("amenity_hot_tub_filter", False),
        "amenity_pet_friendly_filter": st.session_state.get("amenity_pet_friendly_filter", False),
    }

# Check if any filter changed manually (not from pending updates)
# Only check if there are no pending updates and we're not applying filters from a query
# Also skip if we're already in the process of clearing (flag is set)
has_pending_updates = any(key.startswith("_pending_") for key in st.session_state.keys())
applying_from_query = st.session_state.get("_applying_filters_from_query", False)
clearing_query = st.session_state.get("_clear_query_input", False)

# Clear the applying flag after pending updates are processed
if applying_from_query and not has_pending_updates:
    del st.session_state["_applying_filters_from_query"]
    # Store filter text in session state to display after Results
    applied_filters_text = format_actually_applied_filters(df)
    st.session_state["_applied_filters_text"] = applied_filters_text

if not has_pending_updates and not applying_from_query and not clearing_query:
    current_states = {
        "country_filter": st.session_state.get("country_filter", []).copy(),
        "state_filter": st.session_state.get("state_filter", []).copy(),
        "city_filter": st.session_state.get("city_filter", []).copy(),
        "neighborhood_filter": st.session_state.get("neighborhood_filter", []).copy(),
        "prop_type_filter": st.session_state.get("prop_type_filter", []).copy(),
        "room_type_filter": st.session_state.get("room_type_filter", []).copy(),
        "price_filter": st.session_state.get("price_filter", (int(min_price), int(max_price))),
        "guests_filter": st.session_state.get("guests_filter", min_guests),
        "bedrooms_filter": st.session_state.get("bedrooms_filter", min_beds),
        "rating_filter": st.session_state.get("rating_filter", 0.0),
        "amenity_wifi_filter": st.session_state.get("amenity_wifi_filter", False),
        "amenity_TV_filter": st.session_state.get("amenity_TV_filter", False),
        "amenity_kitchen_filter": st.session_state.get("amenity_kitchen_filter", False),
        "amenity_heating_filter": st.session_state.get("amenity_heating_filter", False),
        "amenity_air_conditioning_filter": st.session_state.get("amenity_air_conditioning_filter", False),
        "amenity_washer_filter": st.session_state.get("amenity_washer_filter", False),
        "amenity_dryer_filter": st.session_state.get("amenity_dryer_filter", False),
        "amenity_free_parking_filter": st.session_state.get("amenity_free_parking_filter", False),
        "amenity_pool_filter": st.session_state.get("amenity_pool_filter", False),
        "amenity_hot_tub_filter": st.session_state.get("amenity_hot_tub_filter", False),
        "amenity_pet_friendly_filter": st.session_state.get("amenity_pet_friendly_filter", False),
    }
    
    prev_states = st.session_state["_prev_filter_states"]
    
    # Compare lists by converting to sets for comparison
    def lists_equal(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return set(a) == set(b)
        return a == b
    
    # Check if any filter changed
    filter_changed = False
    for key in current_states:
        if key in prev_states:
            if isinstance(current_states[key], list) and isinstance(prev_states[key], list):
                if set(current_states[key]) != set(prev_states[key]):
                    filter_changed = True
                    break
            elif current_states[key] != prev_states[key]:
                filter_changed = True
                break
    
    # If filters changed manually, clear query context
    if filter_changed and "last_extracted_filters" in st.session_state:
        del st.session_state["last_extracted_filters"]
        # Update previous states first to prevent re-detection on rerun
        st.session_state["_prev_filter_states"] = current_states.copy()
        # Set flag to clear query input (will be checked before widget creation in next run)
        st.session_state["_clear_query_input"] = True
        # Trigger rerun to clear the input
        st.rerun()
    else:
        # Update previous states for next comparison
        st.session_state["_prev_filter_states"] = current_states.copy()

# Store applied filters text in session state to display after Results
# Only update if we're not currently applying a new query (to prevent showing stale data)
if not st.session_state.get("_applying_filters_from_query", False):
    applied_filters_text = format_actually_applied_filters(df)
    st.session_state["_applied_filters_text"] = applied_filters_text

# APPLY FILTERS

filtered = df.copy()

# location - now using lists (multi-select)
if st.session_state.get("country_filter"):
    filtered = filtered[filtered["Country"].isin(st.session_state["country_filter"])]
if st.session_state.get("state_filter"):
    filtered = filtered[filtered["State"].isin(st.session_state["state_filter"])]
if st.session_state.get("city_filter"):
    filtered = filtered[filtered["City"].isin(st.session_state["city_filter"])]
if st.session_state.get("neighborhood_filter"):
    filtered = filtered[filtered["Neighborhood"].isin(st.session_state["neighborhood_filter"])]

# property / room type
if selected_prop_types:
    filtered = filtered[filtered["Property Type Normalized"].isin(selected_prop_types)]
if selected_room_types:
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


def _build_llm_prompt(query: str, available_locations: Dict[str, list]) -> str:
    """Build a prompt for the LLM to extract facets from a natural language query."""
    
    # Get all allowed values from FACET_SCHEMA
    property_types = FACET_SCHEMA["property_type"]["allowed_values"]
    room_types = FACET_SCHEMA["room_type"]["allowed_values"]
    
    # Get amenity labels
    amenity_labels = []
    for category, amenities in FACET_SCHEMA["amenities"]["categories"].items():
        amenity_labels.extend(amenities.keys())
    
    prompt = f"""You are a facet extraction assistant for an Airbnb search system. Your job is to interpret a natural language query and extract structured facet values.

IMPORTANT RULES:
1. Extract ONLY the facets mentioned in the query. If something is not mentioned, return null or empty list.
2. Use EXACTLY the allowed values listed below. Do not invent new values.
3. For location, use the EXACT city/state/country names from the available lists.
4. Return valid JSON only, no additional text.

AVAILABLE LOCATIONS:
Countries: {', '.join(available_locations.get('countries', []))}
States: {', '.join(available_locations.get('states', []))}
Cities: {', '.join(available_locations.get('cities', []))}

ALLOWED PROPERTY TYPES (use exact names):
{', '.join(property_types)}

ALLOWED ROOM TYPES (use exact names):
{', '.join(room_types)}

AVAILABLE AMENITIES (use exact labels):
{', '.join(amenity_labels)}

NUMERIC CONSTRAINTS:
- price_max: maximum price per night (positive number)
- guests_min: minimum number of guests (positive integer)
- bedrooms_min: minimum number of bedrooms (positive integer)
- rating_min: minimum rating in stars 0-5 (float, e.g., 4.0, 4.5)

OUTPUT FORMAT (JSON):
{{
    "country": "exact country name or null (or list for multiple)",
    "state": "exact state name or null (or list for multiple)",
    "city": "exact city name or null (or list for multiple)",
    "neighborhood": "exact neighborhood name or null (or list for multiple)",
    "property_types": ["list of allowed property types or empty list"],
    "room_types": ["list of allowed room types or empty list"],
    "price_max": number or null,
    "guests_min": integer or null,
    "bedrooms_min": integer or null,
    "rating_min": float or null,
    "amenities": ["list of amenity labels or empty list"]
}}

NOTE: For locations, you can return either:
- A single string: "Los Angeles"
- A list of strings: ["Los Angeles", "San Francisco"]
- null if not mentioned

USER QUERY: "{query}"

Extract the facets and return ONLY valid JSON:"""

    return prompt


def log_llm_query(query: str, llm_response: str, extracted_facets: Dict[str, Any] = None):
    """Log LLM queries and responses to a CSV file with each filter as a separate column."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "llm_query_logs.csv")
    timestamp = datetime.now().isoformat()
    
    # Prepare log entry with each filter as a separate column
    log_entry = {
        "timestamp": timestamp,
        "user_query": query,
        "llm_response": llm_response,
        # Location filters
        "extracted_country": extracted_facets.get("country") if extracted_facets else None,
        "extracted_state": extracted_facets.get("state") if extracted_facets else None,
        "extracted_city": extracted_facets.get("city") if extracted_facets else None,
        "extracted_neighborhood": extracted_facets.get("neighborhood") if extracted_facets else None,
        # Property/Room type filters (convert lists to comma-separated strings)
        "extracted_property_types": ", ".join(extracted_facets.get("property_types", [])) if extracted_facets and extracted_facets.get("property_types") else None,
        "extracted_room_types": ", ".join(extracted_facets.get("room_types", [])) if extracted_facets and extracted_facets.get("room_types") else None,
        # Numeric filters
        "extracted_price_max": extracted_facets.get("price_max") if extracted_facets else None,
        "extracted_guests_min": extracted_facets.get("guests_min") if extracted_facets else None,
        "extracted_bedrooms_min": extracted_facets.get("bedrooms_min") if extracted_facets else None,
        "extracted_rating_min": extracted_facets.get("rating_min") if extracted_facets else None,
        # Amenities (convert list to comma-separated string)
        "extracted_amenities": ", ".join(extracted_facets.get("amenities", [])) if extracted_facets and extracted_facets.get("amenities") else None,
    }
    
    # Append to CSV file
    try:
        # Check if file exists
        if os.path.exists(log_file):
            # Append to existing file
            log_df = pd.read_csv(log_file)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            # Create new file
            log_df = pd.DataFrame([log_entry])
        
        log_df.to_csv(log_file, index=False)
    except Exception as e:
        # Silently fail if logging fails - don't break the app
        pass


def _extract_facets_with_llm(query: str, available_locations: Dict[str, list]) -> Optional[Dict[str, Any]]:
    """Extract facets using Claude API. Returns None if LLM is unavailable or fails."""
    
    if not LLM_AVAILABLE:
        st.warning("LLM extraction is unavailable (missing the 'anthropic' package).")
        return None
    
    # Get API key from Streamlit secrets or environment variable
    api_key = None
    
    # Try to access secrets (may not be available in all environments)
    if hasattr(st, 'secrets'):
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, AttributeError, TypeError):
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY")
            except (AttributeError, TypeError):
                pass
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        st.error("❌ No API key found. Please set ANTHROPIC_API_KEY in Streamlit Cloud secrets or as an environment variable.")
        st.info("💡 In Streamlit Cloud: Go to Settings → Secrets and add: `ANTHROPIC_API_KEY = 'your-key-here'`")
        return None
    
    try:
        client = Anthropic(api_key=api_key)
        
        system_prompt = "You are a precise JSON extraction assistant. Return only valid JSON, no additional text."
        user_prompt = _build_llm_prompt(query, available_locations)
        
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Cheapest Claude model
            max_tokens=1024,
            temperature=0.1,  # Low temperature for consistent extraction
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse JSON response
        content = message.content[0].text
        raw_facets = json.loads(content)
        
        # Log the query and response
        log_llm_query(query, content, raw_facets)
        
        # Return raw facets without validation for now
        return raw_facets
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON response: {str(e)}")
        return None
    except Exception as e:
        error_type = type(e).__name__
        st.error(f"❌ API Error ({error_type}): {str(e)}")
        # Don't expose the full API key in error messages
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            st.info("💡 Tip: Check that your ANTHROPIC_API_KEY is correctly set in Streamlit Cloud secrets.")
        return None


def extract_facets_from_query(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract facet values from a natural language query using LLM.
    
    This function uses Claude to interpret the query and extract facet values.
    If the LLM is unavailable or fails, returns an empty facets dict.
    
    Args:
        query: Natural language search query
        df: DataFrame to get available location values for the prompt
    
    Returns:
        Dictionary with facet keys and extracted values
    """
    # Get available locations from dataframe for the prompt
    available_locations = {
        "countries": sorted(df["Country"].dropna().unique().tolist()),
        "states": sorted(df["State"].dropna().unique().tolist()),
        "cities": sorted(df["City"].dropna().unique().tolist())
    }
    
    # Try LLM extraction
    llm_facets = _extract_facets_with_llm(query, available_locations)
    
    if llm_facets is not None:
        return llm_facets
    
    # Return empty facets dict if LLM fails
    return {
        "country": None,
        "state": None,
        "city": None,
        "neighborhood": None,
        "property_types": [],
        "room_types": [],
        "price_max": None,
        "guests_min": None,
        "bedrooms_min": None,
        "rating_min": None,
        "amenities": []
    }

if apply_query and user_query:
    # Clear the applied filters display immediately to prevent showing stale data during rerun
    applied_filters_placeholder.empty()
    st.session_state["_applied_filters_text"] = ""
    
    # Set flag to replace location filters (not append) when applying a new query
    st.session_state["_replace_location_filters"] = True
    # Set flag to indicate filters are being applied from a query (prevents clearing query context)
    st.session_state["_applying_filters_from_query"] = True
    
    extracted = extract_facets_from_query(user_query, df)
    
    # Store extracted filters for display
    st.session_state["last_extracted_filters"] = extracted
    
    # country -> country_filter (supports single value or list)
    country_value = extracted.get("country")
    if country_value is not None:
        country_list = _normalize_string_list(country_value)
        
        # Check if countries exist in available countries
        available_countries = sorted(df["Country"].dropna().unique())
        valid_countries = [c for c in country_list if c in available_countries]
        if valid_countries:
            st.session_state["_pending_country_update"] = valid_countries if len(valid_countries) > 1 else valid_countries[0]
        else:
            st.session_state["_pending_country_update"] = None
    else:
        # Clear country filter if new query doesn't specify a country
        st.session_state["_pending_country_update"] = None

    # state -> state_filter (supports single value or list)
    state_value = extracted.get("state")
    if state_value is not None:
        state_list = _normalize_string_list(state_value)
        
        # Check if states exist in available states
        state_df_check = df.copy()
        if st.session_state.get("country_filter"):
            state_df_check = state_df_check[state_df_check["Country"].isin(st.session_state["country_filter"])]
        available_states = sorted(state_df_check["State"].dropna().unique())
        valid_states = [s for s in state_list if s in available_states]
        if valid_states:
            st.session_state["_pending_state_update"] = valid_states if len(valid_states) > 1 else valid_states[0]
        else:
            st.session_state["_pending_state_update"] = None
    else:
        # Clear state filter if new query doesn't specify a state
        st.session_state["_pending_state_update"] = None

    # city -> city_filter (supports single value or list)
    city_value = extracted.get("city")
    if city_value is not None:
        city_list = _normalize_string_list(city_value)
        
        # Check if cities exist in available cities
        city_df_check = df.copy()
        if st.session_state.get("country_filter"):
            city_df_check = city_df_check[city_df_check["Country"].isin(st.session_state["country_filter"])]
        if st.session_state.get("state_filter"):
            city_df_check = city_df_check[city_df_check["State"].isin(st.session_state["state_filter"])]
        available_cities = sorted(city_df_check["City"].dropna().unique())
        valid_cities = [c for c in city_list if c in available_cities]
        if valid_cities:
            st.session_state["_pending_city_update"] = valid_cities if len(valid_cities) > 1 else valid_cities[0]
        else:
            st.session_state["_pending_city_update"] = None
    else:
        # Clear city filter if new query doesn't specify a city
        st.session_state["_pending_city_update"] = None

    # neighborhood -> neighborhood_filter
    if extracted.get("neighborhood") is not None:
        # Check if the neighborhood exists in the available neighborhoods
        neigh_df_check = df.copy()
        if st.session_state.get("country_filter"):
            neigh_df_check = neigh_df_check[neigh_df_check["Country"].isin(st.session_state["country_filter"])]
        if st.session_state.get("state_filter"):
            neigh_df_check = neigh_df_check[neigh_df_check["State"].isin(st.session_state["state_filter"])]
        if st.session_state.get("city_filter"):
            neigh_df_check = neigh_df_check[neigh_df_check["City"].isin(st.session_state["city_filter"])]
        available_neighborhoods = sorted(neigh_df_check["Neighborhood"].dropna().unique())
        if extracted["neighborhood"] in available_neighborhoods:
            st.session_state["_pending_neighborhood_update"] = extracted["neighborhood"]
    else:
        # Clear neighborhood filter if new query doesn't specify a neighborhood
        st.session_state["_pending_neighborhood_update"] = None

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

# Display applied filters right under Results
applied_filters_text = st.session_state.get("_applied_filters_text", "")
if applied_filters_text:
    st.info(f"**Applied Filters:** {applied_filters_text}")

cols_to_show = [
    "ID", "Listing Url", "Neighborhood", "City", "State",
    "Property Type Normalized", "Property Type",
    "Room Type", "Accommodates", "Bedrooms", "Price", "rating_stars"
]

available_cols = [c for c in cols_to_show if c in filtered.columns]
st.dataframe(filtered[available_cols])