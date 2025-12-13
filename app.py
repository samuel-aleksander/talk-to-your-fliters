import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from facet_schema import FACET_SCHEMA

# LLM imports - try Anthropic Claude first, fallback to rule-based if not available
# Install with: pip install anthropic
# 
# To use LLM extraction, set your Anthropic API key in one of these ways:
# 1. Streamlit secrets: Create .streamlit/secrets.toml with: ANTHROPIC_API_KEY = "your-key-here"
# 2. Environment variable: export ANTHROPIC_API_KEY="your-key-here"
# 
# If no API key is found, the app will automatically fall back to rule-based extraction.
try:
    from anthropic import Anthropic  # type: ignore
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

st.set_page_config(page_title="Airbnb Faceted Search", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("airbnb_final.parquet")

df = load_data()

st.title("Airbnb Faceted Search Prototype")

st.markdown("### Tell us what you are looking for")
user_query = st.text_input(
    "We will automatically apply filters based on your description",
    placeholder="e.g. cheap 2 bedroom in Los Angeles with pool and wifi",
)

apply_query = st.button("Apply filters")

# LOCATION FILTERS - Hierarchical multi-select

st.sidebar.subheader("Location")

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
    # Normalize to list: convert single value to list, keep list as-is
    if isinstance(pending_country, list):
        country_list = pending_country
    elif pending_country is not None:
        country_list = [pending_country]
    else:
        country_list = []
    
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
    # Normalize to list: convert single value to list, keep list as-is
    if isinstance(pending_state, list):
        state_list = pending_state
    elif pending_state is not None:
        state_list = [pending_state]
    else:
        state_list = []
    
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
    # Normalize to list: convert single value to list, keep list as-is
    if isinstance(pending_city, list):
        city_list = pending_city
    elif pending_city is not None:
        city_list = [pending_city]
    else:
        city_list = []
    
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
    # Normalize to list: convert single value to list, keep list as-is
    if isinstance(pending_neighborhood, list):
        neighborhood_list = pending_neighborhood
    elif pending_neighborhood is not None:
        neighborhood_list = [pending_neighborhood]
    else:
        neighborhood_list = []
    
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
selected_country = st.sidebar.multiselect(
    "Countries",
    options=countries,
    key="country_filter",
    placeholder="Any"
)
# Don't modify session_state here - widget handles it automatically

# State filter - filtered by selected countries
if selected_country:
    state_df = df[df["Country"].isin(selected_country)]
    states = sorted(state_df["State"].dropna().unique())
else:
    states = sorted(df["State"].dropna().unique())

# Filter out states that are no longer valid
valid_states = [s for s in st.session_state["state_filter"] if s in states]
selected_state = st.sidebar.multiselect(
    "States",
    options=states,
    default=valid_states,
    key="state_filter",
    placeholder="Any"
)
# Don't modify session_state here - widget handles it automatically

# City filter - filtered by selected countries and states
city_df = df.copy()
if selected_country:
    city_df = city_df[city_df["Country"].isin(selected_country)]
if selected_state:
    city_df = city_df[city_df["State"].isin(selected_state)]

cities = sorted(city_df["City"].dropna().unique())

# Filter out cities that are no longer valid
valid_cities = [c for c in st.session_state["city_filter"] if c in cities]
selected_city = st.sidebar.multiselect(
    "Cities",
    options=cities,
    default=valid_cities,
    key="city_filter",
    placeholder="Any"
)
# Don't modify session_state here - widget handles it automatically

# Neighborhood filter - filtered by selected countries, states, and cities
neigh_df = city_df.copy()
if selected_city:
    neigh_df = neigh_df[neigh_df["City"].isin(selected_city)]

neighborhoods = sorted(neigh_df["Neighborhood"].dropna().unique())

# Filter out neighborhoods that are no longer valid
valid_neighborhoods = [n for n in st.session_state["neighborhood_filter"] if n in neighborhoods]
selected_neighborhood = st.sidebar.multiselect(
    "Neighborhoods",
    options=neighborhoods,
    default=valid_neighborhoods,
    key="neighborhood_filter",
    placeholder="Any"
)
# Don't modify session_state here - widget handles it automatically

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
        st.write("❌ LLM not available")
        return None
    
    # Get API key from Streamlit secrets or environment variable
    api_key = None
    
    # Try to access secrets (may not be available in all environments)
    if hasattr(st, 'secrets'):
        try:
            # Try dictionary-style access first (works in Streamlit Cloud)
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, AttributeError, TypeError) as e:
            # If that fails, try .get() method
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
    
    # Debug: Check if API key was found (without exposing it)
    if api_key:
        st.write("✅ API key found, making API call...")
    else:
        st.error("❌ API key not found")
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
        st.write("**LLM Response:**", content)
        raw_facets = json.loads(content)
        st.write("✅ API call successful")
        
        # Log the query and response
        log_llm_query(query, content, raw_facets)
        
        # Return raw facets without validation for now
        return raw_facets
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON response: {str(e)}")
        if 'content' in locals():
            st.write("**Raw response:**", content)
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
    # Set flag to replace location filters (not append) when applying a new query
    st.session_state["_replace_location_filters"] = True
    
    extracted = extract_facets_from_query(user_query, df)
    
    # Simple debug output
    st.write("**Extracted:**", extracted)
    
    # country -> country_filter (supports single value or list)
    country_value = extracted.get("country")
    if country_value is not None:
        # Normalize to list: convert single string to list, keep list as-is
        if isinstance(country_value, str):
            country_list = [country_value]
        elif isinstance(country_value, list):
            country_list = country_value
        else:
            country_list = []
        
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
        # Normalize to list: convert single string to list, keep list as-is
        if isinstance(state_value, str):
            state_list = [state_value]
        elif isinstance(state_value, list):
            state_list = state_value
        else:
            state_list = []
        
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
        # Normalize to list: convert single string to list, keep list as-is
        if isinstance(city_value, str):
            city_list = [city_value]
        elif isinstance(city_value, list):
            city_list = city_value
        else:
            city_list = []
        
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

cols_to_show = [
    "ID", "Listing Url", "Neighborhood", "City", "State",
    "Property Type Normalized", "Property Type",
    "Room Type", "Accommodates", "Bedrooms", "Price", "rating_stars"
]

available_cols = [c for c in cols_to_show if c in filtered.columns]
st.dataframe(filtered[available_cols])