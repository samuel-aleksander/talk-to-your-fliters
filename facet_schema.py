FACET_SCHEMA = {
    "location": {
        "type": "hierarchical",
        "levels": ["country", "state", "city", "neighbourhood"],
        "fields": ["Country", "State", "City", "Neighbourhood"]
    },

    "property_type": {
        "type": "categorical",
        "field": "Property Type Normalized",
        "allowed_values": [
            "Apartment",
            "House",
            "Condominium",
            "Loft",
            "Townhouse",
            "Guesthouse",
            "Bed & Breakfast",
            "Bungalow",
            "Unique stay"
        ]
    },

    "room_type": {
        "type": "categorical",
        "field": "Room Type",
        "allowed_values": [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    },

    "accommodates_min": {
        "type": "numeric_min",
        "field": "Accommodates"
    },

    "bedrooms_min": {
        "type": "numeric_min",
        "field": "Bedrooms"
    },

    "price_max": {
        "type": "numeric_max",
        "field": "Price"
    },

    "rating_min": {
        "type": "numeric_min",
        "field": "rating_stars",
        "scale": "0_to_5"
    },

    "amenities": {
        "type": "multi_boolean_hierarchical",
        "categories": {
            "Basics": {
                "Wifi": "amenity_wifi",
                "TV": "amenity_TV",
                "Kitchen": "amenity_kitchen",
                "Heating": "amenity_heating",
                "Air conditioning": "amenity_air_conditioning"
            },
            "Laundry": {
                "Washer": "amenity_washer",
                "Dryer": "amenity_dryer"
            },
            "Parking": {
                "Free parking": "amenity_free_parking"
            },
            "Leisure": {
                "Pool": "amenity_pool",
                "Hot tub": "amenity_hot_tub"
            },
            "Pets": {
                "Pet-friendly": "amenity_pet_friendly"
            }
        }
    }
}
