import pandas as pd

df = pd.read_csv("airbnb_small.csv", low_memory=False)
# print(df["Country"].unique())

cols = ["Country", "City", "State", "Number of Reviews", "Review Scores Rating", "Price"]

# Remove NaN
df = df.dropna(subset=cols)

# Remove rows where ANY of these columns equals 0 or "0"
for col in cols:
    df = df[df[col] != 0]
    df = df[df[col] != "0"]

# Define the clean cities you want to keep
clean_cities = [
    "Los Angeles",
    "San Francisco",
    "New York",
    "Boston",
    "Chicago",
    "Denver",
    "Nashville",
    "New Orleans",
    "Austin"
]

# Keep only rows with city exactly matching these
df_clean = df[df["City"].isin(clean_cities)].copy()

# Create boolean columns for each amenity 
df_clean["amenity_wifi"] = df_clean["Amenities"].str.contains("Wireless Internet|Internet", case=False, na=False)
df_clean["amenity_kitchen"] = df_clean["Amenities"].str.contains("Kitchen", case=False, na=False)
df_clean["amenity_washer"] = df_clean["Amenities"].str.contains("Washer", case=False, na=False)
df_clean["amenity_dryer"] = df_clean["Amenities"].str.contains("Dryer", case=False, na=False)
df_clean["amenity_TV"] = df_clean["Amenities"].str.contains("TV", case=False, na=False)
df_clean["amenity_free_parking"] = df_clean["Amenities"].str.contains("Free parking", case=False, na=False)
df_clean["amenity_air_conditioning"] = df_clean["Amenities"].str.contains("Air conditioning", case=False, na=False)
df_clean["amenity_heating"] = df_clean["Amenities"].str.contains("Heating", case=False, na=False)
df_clean["amenity_pool"] = df_clean["Amenities"].str.contains("Pool", case=False, na=False)
df_clean["amenity_hot_tub"] = df_clean["Amenities"].str.contains("Hot tub", case=False, na=False)
df_clean["amenity_pet_friendly"] = df_clean["Amenities"].str.contains("Pets allowed", case=False, na=False)

# convert review scores rating to a score out of 5
df_clean["rating_stars"] = df_clean["Review Scores Rating"] / 20

# Drop the columns that are not needed
df_clean.drop(columns=["Amenities"], inplace=True) # Drop the Amenities column
df_clean.drop(columns=["Features"], inplace=True) # Drop the Amenities column
df_clean.drop(columns=["Country Code"], inplace=True) # Drop the Amenities column
df_clean.drop(columns=["Review Scores Rating"], inplace=True)

# Rename column
df_clean.rename(columns={"Neighbourhood Cleansed": "Neighbourhood"}, inplace=True)


MAIN_TYPES = [
    "Apartment", "House", "Condominium", "Loft",
    "Townhouse", "Guesthouse", "Bed & Breakfast", "Bungalow"
]

# Normalize the Property Type column to include only the main types
df_clean["Property Type Normalized"] = df_clean["Property Type"]
mask_rare = ~df_clean["Property Type"].isin(MAIN_TYPES)
df_clean.loc[mask_rare, "Property Type Normalized"] = "Unique stay"


print(list(df_clean.columns))
print(df_clean["Property Type Normalized"].value_counts())
# print(df_clean["Features"].value_counts())

#print(df_clean["amenity_free_parking"].value_counts())
# Save the cleaned dataframe to a new CSV file
# df_clean.to_csv("airbnb_cleaned_USA.csv", index=False)

df_clean.to_parquet("airbnb_final.parquet")
