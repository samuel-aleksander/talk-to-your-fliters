# Talk to Your Filters: Natural-Language Faceted Search for Airbnb

## Summary

This **implementation project** consists of a faceted search interface that translates natural-language descriptions into structured, metadata-based filters. The goal of the prototype is to demonstrate how human-defined metadata schemas can work together with large language models (LLMs) to make information retrieval more flexible while remaining transparent and controllable at the same time.

Using Airbnb listings as a case study, the system addresses a common problem in browsing large collections. Users often describe what they want in everyday language instead of through predefined filter menus. At the same time, relying solely on natural-language retrieval can obscure which attributes actually influence the results. This project combines both approaches by treating natural language input as an alternative way to set facets rather than as a separate or unique search mechanism.

When a user enters a natural-language description (e.g., “a cheap two-bedroom apartment in Los Angeles”), a large language model extracts only the facets explicitly mentioned and maps them to a controlled vocabulary defined by the dataset’s schema. These inferred facets are then applied as deterministic filters to ensure that all returned listings conform to the selected criteria. 

The interface presents the inferred facets within the same filtering system used for manual browsing, which allows users to review, modify, or remove them at any time. Natural language input does not replace faceted navigation, but rather, it acts as a complementary way to set filters.

## Getting Started

### Option 1: Use the live app

This project is designed to be deployed on Streamlit Cloud so anyone can use it in the browser.

Access the deployed version here:  
[https://talk-to-your-fliters.streamlit.app/](https://talk-to-your-fliters.streamlit.app/)

### Option 2: Run locally

1. **Clone the repository.**
2. **Install dependencies.** `pip install -r requirements.txt`
3. **Add API credentials.** This project uses the Anthropic API to translate natural-language queries into filters.
    
    You will need to provide your own Anthropic API key. 
    
    Create a file at:
    
    ```jsx
    .streamlit/secrets.toml
    ```
    
    and add:
    
    ```jsx
    ANTHROPIC_API_KEY = "your_api_key_here"
    ```
    
4. **Run the app**. `streamlit run app.py`

## Functionality, Design Rationale and Concepts from Class

### Faceted search

The core interaction model of the application is based on faceted search. As we saw in class, faceted search allows users to explore a large collection by progressively narrowing results along multiple dimensions.

The facet schema can be found in the file `facet_schema.py` . Within the UI, independent facets are grouped according to their organizing principles to reduce cognitive workload. For instance, `property type` and `room type` , even though they are independent facets, are grouped together under the title “Listing Type”. 

The schema contains quantitative facets (e.g., `minimum_guests`, `minimum_bedrooms`), categorical facets (e.g., `property_type` , `room_type` ), and hierarchical facets (e.g., `location`, progressing from country to state, city, and neighborhood). 

The facet `location` provide an interesting case on how the nature of the facet can (and should) shape the behavior of the app. Unlike independent facets, these levels are related by containment: a city belongs to a specific state and country, and a neighborhood belongs to a specific city. 

To preserve this hierarchical consistency and prevent invalid filter combinations, the interface automatically sets and locks higher-level location filters when a lower-level location is selected. In these cases, parent filters are displayed as “auto-filtered” and disabled to indicate that their values were inferred by the system rather than explicitly chosen by the user.

Another interesting case is the `amenities` facet. While the interface visually groups amenities into categories such as “Basics” or “Laundry” to improve scannability, these categories are not treated as hierarchical facets by the system. Filtering operates only on individual amenity values (for example, “Wi-Fi”), and users cannot select amenity categories themselves. As a result, the grouping exists purely at the presentation level and does not introduce semantic hierarchy or affect filtering behavior.

### Large Language Model Integration

This project uses the Anthropic API to convert a user’s natural language description into structured facet filters. The LLM is not used to retrieve or rank listings directly. Instead, it functions as a constrained extraction step that outputs a JSON object matching the application’s facet schema.

The prompt provided to the LLM model comprises a set of current allowed values for each facet (countries, states, cities, property types, room types, and amenities) and is instructed to return only values that appear in those lists. The purpose of this controlled-vocabulary approach was to reduce hallucination risk and to ensure that all LLM-produced filters can be applied deterministically to the dataset. 

After receiving the model response, the app parses the JSON and updates the application to apply the inferred filters to the same UI controls used for manual faceting. This keeps the system transparent, which was one of the main objectives of the project. Users can then review, edit, or remove inferred facet values at any time. 

## Technical Specifications

### Data Source

The starting dataset `airbnb-listing.csv` comes from Kaggle: [Airbnb dataset (joebeachcapital)](https://www.kaggle.com/datasets/joebeachcapital/airbnb).

This repo uses a processed subset exported as `talk-to-your-fliters/airbnb_final.parquet`, which is what the app loads.

### Data Processing & Feature Engineering

All data preprocessing is performed offline using Python and pandas. The original CSV dataset from Kaggle included 89 features and was approximately 4.88 GB in size. This scale provided flexibility for different use cases, but it also introduced complexity in terms of normalization, performance, and file handling that went beyond the scope of this project. Additionally, the limitations of the free Streamlit Community Cloud required a significantly smaller dataset. As a result, we performed preprocessing to reduce the dataset to a smaller, more manageable size and to focus on a controlled set of features relevant to user search.

The preprocessing script is located at `talk-to-your-filters/preprocessing.py`. At a high level, it performs the following steps:

- **Retains only columns that are most relevant** to the use case and commonly used by users when searching.
- **Drops incomplete rows** by removing entries with missing or zero values in `Country`, `City`, `State`, `Number of Reviews`, `Review Scores Rating` and `Price`.
- **Restricts the dataset to a curated city set**:
    - Los Angeles, San Francisco, New York, Boston, Chicago, Denver, Nashville, New Orleans, Austin.
- **Creates boolean amenity facet columns** by parsing the raw `Amenities` text field:
    - `amenity_wifi`, `amenity_kitchen`, `amenity_washer`, `amenity_dryer`, `amenity_TV`, `amenity_free_parking`,
    `amenity_air_conditioning`, `amenity_heating`, `amenity_pool`, `amenity_hot_tub`, `amenity_pet_friendly`.
- **Creates a 0–5 rating facet**: `rating_stars = (Review Scores Rating) / 20`.
- **Normalizes property types** into a controlled vocabulary to reduce complexity and filter out rare types:
    - Keeps common types (`Apartment`, `House`, `Condominium`, `Loft`, `Townhouse`, `Guesthouse`, `Bed & Breakfast`, `Bungalow`);
    - All other values are grouped under `Unique stay` .
- **Cleans up geography**:
    - Keeps only states with at least **5 listings**.
    - Removes neighborhood values that match a ZIP code pattern (strings consisting of five digits).
    - Restricts the dataset to listings in the United States.
- **Drops unused columns** from the raw dataset (e.g. `Amenities`, `Features`, `Country Code`, `Review Scores Rating`).
- **Exports** the cleaned dataset to `airbnb_final.parquet` to reduce file size and improve performance of Streamlit application.

The resulting Parquet file contains 52,177 listings and is 1.89 MB in size. 

### Web Application

The interactive interface is built using `Streamlit`. The application loads the preprocessed Parquet file and provides users with:

- Faceted filters for city, state, property type, price range, ratings, and amenities
- Dynamic filtering based on natural language queries

`Streamlit` was selected because it enables rapid development of data-driven interfaces, integrates cleanly with pandas DataFrames, and provides a simple free hosting solution through Streamlit Community Cloud.

### Large Language Model Setup

This project integrates with the Anthropic API to interpret natural-language queries and translate them into structured facet filters. Although LLM support is a core feature, the application can run without it. If the Anthropic API key is not configured, users can still browse listings using the sidebar facets and the rest of the interface functions normally.

When a user clicks “Interpret Query”, the app sends the query (along with the current allowed facet values) to Anthropic’s Messages API and requests a JSON-only response that matches the app’s facet schema. The returned facet values are then parsed and applied deterministically to the sidebar filters via Streamlit session state, allowing users to review and adjust the inferred filters just like manually selected ones. The project uses Claude 3.5 Haiku to reduce latency and cost.

### Large Language Model Query Logs

For development and debugging purposes, the application includes a logging function in `app.py` that records each request made to the LLM. For every query, the function appends a new row to a local CSV file containing the user’s input, a timestamp, and the structured JSON output returned by the model.

These logs are used exclusively during development to provide visibility into how the LLM interprets user queries and how those interpretations are applied by the application. This makes it easier to inspect model behavior, diagnose parsing issues, and evaluate how natural-language descriptions are translated into facet filters. The log file is generated locally and is not required for running the application.

## GenAI Use Disclosure

This project was developed with the support of several generative AI tools. In particular, Cursor was used as a coding assistant during the development of the Streamlit application. Although the overall structure and core logic of the app were implemented manually, Cursor was used to generate and refine specific parts of `app.py`. The data preprocessing pipeline in `preprocessing.py` was written without AI assistance.

More precisely, the functionality of Cursor agent was used to help integrate the app with the Anthropic API. This includes constructing the API calls, handling model responses, and parsing the structured JSON output returned by the LLM. It also assisted with implementing interaction logic in the interface. Mainly, to enforce hierarchical constraints in the location facet by automatically setting and disabling parent filters when a child location is selected. In addition, ChatGPT (models 5.1 and 5.2) was used to help draft and refine the prompt used for facet extraction.

One challenge encountered when using generative AI tools for code generation was the tendency to become overly dependent on them as development progressed. While AI-generated code accelerated implementation, it required a significant investment of time to carefully review, understand, and integrate that code into the broader system. 

Another challenge involved transparency and “debuggability”. Some of the code generated by Cursor, particularly around API integration, lacked sufficient visibility into errors or intermediate states. To address this, a custom logging workflow was added to record LLM inputs and outputs locally. This allowed inspection and debugging of how natural-language queries were interpreted and applied.

Finally, this written report was proofread for grammar and clarity with the assistance of ChatGPT (models 5.1 and 5.2).