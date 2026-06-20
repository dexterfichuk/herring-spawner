# Research Findings: Strait of Georgia Data Centre (SOGDC) Herring Spawn Data

I investigated the provided interactive map and successfully identified the underlying ArcGIS REST service used by the Pacific Salmon Foundation (PSF) for their "Dashboard Map of Herring Spawn in the Strait of Georgia, 1951 to 2021".

## Data Found

The interactive dashboard (Item ID: `c1812e5d016e47fbbab0deff33c50ea1`) utilizes a Web Map (`c8c4d56da9814eb2ae3300b5408a7bc9`) containing the following primary feature layers:

1.  **Pacific Herring Spawn Index (Points)**
    *   **Layer Name:** `SpawnIndexSOG_decades_altered_yeartxt`
    *   **Description:** A point dataset representing spawn index observations from 1951 to 2021.
    *   **Records Found:** 5,606
    *   **Available Attributes:** Year, Longitude, Latitude, Start Date, End Date, Region, StatArea, Section, CombinedSI (Spawn Index).
    *   **Coordinates:** Available as both `Longitude`/`Latitude` attributes and as GeoJSON point geometry.
    *   **Dates:** Available in multiple formats including `Year`, `Start`, and `gis_date`.

2.  **Digitized Herring Spawn Polygons**
    *   **Layer Name:** `Digitized_Herring_Spawn_Polygons_for_SOG_wDecades_yrtxt`
    *   **Description:** High-resolution spatial polygons of historical spawn events.
    *   **Records Found:** 49
    *   **Coordinates:** Available as complex GeoJSON polygon geometry.

## Downloaded Files

The data was queried from the ArcGIS REST API and saved to `/Users/dexterfichuk/Downloads/sog_data/` in GeoJSON format:

- `spawn_index_part1.geojson`: First 2,000 records of Spawn Index data.
- `spawn_index_part2.geojson`: Records 2,001 to 4,000 of Spawn Index data.
- `spawn_index_part3.geojson`: Remaining 1,606 records of Spawn Index data.
- `spawn_polygons.geojson`: Full set of 49 digitized spawn polygons.

## REST Endpoints

The raw data can be accessed directly via these ArcGIS FeatureServer URLs:

- **Spawn Index Service:** [https://services7.arcgis.com/yHbO69mL1QTGCPQG/arcgis/rest/services/SpawnIndexSOG_decades_altered_yeartxt/FeatureServer/0](https://services7.arcgis.com/yHbO69mL1QTGCPQG/arcgis/rest/services/SpawnIndexSOG_decades_altered_yeartxt/FeatureServer/0)
- **Spawn Polygons Service:** [https://services7.arcgis.com/yHbO69mL1QTGCPQG/arcgis/rest/services/Digitized_Herring_Spawn_Polygons_for_SOG_wDecades_yrtxt/FeatureServer/0](https://services7.arcgis.com/yHbO69mL1QTGCPQG/arcgis/rest/services/Digitized_Herring_Spawn_Polygons_for_SOG_wDecades_yrtxt/FeatureServer/0)

## Data Quality Notes
The "Spawn Index" data has been "altered" or enhanced by PSF GIS specialists for better visualization in the dashboard (e.g., adding `decade` and `year_txt` fields). The `gis_date` field appears to be a calculated date for temporal mapping but should be cross-referenced with the `Start` attribute for precision.
