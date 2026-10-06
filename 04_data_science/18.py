import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import os

# Load the geospatial data
base_dir = os.path.dirname(__file__)
bus_stops = gpd.read_file(os.path.join(base_dir, '..', 'data', 'bus_stops.shp')) # Load bus stop locations
ridership = pd.read_csv(os.path.join(base_dir, '..', 'data', 'ridership.csv')) # Load ridership data

# Convert the ridership data to a GeoDataFrame
ridership_gdf = gpd.GeoDataFrame(
    ridership, geometry=gpd.points_from_xy(ridership.longitude, ridership.latitude)
)

# Merge the bus stop locations with the ridership data
bus_stops_with_ridership = bus_stops.merge(ridership, on='stop_id')

# Load the city map
city_map = gpd.read_file(os.path.join(base_dir, '..', 'data', 'city_map.shp'))

# Plot the bus stops with ridership data
fig, ax = plt.subplots(figsize=(10, 10))
city_map.plot(ax=ax, color='lightgrey')  # Assuming you have a city map GeoDataFrame
bus_stops_with_ridership.plot(ax=ax, column='ridership', cmap='OrRd', legend=True)
ridership_gdf.plot(ax=ax, color='blue', markersize=5, label='Ridership Points')
plt.title('Bus Stops with Ridership Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

