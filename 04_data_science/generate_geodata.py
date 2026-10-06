import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import os

# Set paths
base_dir = "/run/media/shubh/New Volume/ai-journey/04_data_science"
city_map_path = os.path.join(base_dir, 'city_map.shp')
bus_stops_path = os.path.join(base_dir, 'bus_stops.shp')
ridership_path = os.path.join(base_dir, 'ridership.csv')

print("Generating mock geospatial data...")

# 1. Create city_map.shp
# A simple square polygon representing the city
city_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
city_map = gpd.GeoDataFrame({'geometry': [city_poly]}, crs="EPSG:4326")
city_map.to_file(city_map_path)
print(f"Saved {city_map_path}")

# 2. Create bus_stops.shp
# 5 random points inside the city
stop_ids = [1, 2, 3, 4, 5]
# Fixed coordinates for reproducibility
bus_coords = [(2, 2), (3, 8), (7, 4), (8, 9), (5, 5)]
bus_points = [Point(x, y) for x, y in bus_coords]
bus_stops = gpd.GeoDataFrame({'stop_id': stop_ids, 'geometry': bus_points}, crs="EPSG:4326")
bus_stops.to_file(bus_stops_path)
print(f"Saved {bus_stops_path}")

# 3. Create ridership.csv
np.random.seed(42)
ridership_data = {
    'stop_id': stop_ids,
    'ridership': np.random.randint(50, 500, size=5),
    'longitude': [x for x, y in bus_coords],
    'latitude': [y for x, y in bus_coords]
}
ridership_df = pd.DataFrame(ridership_data)
ridership_df.to_csv(ridership_path, index=False)
print(f"Saved {ridership_path}")

print("Data generation complete!")

