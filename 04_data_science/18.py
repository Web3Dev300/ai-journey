import os

import folium
import geopandas as gpd
import pandas as pd

# Load the geospatial data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
bus_stops = gpd.read_file(os.path.join(data_dir, 'bus_stops.shp'))  # Bus stop locations (points)
city_map = gpd.read_file(os.path.join(data_dir, 'city_map.shp'))    # City boundary (polygon)
ridership = pd.read_csv(os.path.join(data_dir, 'ridership.csv'))    # Riders per stop

# Attach the ridership numbers to each bus stop
stops = bus_stops.merge(ridership[['stop_id', 'ridership']], on='stop_id')

# Spatial join: keep only the stops that fall inside the city boundary
stops = gpd.sjoin(stops, city_map[['geometry']], how='inner', predicate='within').drop(columns='index_right')
print(f"{len(stops)} bus stops inside the city boundary")
print(stops[['stop_id', 'ridership']].sort_values('ridership', ascending=False).to_string(index=False))

# Colour each stop by how busy it is compared with the other stops
low, high = stops['ridership'].quantile([0.33, 0.66])


def get_color(riders):
    if riders >= high:
        return 'red'     # Busy: candidate for more frequent service
    if riders >= low:
        return 'orange'
    return 'green'       # Quiet


# Build an interactive map that zooms to the stops
min_lon, min_lat, max_lon, max_lat = stops.total_bounds
bus_map = folium.Map()
bus_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

# Iterate through data and add circles to the map
for idx, row in stops.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5 + 20 * row['ridership'] / stops['ridership'].max(),  # Size represents ridership
        popup=f"Stop: {row['stop_id']}<br>Ridership: {row['ridership']}",
        color=get_color(row['ridership']),  # Colour shows how busy the stop is
        fill=True,
        fill_opacity=0.7
    ).add_to(bus_map)

output_path = os.path.join(base_dir, 'bus_ridership_map.html')
bus_map.save(output_path)
print(f"Map saved to {output_path} - open it in a browser")
