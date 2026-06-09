import sys
import os
_PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

import geopandas as gpd
import shapely.geometry as sg
import matplotlib.pyplot as plt

from utils.matplotlib_chinese import setup_matplotlib_chinese
setup_matplotlib_chinese()

trace_data_url = "thkceshi-landmark1.geojson"
traces = gpd.read_file(trace_data_url)

area_data_url = "my_area.geojson"
area = gpd.read_file(area_data_url)

# Step 2: Check the CRS (coordinate reference system)
if traces.crs != area.crs:
    raise ValueError("Traces and area GeoDataFrames should have the same CRS.")

# Step 3: Check if geometries are valid
if not traces.is_valid.all() or not area.is_valid.all():
    raise ValueError("Invalid geometries found in trace or area GeoDataFrames.")

# Step 4: Check for missing values or conflicting data types
if traces.isna().sum().sum() > 0:
    raise ValueError("NaN values found in trace GeoDataFrame.")
if not traces.geometry.dtypes == sg.LineString:
    raise ValueError("Trace GeoDataFrame should only contain LineString geometries.")
if area.isna().sum().sum() > 0:
    raise ValueError("NaN values found in area GeoDataFrame.")
if not area.geometry.dtypes == sg.Polygon:
    raise ValueError("Area GeoDataFrame should only contain Polygon geometries.")

# Step 5: Check snap threshold value
snap_threshold = 0.001
if snap_threshold > traces.geometry.length.min():
    raise ValueError("Snap threshold value too large for this trace GeoDataFrame.")

# Assign a value to 'truncate_traces'
truncate_traces = True
geom_types = traces.geom_type.unique()

# Check if Trace GeoDataFrame only contains LineString geometry
if "LineString" not in geom_types:
    raise ValueError("Trace GeoDataFrame should only contain LineString geometries.")
if len(geom_types) > 1:
    raise ValueError("Trace GeoDataFrame contains multiple geometry types.")

traces.plot()
plt.show()
