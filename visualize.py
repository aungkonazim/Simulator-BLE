import folium
import pickle
import pandas as pd
data = pickle.load(open('./data/data.p','rb'))

def generateBaseMap(default_location=[35.14953, -90.04898], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map
generateBaseMap()