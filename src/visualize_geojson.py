import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import os

def animate_geojson(geojson_path):
    if not os.path.exists(geojson_path):
        print(f"File not found: {geojson_path}")
        return

    print(f"Loading {geojson_path} for visualization...")
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    features = data.get('features', [])
    if not features:
        print("No features found in GeoJSON.")
        return

    # Find global bounds for the plot
    all_lons = []
    all_lats = []
    for feat in features:
        coords = feat['geometry']['coordinates'][0] # Outer ring of polygon
        for lon, lat in coords:
            all_lons.append(lon)
            all_lats.append(lat)

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)

    pad_lon = (max_lon - min_lon) * 0.1 or 0.1
    pad_lat = (max_lat - min_lat) * 0.1 or 0.1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
    ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.5)

    is_backward = 'origin' in geojson_path or 'hindcast' in geojson_path
    direction = "BACKWARD HINDCAST (Origin Probability)" if is_backward else "FORWARD PREDICTION (Spill Trajectory)"
    color = 'crimson' if is_backward else 'dodgerblue'

    # Initialize an empty polygon
    poly_patch = Polygon([[0,0]], closed=True, facecolor=color, alpha=0.5, edgecolor='black', linewidth=1.5)
    ax.add_patch(poly_patch)

    # Text displays
    title_text = ax.set_title('', fontsize=12, fontweight='bold')
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame):
        feat = features[frame]
        coords = feat['geometry']['coordinates'][0]
        time_str = feat['properties']['time']

        poly_patch.set_xy(coords)
        title_text.set_text(direction)

        info = f"Time: {time_str}\n"
        if "is_origin_probability_region" in feat['properties']:
            info += "\n!!! HIGHEST PROBABILITY ORIGIN !!!"
            poly_patch.set_facecolor('gold')
            poly_patch.set_alpha(0.8)
        else:
            poly_patch.set_facecolor(color)
            poly_patch.set_alpha(0.5)

        info_text.set_text(info)
        return poly_patch, title_text, info_text

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(features), interval=300, blit=True, repeat_delay=2000)

    # Display the interactive window
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        animate_geojson(sys.argv[1])
    else:
        print("--- Testing Visualizer ---")
        print("Close the first window to see the second one!")
        animate_geojson("output/predicted_regions.geojson")
        animate_geojson("output/origin_probability_regions.geojson")
