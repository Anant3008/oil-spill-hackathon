import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def create_animation(cloud, start_lat, start_lon, filename="output/particle_animation.gif"):
    """
    Creates a Matplotlib animation of the particle cloud over time.
    
    Args:
        cloud: The ParticleCloud object containing position history.
        start_lat (float): The origin latitude.
        start_lon (float): The origin longitude.
        filename (str): The output file path for the animated GIF.
    """
    print(f"\nGenerating animation: {filename}...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate dynamic bounds based on full history to keep the camera fixed
    all_lats = np.concatenate(cloud.history_lats)
    all_lons = np.concatenate(cloud.history_lons)
    
    lat_min, lat_max = np.min(all_lats), np.max(all_lats)
    lon_min, lon_max = np.min(all_lons), np.max(all_lons)
    
    # Add a 15% padding around the maximum bounds of the simulation
    pad_lat = max(0.01, (lat_max - lat_min) * 0.15)
    pad_lon = max(0.01, (lon_max - lon_min) * 0.15)
    
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Plot the spill origin
    ax.plot(start_lon, start_lat, 'rX', markersize=10, label="Spill Origin")
    
    # Initialize scatter plot for the particles
    scatter = ax.scatter(cloud.history_lons[0], cloud.history_lats[0], 
                         s=8, c='black', alpha=0.6, edgecolors='none', label="Oil Particles")
    
    title = ax.set_title("Time: T+0 hours")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper right")
    
    def update(frame):
        # Update particle coordinates
        # offsets must be an Nx2 array of (x, y) coordinates -> (lon, lat)
        data = np.c_[cloud.history_lons[frame], cloud.history_lats[frame]]
        scatter.set_offsets(data)
        title.set_text(f"Time: T+{frame} hours")
        return scatter, title
        
    anim = animation.FuncAnimation(
        fig, update, frames=len(cloud.history_lats), 
        interval=250, blit=True, repeat=True
    )
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Use pillow to save the GIF without needing ffmpeg installed
    anim.save(filename, writer='pillow', fps=4)
    print(f"Animation successfully saved to {filename}\n")
    plt.close(fig)

