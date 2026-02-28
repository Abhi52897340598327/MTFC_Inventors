import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'maps')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Approximate Virginia boundary polygon (lon, lat)
VA_POLY = np.array([
    (-83.675, 36.60), (-82.95, 36.60), (-82.30, 36.62), (-81.65, 36.64),
    (-80.95, 36.63), (-80.30, 36.75), (-79.60, 36.90), (-78.90, 37.02),
    (-78.30, 37.15), (-77.70, 37.35), (-77.20, 37.55), (-76.80, 37.75),
    (-76.40, 37.95), (-76.10, 38.15), (-75.85, 38.35), (-75.70, 38.55),
    (-75.45, 38.45), (-75.30, 38.00), (-75.20, 37.60), (-75.28, 37.20),
    (-75.40, 36.95), (-75.60, 36.85), (-76.05, 36.86), (-76.50, 36.85),
    (-77.00, 36.84), (-77.50, 36.80), (-78.10, 36.76), (-78.70, 36.73),
    (-79.30, 36.69), (-80.00, 36.66), (-80.70, 36.64), (-81.40, 36.62),
    (-82.10, 36.60), (-82.80, 36.59), (-83.30, 36.58), (-83.675, 36.60)
])

# Project-informed + common VA datacenter hubs
# weight indicates relative concentration intensity (not official census count)
HUBS = [
    # Northern Virginia / Loudoun corridor
    ('Ashburn',        39.0438, -77.4874, 1.00),
    ('Sterling',       39.0062, -77.4286, 0.85),
    ('Reston',         38.9586, -77.3570, 0.70),
    ('Manassas',       38.7509, -77.4753, 0.55),
    ('Chantilly',      38.8943, -77.4311, 0.45),
    ('Culpeper',       38.4735, -77.9967, 0.35),
    ('Winchester',     39.1857, -78.1633, 0.18),
    ('Fredericksburg', 38.3032, -77.4605, 0.16),
    # Central Virginia
    ('Richmond',       37.5407, -77.4360, 0.30),
    ('Henrico',        37.5059, -77.3328, 0.28),
    ('Petersburg',     37.2279, -77.4019, 0.14),
    ('Charlottesville',38.0293, -78.4767, 0.15),
    ('Lynchburg',      37.4138, -79.1422, 0.13),
    # Shenandoah / I-81 corridor
    ('Harrisonburg',   38.4496, -78.8689, 0.13),
    ('Staunton',       38.1496, -79.0717, 0.12),
    ('Roanoke',        37.2709, -79.9414, 0.12),
    ('Salem',          37.2935, -80.0548, 0.10),
    ('Blacksburg',     37.2296, -80.4139, 0.10),
    # Hampton Roads / Coastal
    ('Virginia Beach', 36.8529, -75.9780, 0.25),
    ('Norfolk',        36.8508, -76.2859, 0.20),
    ('Newport News',   37.0871, -76.4730, 0.18),
    ('Chesapeake',     36.7682, -76.2875, 0.17),
    ('Suffolk',        36.7282, -76.5836, 0.14),
    # Southside / Southwest
    ('Danville',       36.5859, -79.3950, 0.10),
    ('Martinsville',   36.6915, -79.8725, 0.09),
    ('Bristol',        36.5951, -82.1887, 0.08),
]


def make_heatmap():
    lats = np.array([h[1] for h in HUBS])
    lons = np.array([h[2] for h in HUBS])
    weights = np.array([h[3] for h in HUBS])

    lon_min, lon_max = -84.1, -75.0
    lat_min, lat_max = 36.3, 39.6
    nx, ny = 500, 350
    lon_grid = np.linspace(lon_min, lon_max, nx)
    lat_grid = np.linspace(lat_min, lat_max, ny)
    X, Y = np.meshgrid(lon_grid, lat_grid)

    # Broader smoothing to represent statewide footprint, not only point hotspots.
    sx, sy = 0.38, 0.28
    Z = np.zeros_like(X)
    for lat, lon, w in zip(lats, lons, weights):
        Z += w * np.exp(-(((X - lon) ** 2) / (2 * sx ** 2) + ((Y - lat) ** 2) / (2 * sy ** 2)))

    poly_path = Path(VA_POLY)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    inside = poly_path.contains_points(pts).reshape(X.shape)
    # Add a low-intensity statewide background so the full state is represented.
    regional_anchors = [
        (38.95, -77.45, 0.10),  # NoVA
        (37.55, -77.45, 0.08),  # Richmond metro
        (37.10, -76.20, 0.08),  # Hampton Roads
        (37.35, -79.80, 0.06),  # Valley / Roanoke
        (36.75, -80.20, 0.05),  # Southside / Southwest
    ]
    bg_sx, bg_sy = 0.85, 0.62
    for lat, lon, w in regional_anchors:
        Z += w * np.exp(-(((X - lon) ** 2) / (2 * bg_sx ** 2) + ((Y - lat) ** 2) / (2 * bg_sy ** 2)))

    Z = np.where(inside, Z, np.nan)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.fill(VA_POLY[:, 0], VA_POLY[:, 1], facecolor='#f5f7fa', edgecolor='#2d3e50', linewidth=1.8, zorder=1)

    im = ax.imshow(
        Z,
        origin='lower',
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap='YlOrRd',
        alpha=0.88,
        aspect='auto',
        zorder=2,
    )

    ax.scatter(lons, lats, s=22 + 110 * weights, c='#1f77b4', edgecolors='white', linewidths=0.8, zorder=3)
    for name, lat, lon, w in HUBS:
        if w >= 0.24 or name in {'Virginia Beach', 'Norfolk', 'Roanoke', 'Charlottesville'}:
            ax.text(lon + 0.06, lat + 0.03, name, fontsize=8, color='#1d3557', zorder=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Relative Datacenter Density (weighted)')

    ax.set_title('Virginia Datacenter Location Heatmap\n(Project-anchored Ashburn focus + common Virginia hub distribution)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(alpha=0.18)

    note = (
        'Note: Relative weighted density from project context + common VA hub distribution,\n'
        'with statewide background anchors to visualize broader in-state footprint.'
    )
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8, color='#444')

    out = os.path.join(OUTPUT_DIR, 'virginia_datacenter_location_heatmap.png')
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    make_heatmap()
