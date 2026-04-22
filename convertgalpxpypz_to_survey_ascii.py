import numpy as np
import pandas as pd
from scipy.io import FortranFile
import glob
import os

##----------------------------------------------------------------
# This program is a small program I use to extract galaxy snapshot data in fortran format into 
# a galaxies.dat text file in survey ascii format in preparation for use with disperse.
# Example used is snapshot 970.
# (Documentation for survey ascii:
# https://www.iap.fr/useriap/sousbie/web/html/index744c.html?post/survey_ascii-format)
# This file must be ran through delaunay_3D so that its results can be ran successively
# through netconv, mse, and then skelconv. Further documentation found in PlotGalFilNDsklAscii.py.
# There might already be a version of this in FortranFile written by Janvi but I have to check.
# Must make sure there is no hashtag at top of file or delaunay gets confused. Should only have
# px py pz id as header. px py pz are position coordinates of the galaxies, id is galaxy id
# galaxy id is useful to later compare how the galaxy positions evolve over time through snapshots.
##----------------------------------------------------------------

# ── CONFIGURE THESE ──────────────────────────────────────────────
GAL_DIR = "C:/Users/ronal/Documents/FilamentSFH/New Horizon Data/Treebricks/Stars/Galaxy970StarData/GAL_00970"
OUTPUT  = "galaxies.dat"
# ─────────────────────────────────────────────────────────────────

gal_files = sorted(glob.glob(os.path.join(GAL_DIR, "gal_stars_*")))
print(f"Found {len(gal_files)} GAL files")

positions = []

for filepath in gal_files:
    try:
        f = FortranFile(filepath)
        my_number = np.squeeze(f.read_record("i"))
        level     = np.squeeze(f.read_record("i"))
        m         = np.squeeze(f.read_record("d"))
        px, py, pz = np.squeeze(f.read_record("d", "d", "d"))
        f.close()

        positions.append((px, py, pz, my_number))
    except Exception as e:
        print(f"  Skipping {filepath}: {e}")

print(f"Successfully read {len(positions)} galaxies")

# Write survey_ascii format — no # on header, no unit conversion
with open(OUTPUT, "w") as out:
    out.write("px py pz id\n")
    for (px, py, pz, gal_id) in positions:
        out.write(f"{px:+.6e}\t{py:+.6e}\t{pz:+.6e}\t{int(gal_id)}\n")

print(f"Written to {OUTPUT}")
print(f"Preview of first 3 lines:")
with open(OUTPUT) as f:
    for i, line in enumerate(f):
        if i > 3: break
        print(" ", line.strip())