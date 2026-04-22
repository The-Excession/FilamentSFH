import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##----------------------------------------------------------------
#This program plots the filaments created from galaxies over the galaxies themselves.
#The filaments were created by running convertgalpxpypz_to_survey_ascii.py
#(Documentation for survey ascii:
# https://www.iap.fr/useriap/sousbie/web/html/index744c.html?post/survey_ascii-format)
#and then running that .dat file through a submit script to infinity.
#use .dat file as input into delaunay_3D
#then run successive outputs through netconv, mse, skelconv for skeleton file
#script can be found in /data70/cuny/rona/galpositionsurveyascii/galfil.sh
#final format documentation: 
# https://www.iap.fr/useriap/sousbie/web/html/indexbea5.html?post/NDskl_ascii-format
##----------------------------------------------------------------

# ── CONFIGURE THESE ──────────────────────────────────────────────
#This is the skeleton file output from skelconv:
SKL_FILE = "C:/Users/ronal/Documents/FilamentSFH/New Horizon Data/ProcessedFilSkel/GalaxyNDsklAscii/galaxies.dat_nsig3_final.S002.BRK.a.NDskl"
#This is the original galaxies text file that the filaments were made from:
GAL_FILE = "C:/Users/ronal/Documents/FilamentSFH/GalaxyCentering/galaxies.dat"
# NOTE: if plot looks mirrored swap px and py due to galaxy px and py coord swap
# ─────────────────────────────────────────────────────────────────

def read_ndskl_ascii(filename):
    filaments = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    # Skip to FILAMENTS section
    while i < len(lines):
        if lines[i].strip() == '[FILAMENTS]':
            i += 1
            break
        i += 1
    
    if i >= len(lines):
        print("ERROR: [FILAMENTS] section not found!")
        return filaments
    
    nfil = int(lines[i].strip())
    i += 1
    print(f"Number of filaments: {nfil}")
    
    for _ in range(nfil):
        header = lines[i].strip().split()
        nsamp = int(header[2])
        i += 1
        points = []
        for _ in range(nsamp):
            coords = list(map(float, lines[i].strip().split()))
            points.append(coords)
            i += 1
        filaments.append(np.array(points))
    
    return filaments

def read_galaxies(filename):
    # Read galaxy positions from survey_ascii file
    # Skips header line, loads px py pz
    data = np.genfromtxt(filename, skip_header=1, usecols=(0, 1, 2))
    return data

# Read data
filaments = read_ndskl_ascii(SKL_FILE)
galaxies = read_galaxies(GAL_FILE)
print(f"Number of galaxies: {len(galaxies)}")

# ── 3D PLOT ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot galaxies as scatter points
ax.scatter(galaxies[:,1],   # ← swap 0 and 1 if x/y need switching
           galaxies[:,0],   # ← swap 0 and 1 if x/y need switching
           galaxies[:,2],
           s=2,
           color='orange',
           alpha=0.5,
           label='Galaxies')

# Plot filaments as lines
for fil in filaments:
    ax.plot(fil[:,1],   # ← swap 0 and 1 if x/y need switching
            fil[:,0],   # ← swap 0 and 1 if x/y need switching
            fil[:,2],
            color='steelblue',
            linewidth=0.5,
            alpha=0.7)

ax.set_xlabel('px')
ax.set_ylabel('py')  
ax.set_zlabel('pz')
ax.set_title('NewHorizon Galaxy Filaments (snapshot 970)')
ax.legend()

plt.tight_layout()
plt.savefig('filaments_and_galaxies_3d.png', dpi=150)
plt.show()
print("Plot saved to filaments_and_galaxies_3d.png")