import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from LuzFDTD.FDTD2D_jax import FDTD_2D


# define simulation size and step size
simulation_sizex = 7e-6
simulation_sizey = 7e-6
step_size = 40e-9 # dy

# wavelength information
center_wavelength = 1550e-9
wavelengths = np.linspace(center_wavelength - 100e-9, center_wavelength + 100e-9, 11)

# direction of source and flux monitors
direction = "+x"

# mode for the source and flux monitors
mode = 0
TE = True

# create a geometry list
geometry = [{"type":"rectangle","size":[simulation_sizex,0.5e-6,0],"center":[0,0,0], "refractive index": 2.5}] # waveguide

# create a source list
source = [{"type":"mode","size":[0,5e-6,0],"center":[-simulation_sizex/2+1.5e-6,0,0], "pulse width":100e-15, "mode": mode, "wavelength":center_wavelength,"direction":direction}]

# list of mode monitors
fluxes = [{"direction": "+x","center":[-simulation_sizex/2 + 2.0e-6, 0, 0],"size":[0,4e-6],"wavelengths": wavelengths, "mode": mode}] # incident monitor
fluxes += [{"direction": "+x","center":[simulation_sizex/2 - 1.0e-6, 0, 0],"size":[0,4e-6],"wavelengths": wavelengths, "mode": mode}] # transmittance monitor

# list of DFT monitors
dfts = [{"type": "normal", "center": [0,0,0], "size":[simulation_sizex,simulation_sizey],"wavelengths": np.array([center_wavelength])}]

# movie monitor
movies = [{"center": [0,0,0], "size":[simulation_sizex,simulation_sizey]}]

# Define siulation object
sim = FDTD_2D([simulation_sizex,simulation_sizey],
                   step_size,
                   source=source,
                   geometry = geometry,
                   flux_region=fluxes,
                   dft_region = dfts,
                   movie_region = movies,
                   cutoff = 1e-5,
                   movie_update=100,
                   TE = TE)

# Run simulation which returns all monitor data
movie_region, flux_region, dft_region = sim.run()

# plot the mode transmittance/reflectance
trans = np.abs(flux_region[1]["mode amplitude +x"]) ** 2
inc = np.abs(flux_region[0]["mode amplitude +x"]) ** 2
ref = np.abs(flux_region[0]["mode amplitude -x"]) ** 2

plt.plot(wavelengths, trans/inc)
plt.plot(wavelengths, ref/inc)
plt.legend(["transmittance","reflectance"])
plt.savefig("trans-ref" + direction + ".png")
plt.show()


# plot DFT fields
for d in dft_region:
    for w, wl in enumerate(d["wavelengths"]):
        E = np.abs(d["Ex"]) ** 2 + np.abs(d["Ey"]) ** 2 + np.abs(d["Ez"]) ** 2
        plt.imshow(E[w,:,:])
        plt.colorbar(label = f"|E|^2 @ {wl * 1e6} um")
        plt.savefig(f"E_{wl}.png")
        plt.show()


E_movie = movie_region[0]["E"]
frames = [] # for storing the generated images
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
emax = np.max(np.array(E_movie)) / 1.1
emin = np.min(np.array(E_movie)) / 1.1
for i in range(len(E_movie)):
    plt.imshow(E_movie[i])
    im = ax.imshow(E_movie[i],cmap = 'seismic', vmax = emax, vmin = emin)
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                repeat_delay=1000)
plt.show()