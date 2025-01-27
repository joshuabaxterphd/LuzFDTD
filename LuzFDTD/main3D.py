import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FDTD3D_jax import FDTD_3D

simulation_sizex = 4e-6
simulation_sizey = 7e-6
simulation_sizez = 3e-6

step_size = 50e-9 # dy
Nx = int(simulation_sizex/step_size)
Ny = int(simulation_sizey/step_size)
eps = np.ones((Nx,Ny))

ri = np.sqrt(eps)

# set up pulse stuff
center_wavelength = 1550e-9

wavelengths = np.linspace(center_wavelength - 100e-9, center_wavelength + 100e-9, 5)


directions = ["+x","-x","+y","-y"]
directions = ["+y"]
mode = 1
TE = False
for direction in directions:

    if direction == "+x":
        fluxes = [{"direction": "+x","center":[-simulation_sizex/2 + 2.0e-6, 0, 0],"size":[0,4e-6,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "+x","center":[simulation_sizex/2 - 1.0e-6, 0, 0],"size":[0,4e-6,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "+x","center":[-simulation_sizex/2 + 1.0e-6, 0, 0],"size":[0,4e-6,3e-6],"wavelengths": wavelengths, "mode": mode}]
        geometry = [{"type":"rectangle","size":[simulation_sizex,0.5e-6,0.25e-6],"center":[0,0,0], "refractive index": 3.5}]
        source = [{"type":"mode","size":[0,4e-6,3e-6],"center":[-simulation_sizex/2+1.5e-6,0,0], "mode": mode, "wavelength":1.55e-6,"direction":direction}]

    if direction == "-x":
        fluxes = [{"direction": "-x","center":[simulation_sizex/2 - 2.0e-6, 0, 0],"size":[0,4e-6,3e-6,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "-x","center":[-simulation_sizex/2 + 1.0e-6, 0, 0],"size":[0,4e-6,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "-x","center":[simulation_sizex/2 - 1.0e-6, 0, 0],"size":[0,4e-6,3e-6],"wavelengths": wavelengths, "mode": mode}]
        geometry = [{"type":"rectangle","size":[simulation_sizex,0.5e-6,0.25e-6],"center":[0,0,0], "refractive index": 3.5}]
        source = [{"type":"mode","size":[0,5e-6,3e-6],"center":[simulation_sizex/2-1.5e-6,0,0], "mode": mode, "wavelength":1.55e-6,"direction":direction}]

    if direction == "+y":
        fluxes = [{"direction": "+y","center":[0,-simulation_sizey/2+2.0e-6, 0],"size":[4e-6,0,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "+y","center":[0, simulation_sizey/2 - 1.0e-6, 0],"size":[4e-6,0,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "+y","center":[0, -simulation_sizey/2 + 1.0e-6, 0],"size":[4e-6,0,3e-6],"wavelengths": wavelengths, "mode": mode}]
        geometry = [{"type":"rectangle","size":[0.5e-6,simulation_sizey,0.25e-6],"center":[0,0,0], "refractive index": 3.0}]
        source = [{"type":"mode","size":[4e-6,0,3e-6],"center":[0,-simulation_sizey/2+1.5e-6,0], "mode": mode, "wavelength":1.55e-6,"direction":direction}]

    if direction == "-y":
        fluxes = [{"direction": "-y","center":[0,+simulation_sizey/2-2.0e-6, 0],"size":[4e-6,0,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "-y","center":[0, -simulation_sizey/2 + 1.0e-6, 0],"size":[4e-6,0,3e-6],"wavelengths": wavelengths, "mode": mode}]
        fluxes += [{"direction": "-y","center":[0, simulation_sizey/2 - 1.0e-6, 0],"size":[4e-6,0,3e-6],"wavelengths": wavelengths, "mode": mode}]
        geometry = [{"type":"rectangle","size":[0.5e-6,simulation_sizey,0.25e-6],"center":[0,0,0], "refractive index": 3.0}]
        source = [{"type":"mode","size":[4e-6,0,3e-6],"center":[0,simulation_sizey/2-1.5e-6,0], "mode": mode, "wavelength":1.55e-6,"direction":direction}]

    sim = FDTD_3D([simulation_sizex,simulation_sizey,simulation_sizez],
                       step_size,
                       source=source,
                       geometry = geometry,
                       flux_region=fluxes,
                       cutoff = 1e-3,
                       movie_update=100,
                       TE = TE)
    E_movie, flux_region, dft_region = sim.run()
    if direction == "+y":
        trans = np.abs(flux_region[1]["mode amplitude +y"]) ** 2
        inc0 = np.abs(flux_region[0]["mode amplitude +y"]) ** 2
        ref = np.abs(flux_region[0]["mode amplitude -y"]) ** 2
        ref2 = np.abs(flux_region[2]["mode amplitude -y"]) ** 2

    if direction == "-y":
        trans = np.abs(flux_region[1]["mode amplitude -y"]) ** 2
        inc0 = np.abs(flux_region[0]["mode amplitude -y"]) ** 2
        ref = np.abs(flux_region[0]["mode amplitude +y"]) ** 2
        ref2 = np.abs(flux_region[2]["mode amplitude +y"]) ** 2

    if direction == "+x":
        trans = np.abs(flux_region[1]["mode amplitude +x"]) ** 2
        inc0 = np.abs(flux_region[0]["mode amplitude +x"]) ** 2
        ref = np.abs(flux_region[0]["mode amplitude -x"]) ** 2
        ref2 = np.abs(flux_region[2]["mode amplitude -x"]) ** 2

    if direction == "-x":
        trans = np.abs(flux_region[1]["mode amplitude -x"]) ** 2
        inc0 = np.abs(flux_region[0]["mode amplitude -x"]) ** 2
        ref = np.abs(flux_region[0]["mode amplitude +x"]) ** 2
        ref2 = np.abs(flux_region[2]["mode amplitude +x"]) ** 2

    plt.plot(wavelengths, trans/inc0)
    plt.savefig("trans" + direction + ".png")
    plt.show()
    plt.plot(wavelengths, ref/inc0)
    plt.savefig("ref" + direction + ".png")
    plt.show()
    plt.plot(wavelengths, ref2/inc0)
    plt.savefig("ref2" + direction + ".png")
    plt.show()
E_movie = E_movie[0]
frames = [] # for storing the generated images
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
emax = np.max(np.array(E_movie)) / 2
emin = np.min(np.array(E_movie)) / 2

print(np.max(E_movie))

for i in range(len(E_movie)):
    plt.imshow(E_movie[i])
    im = ax.imshow(E_movie[i],cmap = 'seismic', vmax = emax, vmin = emin)
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                repeat_delay=1000)
plt.show()