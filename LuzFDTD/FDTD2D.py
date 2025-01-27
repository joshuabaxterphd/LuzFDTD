import copy
import time as time_mod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import *
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from jax.scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
# from scipy.interpolate import RegularGridInterpolator
from .mode_solver import Mode_Solver

from jax import jacobian
from jax import vjp

eps0 = 8.8541878128e-12 # permittivity of free space
mu0 = 1.256637062e-6 # permeability of free space
c0 = 2.99792458e8 # speed of light in vacuum
imp0 = np.sqrt(mu0/eps0) # impedance of free space

complex_type = "complex64"
real_type = "float32"


def GetModeOverlap(Exsim, Eysim, Hxsim, Hysim, Exmode, Eymode, Hxmode, Hymode):
    demoninator = 1.0
    return np.sum(Exsim * np.conjugate(Hymode) - Eysim * np.conjugate(Hxmode) + np.conjugate(Exmode) *
                  Hysim - np.conjugate(Eymode) * Hxsim, axis=(1))

def Discrete_Fourier_Transform(field, time, omega):
    N_freq = omega.shape[0]
    field_omega = np.zeros(N_freq, dtype= 'complex128')
    for w in range(N_freq):
        field_omega[w] = np.sum(field * np.exp(-1j * omega[w] * time))
    return field_omega

def thin_film_TR(n1,n2,n3,wavelengths,thickness):
    r12 = (n1 - n2)/(n1 + n2)
    r23 = (n2 - n3)/(n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    beta = 2 * np.pi * n2 * thickness / wavelengths
    r = (r12 + r23 * np.exp(-2 * 1j * beta))/(1.0 + r12 * r23 * np.exp(-2 * 1j * beta))
    t = (t12 * t23 * np.exp(-1j * beta))/(1.0 + r12 * r23 * np.exp(-2 * 1j * beta))
    return (n3/n1) * np.abs(t) ** 2, np.abs(r) ** 2


def gaussian(time, pulse_width, pulse_delay, omega0, phase = 0, amplitude = 1.0):
    return amplitude * np.exp(- 2 * ((time - pulse_delay)/pulse_width) ** 2) * np.cos(omega0 * time + phase)

def complex_source(time,srcre,srcim):
    return srcre(time) + 1j * srcim(time)

def add_design_grid(ce,si,x_axis_sub,y_axis_sub):
    X_axis_sub, Y_axis_sub = np.meshgrid(x_axis_sub,y_axis_sub,indexing='ij')
    rect = ((np.abs(X_axis_sub - ce[0]) <= si[0]/2)
             & (np.abs(Y_axis_sub - ce[1]) <= si[1]/2))
    grid = rect * 0.5
    return grid

def density_to_perm(dens,epsmin,epsmax):
    return dens * epsmax + (1. - dens) * epsmin


class FDTD_2D:
    def __init__(self, simulation_size, step_size,
            geometry = [],
            source = [],
            simulation_time = 2000e-15,
            cutoff = 1e-4 ,
            dft_region = [],
            flux_region = [],
            courant = 0.5,
            movie_update = 10,
            n_cells_pml = 10,
            complex_sim = False,
            TE = False,
            staircasing = True):
        self.simulation_size = simulation_size
        self.step_size = step_size
        self.geometry = geometry
        self.source = source
        self.simulation_time = simulation_time
        self.cutoff = cutoff
        self.dft_region = dft_region
        self.flux_region = flux_region
        self.courant = courant
        self.movie_update = movie_update
        self.n_cells_pml = n_cells_pml
        self.complex_sim = complex_sim
        self.TE = TE
        self.staircasing = staircasing
        self.dx = self.step_size
        self.dy = self.step_size
        self.dt = self.courant * min([self.dx, self.dy]) / c0

    def run(self):
        if self.complex_sim:
            type = complex_type
        else:
            type = real_type

        dx = self.dx
        dy = self.dy
        dt = self.dt
        Sx, Sy = self.simulation_size
        x_axis = np.arange(-Sx / 2, Sx / 2, dx)
        y_axis = np.arange(-Sy / 2, Sy / 2, dy)
        x_axis_sub = np.arange(-Sx / 2, Sx / 2, dx / 2)
        y_axis_sub = np.arange(-Sy / 2, Sy / 2, dy / 2)
        # print(x_axis,x_axis_sub)
        x_axis_sub_ = np.arange(len(x_axis_sub)) * dx / 2 - Sx / 2
        # print(Sx / 2, dx / 2)
        x_axis_x = x_axis_sub[1::2]
        y_axis_x = y_axis_sub[::2]
        Xx, Yx = np.meshgrid(x_axis_x, y_axis_x, indexing='ij')
        x_axis_y = x_axis_sub[::2]
        y_axis_y = y_axis_sub[1::2]
        Xy, Yy = np.meshgrid(x_axis_y, y_axis_y, indexing='ij')
        x_axis_z = x_axis_sub[::2]
        y_axis_z = y_axis_sub[::2]
        Xz, Yz = np.meshgrid(x_axis_z, y_axis_z, indexing='ij')

        X_axis, Y_axis = np.meshgrid(x_axis, y_axis, indexing='ij')
        X_axis_sub, Y_axis_sub = np.meshgrid(x_axis_sub, y_axis_sub, indexing='ij')
        Nx = x_axis.shape[0]
        Ny = y_axis.shape[0]
        Domain_shape = (Nx, Ny)
        Nxs = x_axis_sub.shape[0]
        Nys = y_axis_sub.shape[0]
        Domain_shape_sub = (Nxs, Nys)
        Inv_dx = 1. / dx
        Inv_dy = 1. / dy

        epsxx = np.ones(Domain_shape)
        epsyy = np.ones(Domain_shape)
        epszz = np.ones(Domain_shape)
        eps = np.ones(Domain_shape)
        eps_sub = np.ones(Domain_shape_sub)
        sigma_exx = np.zeros(Domain_shape)
        sigma_eyy = np.zeros(Domain_shape)
        sigma_ezz = np.zeros(Domain_shape)
        sigma_e = np.zeros(Domain_shape)
        sigma_h = 0

        design_grid = []

        for g in self.geometry:
            if g["type"] == "rectangle":
                ce = g["center"]
                si = g["size"]
                if self.staircasing:
                    dx_ = 0.25 * dx
                    dy_ = 0.25 * dy
                    rectx = ((np.abs(X_axis + dx_ - ce[0]) <= si[0] / 2)
                             & (np.abs(Y_axis + dy_ - ce[1]) <= si[1] / 2))
                    epsxx[rectx] = g["refractive index"] ** 2
                    recty = ((np.abs(X_axis + dx_ - ce[0]) <= si[0] / 2)
                             & (np.abs(Y_axis + dy_ - ce[1]) <= si[1] / 2))
                    epsyy[recty] = g["refractive index"] ** 2
                    rectz = ((np.abs(X_axis + dx_ - ce[0]) <= si[0] / 2)
                             & (np.abs(Y_axis + dy_ - ce[1]) <= si[1] / 2))
                    epszz[rectz] = g["refractive index"] ** 2

                    rect = ((np.abs(X_axis - ce[0]) <= si[0] / 2)
                            & (np.abs(Y_axis - ce[1]) <= si[1] / 2))
                    eps[rect] = g["refractive index"] ** 2

                    rects = ((np.abs(X_axis_sub - ce[0]) <= si[0] / 2)
                             & (np.abs(Y_axis_sub - ce[1]) <= si[1] / 2))
                    eps_sub[rects] = g["refractive index"] ** 2
                else:
                    print("here")
                    rectx = ((np.abs(Xx - ce[0]) <= si[0] / 2)
                             & (np.abs(Yx - ce[1]) <= si[1] / 2))
                    epsxx[rectx] = g["refractive index"] ** 2

                    recty = ((np.abs(Xy - ce[0]) <= si[0] / 2)
                             & (np.abs(Yy - ce[1]) <= si[1] / 2))
                    epsyy[recty] = g["refractive index"] ** 2
                    rectz = ((np.abs(Xz - ce[0]) <= si[0] / 2)
                             & (np.abs(Yz - ce[1]) <= si[1] / 2))
                    epszz[rectz] = g["refractive index"] ** 2

                    rect = ((np.abs(X_axis - ce[0]) <= si[0] / 2)
                            & (np.abs(Y_axis - ce[1]) <= si[1] / 2))
                    eps[rect] = g["refractive index"] ** 2

                    rects = ((np.abs(X_axis_sub - ce[0]) <= si[0] / 2)
                             & (np.abs(Y_axis_sub - ce[1]) <= si[1] / 2))
                    eps_sub[rects] = g["refractive index"] ** 2

            if g["type"] == "design":
                ce = g["center"]
                si = g["size"]

                if self.staircasing:
                    dx_ = 0.25 * dx
                    dy_ = 0.25 * dy
                    nx, ny = g["grid"].shape
                    imin = np.argmin(np.abs(x_axis + dx_ - ce[0] + si[0] / 2))
                    imax = imin + int(round(si[0] / dx))
                    nx_ = int(np.floor((imax - imin) / nx))
                    i_list = [0]
                    remainder = (imax - imin) - nx * nx_
                    print(imax - imin, nx_ * nx, remainder)
                    for i in range(nx):
                        if i < remainder:
                            i_list.append(i_list[-1] + nx_ + 1)
                        else:
                            i_list.append(i_list[-1] + nx_)

                    jmin = np.argmin(np.abs(y_axis + dy_ - ce[1] + si[1] / 2))
                    jmax = jmin + int(round(si[1] / dy))
                    ny_ = int(np.floor((jmax - jmin) / ny))
                    j_list = [0]
                    remainder = (jmax - jmin) - ny * ny_
                    print(jmax - jmin, ny_ * ny, remainder)
                    for j in range(ny):
                        if j < remainder:
                            j_list.append(j_list[-1] + ny_ + 1)
                        else:
                            j_list.append(j_list[-1] + ny_)
                    g["i_list"] = copy.deepcopy(i_list)
                    g["j_list"] = copy.deepcopy(j_list)
                    i_list = [il + imin for il in i_list]
                    j_list = [jl + jmin for jl in j_list]
                    if "loss" in g:
                        sigma_exx[imin:imax, jmin:jmax] = g["loss"]
                        sigma_eyy[imin:imax, jmin:jmax] = g["loss"]
                        sigma_ezz[imin:imax, jmin:jmax] = g["loss"]

                    for i in range(1, len(i_list)):
                        for j in range(1, len(j_list)):
                            epsxx[i_list[i - 1]:i_list[i], j_list[j - 1]:j_list[j]] = density_to_perm(
                                g["grid"][i - 1, j - 1], g["ri min"] ** 2, g["ri max"] ** 2)
                            epsyy[i_list[i - 1]:i_list[i], j_list[j - 1]:j_list[j]] = density_to_perm(
                                g["grid"][i - 1, j - 1], g["ri min"] ** 2, g["ri max"] ** 2)
                            epszz[i_list[i - 1]:i_list[i], j_list[j - 1]:j_list[j]] = density_to_perm(
                                g["grid"][i - 1, j - 1], g["ri min"] ** 2, g["ri max"] ** 2)
                    design_grid += [g]
                else:
                    nx, ny = g["grid"].shape
                    imin = 0
                    imax = 10000
                    jmin = 0
                    jmax = 10000
                    for i in range(1, len(x_axis_sub) - 2):
                        if x_axis_sub[i] - ce[0] >= -si[0] / 2 and x_axis_sub[i - 1] - ce[0] < -si[0] / 2:
                            imin = i

                        if x_axis_sub[i] - ce[0] <= si[0] / 2 and x_axis_sub[i + 1] - ce[0] > si[0] / 2:
                            imax = i
                            break
                    for i in range(1, len(y_axis_sub) - 2):
                        if y_axis_sub[i] - ce[1] >= -si[1] / 2 and y_axis_sub[i - 1] - ce[1] < -si[1] / 2:
                            jmin = i
                        if y_axis_sub[i] - ce[1] <= si[1] / 2 and y_axis_sub[i + 1] - ce[1] > si[1] / 2:
                            jmax = i
                            break

                    print(imin, imax, jmin, jmax)
                    x_g = x_axis_sub[imin:imax + 1]
                    y_g = y_axis_sub[jmin:jmax + 1]
                    Xg, Yg = np.meshgrid(x_g, y_g, indexing='ij')

                    grid = g["grid"]
                    eps_ = density_to_perm(grid, g["ri min"] ** 2, g["ri max"] ** 2).reshape(nx, ny)
                    interp = RegularGridInterpolator((x_g, y_g), eps_, method="nearest", fill_value=None)
                    print(np.min(x_g), np.max(x_g), np.min(y_g), np.max(y_g), y_axis_sub[jmin - 1])
                    eps_sub[imin:imax + 1, jmin:jmax + 1] = interp(
                        np.array([Xg.flatten(), Yg.flatten()]).transpose()).reshape(len(x_g), len(y_g))

                    for i in range(1, len(x_axis) - 2):
                        if x_axis_x[i] - ce[0] >= -si[0] / 2 and x_axis_x[i - 1] - ce[0] < -si[0] / 2:
                            imin = i
                        if x_axis_x[i] - ce[0] <= si[0] / 2 and x_axis_x[i + 1] - ce[0] > si[0] / 2:
                            imax = i
                            break
                    for i in range(1, len(y_axis) - 2):
                        if y_axis_x[i] - ce[1] >= -si[1] / 2 and y_axis_x[i - 1] - ce[1] < -si[1] / 2:
                            jmin = i
                        if y_axis_x[i] - ce[1] <= si[1] / 2 and y_axis_x[i + 1] - ce[1] > si[1] / 2:
                            jmax = i
                            break
                    x_gx = x_axis_x[imin:imax + 1]
                    y_gx = y_axis_x[jmin:jmax + 1]
                    Xgx, Ygx = np.meshgrid(x_gx, y_gx, indexing='ij')
                    print(np.min(x_gx), np.max(x_gx), np.min(y_gx), np.max(y_gx))

                    def gridderx(grid):
                        interp = RegularGridInterpolator((x_g, y_g), grid, method="nearest", fill_value=None,
                                                         bounds_error=False)
                        inputs = np.array([Xgx.flatten(), Ygx.flatten()]).transpose()
                        vals = interp(inputs)
                        return vals

                    valsx, jacx = vjp(gridderx, eps_)
                    print("valsx")
                    # valsx = gridderx(eps_)
                    # jacx = jacobian(gridderx,0)(eps_)
                    epsxx[imin:imax + 1, jmin:jmax + 1] = valsx.reshape(len(x_gx), len(y_gx))
                    g["iminx"] = imin
                    g["imaxx"] = imax + 1
                    g["jminx"] = jmin
                    g["jmaxx"] = jmax + 1
                    g["jacx"] = jacx
                    print((len(x_gx), len(y_gx)), valsx.shape)

                    for i in range(1, len(x_axis) - 2):
                        if x_axis_y[i] - ce[0] >= -si[0] / 2 and x_axis_y[i - 1] - ce[0] < -si[0] / 2:
                            imin = i
                        if x_axis_y[i] - ce[0] <= si[0] / 2 and x_axis_y[i + 1] - ce[0] > si[0] / 2:
                            imax = i
                            break
                    for i in range(1, len(y_axis) - 2):
                        if y_axis_y[i] - ce[1] >= -si[1] / 2 and y_axis_y[i - 1] - ce[1] < -si[1] / 2:
                            jmin = i
                        if y_axis_y[i] - ce[1] <= si[1] / 2 and y_axis_y[i + 1] - ce[1] > si[1] / 2:
                            jmax = i
                            break
                    x_gy = x_axis_y[imin:imax + 1]
                    y_gy = y_axis_y[jmin:jmax + 1]
                    Xgy, Ygy = np.meshgrid(x_gy, y_gy, indexing='ij')

                    def griddery(grid):
                        interp = RegularGridInterpolator((x_g, y_g), grid, method="nearest", fill_value=None,
                                                         bounds_error=False)
                        inputs = np.array([Xgy.flatten(), Ygy.flatten()]).transpose()
                        vals = interp(inputs)
                        return vals

                    valsy, jacy = vjp(griddery, eps_)  #
                    # valsy = griddery(eps_)
                    epsyy[imin:imax + 1, jmin:jmax + 1] = valsy.reshape(len(x_gy), len(y_gy))
                    # jacy = jacobian(griddery,0)(eps_)

                    print((len(x_gy), len(y_gy)))
                    g["iminy"] = imin
                    g["imaxy"] = imax + 1
                    g["jminy"] = jmin
                    g["jmaxy"] = jmax + 1
                    g["jacy"] = jacy

                    for i in range(1, len(x_axis) - 2):
                        if x_axis_z[i] - ce[0] >= -si[0] / 2 and x_axis_z[i - 1] - ce[0] < -si[0] / 2:
                            imin = i
                        if x_axis_z[i] - ce[0] <= si[0] / 2 and x_axis_z[i + 1] - ce[0] > si[0] / 2:
                            imax = i
                            break
                    for i in range(1, len(y_axis) - 2):
                        if y_axis_z[i] - ce[1] >= -si[1] / 2 and y_axis_z[i - 1] - ce[1] < -si[1] / 2:
                            jmin = i
                        if y_axis_z[i] - ce[1] <= si[1] / 2 and y_axis_z[i + 1] - ce[1] > si[1] / 2:
                            jmax = i
                            break
                    x_gz = x_axis_z[imin:imax + 1]
                    y_gz = y_axis_z[jmin:jmax + 1]
                    print((len(x_gz), len(y_gz)))
                    Xgz, Ygz = np.meshgrid(x_gz, y_gz, indexing='ij')

                    def gridderz(grid):
                        interp = RegularGridInterpolator((x_g, y_g), grid, method="nearest", fill_value=None,
                                                         bounds_error=False)
                        inputs = np.array([Xgz.flatten(), Ygz.flatten()]).transpose()
                        vals = interp(inputs)
                        return vals

                    valsz, jacz = vjp(gridderz, eps_)
                    # valsz = gridderz(eps_)
                    epszz[imin:imax + 1, jmin:jmax + 1] = valsz.reshape(len(x_gz), len(y_gz))
                    # jacz = jacobian(gridderz,0)(eps_)
                    g["iminz"] = imin
                    g["imaxz"] = imax + 1
                    g["jminz"] = jmin
                    g["jmaxz"] = jmax + 1
                    g["jacz"] = jacz

                    if "loss" in g:
                        sigma_exx[g["iminx"]:g["imaxx"], g["jminx"]:g["imaxx"]] = g["loss"]
                        sigma_eyy[g["iminy"]:g["imaxy"], g["jminy"]:g["imaxy"]] = g["loss"]
                        sigma_ezz[g["iminz"]:g["imaxz"], g["jminz"]:g["imaxz"]] = g["loss"]

                    design_grid += [g]

        plt.contourf(x_axis_sub * 1e6, y_axis_sub * 1e6, eps_sub.transpose(), 200)
        plt.xlabel("x [um]")
        plt.ylabel("y [um]")
        plt.savefig("sim.png")
        plt.close()
        plt.contourf(x_axis * 1e6, y_axis * 1e6, epsxx.transpose(), 200)
        plt.xlabel("x [um]")
        plt.ylabel("y [um]")
        plt.savefig("simxx.png")
        plt.close()
        plt.contourf(x_axis * 1e6, y_axis * 1e6, epsyy.transpose(), 200)
        plt.xlabel("x [um]")
        plt.ylabel("y [um]")
        plt.savefig("simyy.png")
        plt.close()
        plt.contourf(x_axis * 1e6, y_axis * 1e6, epszz.transpose(), 200)
        plt.xlabel("x [um]")
        plt.ylabel("y [um]")
        plt.savefig("simzz.png")
        plt.close()

        plt.imshow(epsxx - np.flip(epsxx, axis=1))
        plt.colorbar()
        plt.savefig("simzxxflip.png")
        plt.close()

        for s in self.source:
            if not "pulse width" in s:
                pulse_width = 10e-15
                s["pulse width"] = pulse_width
            if not "amplitude" in s:
                amplitude = 1.0
                s["amplitude"] = amplitude
            if not "phase" in s:
                pulse_phase = 0.0
                s["phase"] = pulse_phase
            if not "delay" in s:
                pulse_delay_fact = 4
                s["delay"] = pulse_delay_fact * s["pulse width"]
            if "pulse" not in s:
                pulse = None
                s["pulse"] = None
            if s["pulse"] is None:

                omega0 = 2 * np.pi * c0 / s["wavelength"]

                signal = partial(gaussian, pulse_width=s["pulse width"],
                                 pulse_delay=s["delay"],
                                 omega0=omega0,
                                 phase=s["phase"],
                                 amplitude=1)

                pulse = signal(np.arange(int(self.simulation_time / dt)) * dt)
                pulsew = Discrete_Fourier_Transform(pulse, np.arange(int(self.simulation_time / dt)) * dt,
                                                    np.array([omega0]))

                s["signal"] = partial(gaussian, pulse_width=s["pulse width"],
                                      pulse_delay=s["delay"],
                                      omega0=omega0,
                                      phase=s["phase"],
                                      amplitude=s["amplitude"] / np.abs(pulsew)[0])

                min_simulation_time = s["delay"] + s["pulse width"]
                print(min_simulation_time / dt)
            else:
                if type == "complex128" or type == "complex64":
                    s["signal"] = partial(complex_source, srcre=s["pulse"][0], srcim=s["pulse"][1])
                else:
                    s["signal"] = s["pulse"][0]
                min_simulation_time = s["pulse"][2] * dt

            if s["type"] == "mode" or s["type"] == "adjmode":
                si = s["size"]
                ce = s["center"]
                imin = np.argmin(np.abs(x_axis - (ce[0] - si[0] / 2)))
                imax = np.argmin(np.abs(x_axis - (ce[0] + si[0] / 2)))
                jmin = np.argmin(np.abs(y_axis - (ce[1] - si[1] / 2)))
                jmax = np.argmin(np.abs(y_axis - (ce[1] + si[1] / 2)))

                if imin == imax:
                    imax = imin + 1
                    epsyy_ = epszz[imin:imax, jmin:jmax].transpose()
                    epszz_ = epsxx[imin:imax, jmin:jmax].transpose()
                    epsxx_ = epsyy[imin:imax, jmin:jmax].transpose()
                    if "direction" not in s:
                        s["direction"] = "+x"
                elif jmin == jmax:
                    jmax = jmax + 1
                    epsxx_ = epsxx[imin:imax, jmin:jmax]  # .transpose()
                    epsyy_ = epszz[imin:imax, jmin:jmax]  # .transpose()
                    epszz_ = epsyy[imin:imax, jmin:jmax]  # .transpose()
                    if "direction" not in s:
                        s["direction"] = "+y"
                s["imin"] = imin
                s["imax"] = imax
                s["jmin"] = jmin
                s["jmax"] = jmax

                n, E1, E2, H1, H2 = Mode_Solver(epsxx_, epsyy_, epszz_, dx, dy, s["wavelength"], s["mode"])

                if type.startswith("float"):
                    E1 = E1.real
                    E2 = E2.real
                    H1 = H1.real
                    H2 = H2.real

                norm = 1.0
                s["mode index"] = n
                if s["direction"] == "+x":
                    s["Ey"] = E1.copy()[:, 0] / norm
                    s["Ez"] = E2.copy()[:, 0] / norm
                    s["Hy"] = H1.copy()[:, 0] / norm
                    s["Hz"] = H2.copy()[:, 0] / norm
                elif s["direction"] == "-x":
                    s["Ey"] = E1.copy()[:, 0] / norm
                    s["Ez"] = E2.copy()[:, 0] / norm
                    s["Hy"] = -H1.copy()[:, 0] / norm
                    s["Hz"] = -H2.copy()[:, 0] / norm
                elif s["direction"] == "+y":
                    s["Ex"] = E1.copy()[:, 0] / norm
                    s["Ez"] = E2.copy()[:, 0] / norm
                    s["Hx"] = -H1.copy()[:, 0] / norm
                    s["Hz"] = -H2.copy()[:, 0] / norm
                elif s["direction"] == "-y":
                    s["Ex"] = E1.copy()[:, 0] / norm
                    s["Ez"] = E2.copy()[:, 0] / norm
                    s["Hx"] = H1.copy()[:, 0] / norm
                    s["Hz"] = H2.copy()[:, 0] / norm

                s["t_offset"] = n * self.step_size / (2 * c0)
                s["Z"] = imp0 / n

        Ex = np.zeros(Domain_shape, dtype=type)
        Ey = np.zeros(Domain_shape, dtype=type)
        Ez = np.zeros(Domain_shape, dtype=type)
        Hx = np.zeros(Domain_shape, dtype=type)
        Hy = np.zeros(Domain_shape, dtype=type)
        Hz = np.zeros(Domain_shape, dtype=type)

        n_cells_pml = self.n_cells_pml

        psi_Hxy = np.zeros((Nx, n_cells_pml, 2), dtype=type)
        psi_Hyx = np.zeros((n_cells_pml, Ny, 2), dtype=type)
        psi_Hzx = np.zeros((n_cells_pml, Ny, 2), dtype=type)
        psi_Hzy = np.zeros((Nx, n_cells_pml, 2), dtype=type)
        psi_Exy = np.zeros((Nx, n_cells_pml, 2), dtype=type)
        psi_Eyx = np.zeros((n_cells_pml, Ny, 2), dtype=type)
        psi_Ezx = np.zeros((n_cells_pml, Ny, 2), dtype=type)
        psi_Ezy = np.zeros((Nx, n_cells_pml, 2), dtype=type)

        simulation_time = max([self.simulation_time, min_simulation_time])
        N_time_steps = int(simulation_time / dt)
        print(f"there are {N_time_steps} FDTD time steps")

        E_movie = []

        # pulse = signal((np.arange(N_time_steps)+1)*dt)

        # setup pmls
        cpml_exp = 3  # should be 3 or 4
        sigma_max = -(cpml_exp + 1) * 0.8 / (imp0 * dx)
        alpha_max = 0.05
        kappa_max = 5

        # setup electric field PMLs
        sigma = sigma_max * ((n_cells_pml - 1 - np.arange(n_cells_pml)) / n_cells_pml) ** cpml_exp
        alpha = alpha_max * ((np.arange(n_cells_pml) - 1) / n_cells_pml) ** 1
        kappa = 1 + (kappa_max - 1) * ((n_cells_pml - 1 - np.arange(n_cells_pml)) / n_cells_pml) ** cpml_exp

        # setup electric field PMLs
        sigmah = sigma_max * ((n_cells_pml - 1 - (np.arange(n_cells_pml) + 0.5)) / n_cells_pml) ** cpml_exp
        alpha_h = alpha_max * ((np.arange(n_cells_pml) + 0.5 - 1) / n_cells_pml) ** 1
        kappa_h = 1 + (kappa_max - 1) * ((n_cells_pml - 1 - (np.arange(n_cells_pml) + 0.5)) / n_cells_pml) ** cpml_exp

        bh_x = np.exp((sigmah / kappa_h + alpha_h) * dt / eps0)
        bh_x_f = np.flip(bh_x)
        bh_y = np.exp((sigmah / kappa_h + alpha_h) * dt / eps0)
        bh_y_f = np.flip(bh_y)
        ch_x = sigmah * (bh_x - 1.0) / (sigmah + kappa_h * alpha_h) / kappa_h
        ch_x_f = np.flip(ch_x)
        ch_y = sigmah * (bh_y - 1.0) / (sigmah + kappa_h * alpha_h) / kappa_h
        ch_y_f = np.flip(ch_y)

        be_x = np.exp((sigma / kappa + alpha) * dt / eps0)
        be_x_f = np.flip(be_x)
        be_y = np.exp((sigma / kappa + alpha) * dt / eps0)
        be_y_f = np.flip(be_y)
        ce_x = sigma * (be_x - 1.0) / (sigma + kappa * alpha) / kappa
        ce_x_f = np.flip(ce_x)
        ce_y = sigma * (be_y - 1.0) / (sigma + kappa * alpha) / kappa
        ce_y_f = np.flip(ce_y)

        kappa_e_x = np.ones(eps.shape[0])
        kappa_e_y = np.ones(eps.shape[1])
        kappa_h_x = np.ones(eps.shape[0])
        kappa_h_y = np.ones(eps.shape[1])

        kappa_e_x[:len(kappa)] = kappa
        kappa_e_x[-len(kappa):] = np.flip(kappa)
        kappa_e_y[:len(kappa)] = kappa
        kappa_e_y[-len(kappa):] = np.flip(kappa)
        kappa_h_x[:len(kappa_h)] = kappa_h
        kappa_h_x[-len(kappa_h) - 1:-1] = np.flip(kappa_h)
        kappa_h_y[:len(kappa_h)] = kappa_h
        kappa_h_y[-len(kappa_h) - 1:-1] = np.flip(kappa_h)

        kappa_e_x = np.expand_dims(kappa_e_x, axis=1)
        kappa_e_y = np.expand_dims(kappa_e_y, axis=0)
        ce_x = np.expand_dims(ce_x, axis=1)
        ce_y = np.expand_dims(ce_y, axis=0)
        be_x = np.expand_dims(be_x, axis=1)
        be_y = np.expand_dims(be_y, axis=0)
        be_x_f = np.expand_dims(be_x_f, axis=1)
        be_y_f = np.expand_dims(be_y_f, axis=0)
        ce_x_f = np.expand_dims(ce_x_f, axis=1)
        ce_y_f = np.expand_dims(ce_y_f, axis=0)

        kappa_h_x = np.expand_dims(kappa_h_x, axis=1)
        kappa_h_y = np.expand_dims(kappa_h_y, axis=0)
        ch_x = np.expand_dims(ch_x, axis=1)
        ch_y = np.expand_dims(ch_y, axis=0)
        bh_x = np.expand_dims(bh_x, axis=1)
        bh_y = np.expand_dims(bh_y, axis=0)
        ch_x_f = np.expand_dims(ch_x_f, axis=1)
        ch_y_f = np.expand_dims(ch_y_f, axis=0)
        bh_x_f = np.expand_dims(bh_x_f, axis=1)
        bh_y_f = np.expand_dims(bh_y_f, axis=0)

        # Electric field update coefficients
        denominatorx = eps0 * epsxx / dt + sigma_exx / 2
        e_coeff_1x = (eps0 * epsxx / dt - sigma_exx / 2) / denominatorx
        denominatory = eps0 * epsyy / dt + sigma_eyy / 2
        e_coeff_1y = (eps0 * epsyy / dt - sigma_eyy / 2) / denominatory
        denominatorz = eps0 * epszz / dt + sigma_ezz / 2
        e_coeff_1z = (eps0 * epszz / dt - sigma_ezz / 2) / denominatorz

        e_coeffx = 1.0 / denominatorx
        e_coeffy = 1.0 / denominatory
        e_coeffz = 1.0 / denominatorz

        e_coeff_yx = e_coeffy / (dx * kappa_e_x)
        e_coeff_xy = e_coeffx / (dy * kappa_e_y)
        e_coeff_zx = e_coeffz / (dx * kappa_e_x)
        e_coeff_zy = e_coeffz / (dy * kappa_e_y)

        # Magnetic field update coefficients
        denominator_h = mu0 / dt + sigma_h / 2
        h_coeff_1 = (mu0 / dt - sigma_h / 2) / denominator_h
        h_coeff = 1.0 / (denominator_h)
        h_coeff_x = h_coeff / (dx * kappa_h_x)
        h_coeff_y = h_coeff / (dx * kappa_h_y)

        probe = []

        for dft in self.dft_region:
            ce = dft["center"]
            si = dft["size"]
            dft["omegas"] = 2 * np.pi * c0 / dft["wavelengths"]

            dx_ = 0.25 * dx
            dy_ = 0.25 * dy

            if dft["type"] == "design" and not self.staircasing:
                iminx = design_grid[0]["iminx"]
                imaxx = design_grid[0]["imaxx"]
                jminx = design_grid[0]["jminx"]
                jmaxx = design_grid[0]["jmaxx"]
                iminy = design_grid[0]["iminy"]
                imaxy = design_grid[0]["imaxy"]
                jminy = design_grid[0]["jminy"]
                jmaxy = design_grid[0]["jmaxy"]
                iminz = design_grid[0]["iminz"]
                imaxz = design_grid[0]["imaxz"]
                jminz = design_grid[0]["jminz"]
                jmaxz = design_grid[0]["jmaxz"]

                dft["iminx"] = iminx
                dft["imaxx"] = imaxx
                dft["jminx"] = jminx
                dft["jmaxx"] = jmaxx
                dft["iminy"] = iminy
                dft["imaxy"] = imaxy
                dft["jminy"] = jminy
                dft["jmaxy"] = jmaxy
                dft["iminz"] = iminz
                dft["imaxz"] = imaxz
                dft["jminz"] = jminz
                dft["jmaxz"] = jmaxz
                print(imaxx - iminx)

                dft["Ex"] = np.zeros([len(dft["wavelengths"]), imaxx - iminx, jmaxx - jminx], dtype=complex_type)
                dft["Ey"] = np.zeros([len(dft["wavelengths"]), imaxy - iminy, jmaxy - jminy], dtype=complex_type)
                dft["Ez"] = np.zeros([len(dft["wavelengths"]), imaxz - iminz, jmaxz - jminz], dtype=complex_type)

            else:
                imin = np.argmin(np.abs(x_axis + dx_ - ce[0] + si[0] / 2))
                imax = imin + int(round(si[0] / dx))
                jmin = np.argmin(np.abs(y_axis + dy_ - ce[1] + si[1] / 2))
                jmax = jmin + int(round(si[1] / dy))
                dft["iminx"] = imin
                dft["imaxx"] = imax
                dft["jminx"] = jmin
                dft["jmaxx"] = jmax
                dft["iminy"] = imin
                dft["imaxy"] = imax
                dft["jminy"] = jmin
                dft["jmaxy"] = jmax
                dft["iminz"] = imin
                dft["imaxz"] = imax
                dft["jminz"] = jmin
                dft["jmaxz"] = jmax
                dft["Ex"] = np.zeros([len(dft["wavelengths"]), imax - imin, jmax - jmin], dtype=complex_type)
                dft["Ey"] = np.zeros([len(dft["wavelengths"]), imax - imin, jmax - jmin], dtype=complex_type)
                dft["Ez"] = np.zeros([len(dft["wavelengths"]), imax - imin, jmax - jmin], dtype=complex_type)

        for fr in self.flux_region:
            si = fr["size"]
            ce = fr["center"]
            imin = np.argmin(np.abs(x_axis - (ce[0] - si[0] / 2)))
            imax = np.argmin(np.abs(x_axis - (ce[0] + si[0] / 2)))
            jmin = np.argmin(np.abs(y_axis - (ce[1] - si[1] / 2)))
            jmax = np.argmin(np.abs(y_axis - (ce[1] + si[1] / 2)))

            if imin == imax:
                imax = imin + 1
                if "direction" not in fr:
                    fr["direction"] = "+x"
            if jmin == jmax:
                jmax = jmin + 1
                if "direction" not in fr:
                    fr["direction"] = "+y"

            fr["imin"] = imin
            fr["imax"] = imax
            fr["jmin"] = jmin
            fr["jmax"] = jmax

            if fr["direction"] == "+y" or fr["direction"] == "-y":
                fr["Ex"] = np.zeros([len(fr["wavelengths"]), abs(fr["imax"] - fr["imin"])], dtype="complex128")
                fr["Ez"] = np.zeros([len(fr["wavelengths"]), abs(fr["imax"] - fr["imin"])], dtype="complex128")
                fr["Hx"] = np.zeros([len(fr["wavelengths"]), abs(fr["imax"] - fr["imin"])], dtype="complex128")
                fr["Hz"] = np.zeros([len(fr["wavelengths"]), abs(fr["imax"] - fr["imin"])], dtype="complex128")
                fr["direction_"] = "y"
            if fr["direction"] == "+x" or fr["direction"] == "-x":
                fr["Ey"] = np.zeros([len(fr["wavelengths"]), abs(fr["jmax"] - fr["jmin"])], dtype="complex128")
                fr["Ez"] = np.zeros([len(fr["wavelengths"]), abs(fr["jmax"] - fr["jmin"])], dtype="complex128")
                fr["Hy"] = np.zeros([len(fr["wavelengths"]), abs(fr["jmax"] - fr["jmin"])], dtype="complex128")
                fr["Hz"] = np.zeros([len(fr["wavelengths"]), abs(fr["jmax"] - fr["jmin"])], dtype="complex128")
                fr["direction_"] = "x"

            fr["omegas"] = 2 * np.pi * c0 / fr["wavelengths"]

        max_field = 0.0
        begin = time_mod.time()
        hupdate_time = []
        if self.TE:
            # FDTD algorithm
            for n in range(N_time_steps):
                time = (n + 1) * dt
                time_ = (n + 0.5) * dt

                Hx_prev = Hx.copy()
                Hy_prev = Hy.copy()
                Ez_prev = Ez.copy()

                # begin_h = time_mod.time()
                # update magnetic field at n+1/2
                Hx[1:-1, :-1] = (h_coeff_1 * Hx_prev[1:-1, :-1]
                                 - h_coeff_y[:, :-1] * (Ez[1:-1, 1:] - Ez[1:-1, :-1]))

                Hy[:-1, 1:-1] = (h_coeff_1 * Hy_prev[:-1, 1:-1]
                                 + h_coeff_x[:-1, :] * (Ez[1:, 1:-1] - Ez[:-1, 1:-1]))

                # Hx PML along y-direction
                psi_Hxy[:, :n_cells_pml - 1, 0] = (psi_Hxy[:, :n_cells_pml - 1, 0] * bh_y[:, :-1]
                                                   + ch_y[:, :-1] * Inv_dy * (
                                                               Ez[:, 1:n_cells_pml] - Ez[:, :n_cells_pml - 1]))

                psi_Hxy[:, :n_cells_pml - 1, 1] = (psi_Hxy[:, :n_cells_pml - 1, 1] * bh_y_f[:, 1:]
                                                   + ch_y_f[:, 1:] * Inv_dy * (
                                                               Ez[:, -n_cells_pml + 1:] - Ez[:, -n_cells_pml:-1]))

                Hx[:-1, :n_cells_pml - 1] -= h_coeff * psi_Hxy[:-1, :n_cells_pml - 1, 0]
                Hx[:-1, -n_cells_pml:-1] -= h_coeff * psi_Hxy[:-1, :n_cells_pml - 1, 1]

                # Hy PML along x-direction
                psi_Hyx[:n_cells_pml - 1, :, 0] = (psi_Hyx[:n_cells_pml - 1, :, 0] * bh_x[:-1, :]
                                                   + ch_x[:-1, :] * Inv_dx * (
                                                               Ez[1:n_cells_pml, :] - Ez[:n_cells_pml - 1, :]))

                psi_Hyx[:n_cells_pml - 1, :, 1] = (psi_Hyx[:n_cells_pml - 1, :, 1] * bh_x_f[1:, :]
                                                   + ch_x_f[1:, :] * Inv_dy * (
                                                               Ez[-n_cells_pml + 1:, :] - Ez[-n_cells_pml:-1, :]))

                Hy[:n_cells_pml - 1, :-1] += h_coeff * psi_Hyx[:n_cells_pml - 1, :-1, 0]
                Hy[-n_cells_pml:-1, :-1] += h_coeff * psi_Hyx[:n_cells_pml - 1, :-1, 1]

                # end_h = time_mod.time()
                # hupdate_time.append(end_h - begin_h)
                # add magnetic field source
                for s in self.source:
                    source_dir = s["direction"]
                    imin = s["imin"]
                    imax = s["imax"]
                    jmin = s["jmin"]
                    jmax = s["jmax"]
                    t_offset = s["t_offset"]
                    Z = 1  # s["Z"]
                    # print(s["Hz"].shape, Hz[imin:imax,jmin-1].shape, (s["Hz"] - Hz[imin:imax,jmin-1]).shape)
                    if source_dir == "+y":
                        Hx[imin:imax, jmin - 1] = Hx[imin:imax, jmin - 1] + h_coeff * s["Ez"] * s["signal"](
                            (n + 0.5) * dt - t_offset) / dy
                    elif source_dir == "-y":
                        Hx[imin:imax, jmin - 1] = Hx[imin:imax, jmin - 1] - h_coeff * s["Ez"] * s["signal"](
                            (n + 0.5) * dt + t_offset) / dy
                    elif source_dir == "-x":
                        Hy[imin - 1, jmin:jmax] = Hy[imin - 1, jmin:jmax] + h_coeff * s["Ez"] * s["signal"](
                            (n + 0.5) * dt + t_offset) / dx
                    elif source_dir == "+x":
                        Hy[imin - 1, jmin:jmax] = Hy[imin - 1, jmin:jmax] - h_coeff * s["Ez"] * s["signal"](
                            (n + 0.5) * dt - t_offset) / dx
                            # pass
                # update electric field at n+1
                Ez[1:-1, 1:-1] = (e_coeff_1z[1:-1, 1:-1] * Ez_prev[1:-1, 1:-1]
                                  + e_coeff_zx[1:-1, 1:-1] * (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1])
                                  - e_coeff_zy[1:-1, 1:-1] * (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]))

                # Ez PML along x-direction
                psi_Ezx[1:n_cells_pml, :, 0] = (be_x[1:, :] * psi_Ezx[1:n_cells_pml, :, 0] +
                                                Inv_dx * ce_x[1:, :] * (Hy[1:n_cells_pml, :] - Hy[:n_cells_pml - 1, :]))
                psi_Ezx[1:n_cells_pml, :, 1] = (be_x_f[:-1, :] * psi_Ezx[1:n_cells_pml, :, 1] +
                                                Inv_dx * ce_x_f[:-1, :] * (
                                                            Hy[-n_cells_pml:-1, :] - Hy[-n_cells_pml - 1:-2, :]))

                Ez[1:n_cells_pml, 1:-1] += e_coeffz[1:n_cells_pml, 1:-1] * psi_Ezx[1:n_cells_pml, 1:-1, 0]
                Ez[-n_cells_pml:-1, 1:-1] += e_coeffz[-n_cells_pml:-1, 1:-1] * psi_Ezx[1:n_cells_pml, 1:-1, 1]

                # Ez PML along y-direction
                psi_Ezy[:, 1:n_cells_pml, 0] = (be_y[:, 1:] * psi_Ezy[:, 1:n_cells_pml, 0] +
                                                ce_y[:, 1:] * Inv_dy * (Hx[:, 1:n_cells_pml] - Hx[:, :n_cells_pml - 1]))
                psi_Ezy[:, 1:n_cells_pml, 1] = (be_y_f[:, :-1] * psi_Ezy[:, 1:n_cells_pml, 1] +
                                                ce_y_f[:, :-1] * Inv_dy * (
                                                            Hx[:, -n_cells_pml:-1] - Hx[:, -n_cells_pml - 1:-2]))

                Ez[1:-1, 1:n_cells_pml] -= e_coeffz[1:-1, 1:n_cells_pml, ] * psi_Ezy[1:-1, 1:n_cells_pml, 0]
                Ez[1:-1, -n_cells_pml:-1] -= e_coeffz[1:-1, -n_cells_pml:-1] * psi_Ezy[1:-1, 1:n_cells_pml, 1]

                # add electric field source
                for s in self.source:
                    source_dir = s["direction"]
                    imin = s["imin"]
                    imax = s["imax"]
                    jmin = s["jmin"]
                    jmax = s["jmax"]
                    if source_dir == "+y":
                        Ez[imin:imax, jmin] = Ez[imin:imax, jmin] + s["Hx"] * s["signal"]((n + 1) * dt) * e_coeffz[
                                                                                                          imin:imax,
                                                                                                          jmin] / dy
                    elif source_dir == "-y":
                        Ez[imin:imax, jmin] = Ez[imin:imax, jmin] - s["Hx"] * s["signal"]((n + 1) * dt) * e_coeffz[
                                                                                                          imin:imax,
                                                                                                          jmin] / dy
                    elif source_dir == "+x":
                        Ez[imin, jmin:jmax] = Ez[imin, jmin:jmax] - s["Hy"] * s["signal"]((n + 1) * dt) * e_coeffz[imin,
                                                                                                          jmin:jmax] / dx
                    elif source_dir == "-x":
                        Ez[imin, jmin:jmax] = Ez[imin, jmin:jmax] + s["Hy"] * s["signal"]((n + 1) * dt) * e_coeffz[imin,
                                                                                                          jmin:jmax] / dx

                for d, dft in enumerate(self.dft_region):
                    for ww, w in enumerate(dft["omegas"]):
                        exp_term = np.exp(-1j * time * w)
                        dft["Ez"][ww, :, :] += Ez[dft["iminz"]:dft["imaxz"], dft["jminz"]:dft["jmaxz"]] * exp_term

                for fr in self.flux_region:
                    for ww, w in enumerate(fr["omegas"]):
                        exp_e = np.exp(-1j * time * w)
                        exp_h = np.exp(-1j * time_ * w)
                        if fr["direction_"] == "y":
                            fr["Ex"][ww, :] += Ex[fr["imin"]:fr["imax"], fr["jmin"]] * exp_e
                            fr["Ez"][ww, :] += Ez[fr["imin"]:fr["imax"], fr["jmin"]] * exp_e
                            fr["Hx"][ww, :] += 0.5 * (Hx[fr["imin"]:fr["imax"], fr["jmin"]]
                                                      + Hx[fr["imin"]:fr["imax"], fr["jmin"] - 1]) * exp_h
                            fr["Hz"][ww, :] += 0.5 * (Hz[fr["imin"]:fr["imax"], fr["jmin"]]
                                                      + Hz[fr["imin"]:fr["imax"], fr["jmin"] - 1]) * exp_h
                        if fr["direction_"] == "x":
                            fr["Ey"][ww, :] += Ey[fr["imin"], fr["jmin"]:fr["jmax"]] * exp_e
                            fr["Ez"][ww, :] += Ez[fr["imin"], fr["jmin"]:fr["jmax"]] * exp_e
                            fr["Hy"][ww, :] += 0.5 * (Hy[fr["imin"], fr["jmin"]:fr["jmax"]]
                                                      + Hy[fr["imin"] - 1, fr["jmin"]:fr["jmax"]]) * exp_h
                            fr["Hz"][ww, :] += 0.5 * (Hz[fr["imin"], fr["jmin"]:fr["jmax"]]
                                                      + Hz[fr["imin"] - 1, fr["jmin"]:fr["jmax"]]) * exp_h

                mf = np.sqrt(np.max(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2))
                if mf > max_field:
                    max_field = mf

                # shutoff simulation early
                if time > min_simulation_time:
                    if mf < self.cutoff * max_field:
                        print("Electric field below threshold, breaking")
                        break

                if n % 100 == 0:
                    print(n, np.max(np.abs(Ex)), np.max(np.abs(Ey)), np.max(np.abs(Ez)), max_field, mf / max_field)

                if n % self.movie_update == 0:
                    E_movie.append(Ex ** 2 + Ey ** 2 + Ez ** 2)
        else:
            # FDTD algorithm
            for n in range(N_time_steps):
                time = (n + 1) * dt
                time_ = (n + 0.5) * dt

                Hz_prev = Hz.copy()
                Ex_prev = Ex.copy()
                Ey_prev = Ey.copy()

                # begin_h = time_mod.time()
                # update magnetic field at n+1/2
                Hz[:-1, :-1] = (h_coeff_1 * Hz_prev[:-1, :-1]
                                + h_coeff_y[:, :-1] * (Ex[:-1, 1:] - Ex[:-1, :-1])
                                - h_coeff_x[:-1, :] * (Ey[1:, :-1] - Ey[:-1, :-1]))

                # Hz PML along x-direction
                psi_Hzx[:n_cells_pml - 1, :, 0] = (psi_Hzx[:n_cells_pml - 1, :, 0] * bh_x[:-1, :]
                                                   + (Ey[1:n_cells_pml, :] - Ey[:n_cells_pml - 1, :]) * ch_x[:-1,
                                                                                                        :] * Inv_dx)

                psi_Hzx[:n_cells_pml - 1, :, 1] = (psi_Hzx[:n_cells_pml - 1, :, 1] * bh_x_f[1:, :]
                                                   + (Ey[-n_cells_pml + 1:, :] - Ey[-n_cells_pml:-1, :]) * ch_x_f[1:,
                                                                                                           :] * Inv_dx)

                Hz[:n_cells_pml - 1, :-1] -= h_coeff * psi_Hzx[:n_cells_pml - 1, :-1, 0]
                Hz[-n_cells_pml:-1, :-1] -= h_coeff * psi_Hzx[:n_cells_pml - 1, :-1, 1]

                # Hz PML along y-direction
                psi_Hzy[:, :n_cells_pml - 1, 0] = (psi_Hzy[:, :n_cells_pml - 1, 0] * bh_y[:, :-1]
                                                   + (Ex[:, 1:n_cells_pml] - Ex[:, :n_cells_pml - 1]) * ch_y[:,
                                                                                                        :-1] * Inv_dy)

                psi_Hzy[:, :n_cells_pml - 1, 1] = (psi_Hzy[:, :n_cells_pml - 1, 1] * bh_y_f[:, 1:]
                                                   + ch_y_f[:, 1:, ] * Inv_dy * (
                                                               Ex[:, -n_cells_pml + 1:] - Ex[:, -n_cells_pml:-1]))

                Hz[:-1, :n_cells_pml - 1] += h_coeff * psi_Hzy[:-1, :n_cells_pml - 1, 0]
                Hz[:-1, -n_cells_pml:-1] += h_coeff * psi_Hzy[:-1, :n_cells_pml - 1, 1]
                end_h = time_mod.time()
                # hupdate_time.append(end_h - begin_h)
                # add magnetic field source
                for s in self.source:
                    source_dir = s["direction"]
                    imin = s["imin"]
                    imax = s["imax"]
                    jmin = s["jmin"]
                    jmax = s["jmax"]
                    t_offset = s["t_offset"]
                    Z = 1  # s["Z"]
                    # print(s["Hz"].shape, Hz[imin:imax,jmin-1].shape, (s["Hz"] - Hz[imin:imax,jmin-1]).shape)
                    if source_dir == "+y":
                        Hz[imin:imax, jmin - 1] = Hz[imin:imax, jmin - 1] - h_coeff * s["Ex"] * s["signal"](
                            (n + 0.5) * dt - t_offset) / dy
                    elif source_dir == "-y":
                        Hz[imin:imax, jmin - 1] = Hz[imin:imax, jmin - 1] + h_coeff * s["Ex"] * s["signal"](
                            (n + 0.5) * dt + t_offset) / dy
                    elif source_dir == "-x":
                        Hz[imin - 1, jmin:jmax] = Hz[imin - 1, jmin:jmax] - h_coeff * s["Ey"] * s["signal"](
                            (n + 0.5) * dt + t_offset) / dx
                    elif source_dir == "+x":
                        Hz[imin - 1, jmin:jmax] = Hz[imin - 1, jmin:jmax] + h_coeff * s["Ey"] * s["signal"](
                            (n + 0.5) * dt - t_offset) / dx

                # update electric field at n+1
                Ex[:-1, 1:-1] = (e_coeff_1x[:-1, 1:-1] * Ex_prev[:-1, 1:-1]
                                 + e_coeff_xy[:-1, 1:-1] * (Hz[:-1, 1:-1] - Hz[:-1, :-2]))

                Ey[1:-1, :-1] = (e_coeff_1y[1:-1, :-1] * Ey_prev[1:-1, :-1]
                                 - e_coeff_yx[1:-1, :-1] * (Hz[1:-1, :-1] - Hz[:-2, :-1]))
                # Ex PML along y-direction
                psi_Exy[:, 1:n_cells_pml, 0] = (be_y[:, 1:] * psi_Exy[:, 1:n_cells_pml, 0]
                                                + ce_y[:, 1:] * Inv_dy * (
                                                            Hz[:, 1:n_cells_pml] - Hz[:, :n_cells_pml - 1]))

                psi_Exy[:, 1:n_cells_pml, 1] = (be_y_f[:, :-1] * psi_Exy[:, 1:n_cells_pml, 1]
                                                + ce_y_f[:, :-1] * Inv_dy * (
                                                            Hz[:, -n_cells_pml:-1] - Hz[:, -n_cells_pml - 1:-2]))

                Ex[:-1, 1:n_cells_pml] += e_coeffx[:-1, 1:n_cells_pml] * psi_Exy[:-1, 1:n_cells_pml, 0]
                Ex[:-1, -n_cells_pml:-1] += e_coeffx[:-1, -n_cells_pml:-1] * psi_Exy[:-1, 1:n_cells_pml, 1]

                # Ey PML along x-direction
                psi_Eyx[1:n_cells_pml, :, 0] = (be_x[1:, :] * psi_Eyx[1:n_cells_pml, :, 0] +
                                                Inv_dx * ce_x[1:, :] * (Hz[1:n_cells_pml, :] - Hz[:n_cells_pml - 1, :]))
                psi_Eyx[1:n_cells_pml, :, 1] = (be_x_f[:-1, :] * psi_Eyx[1:n_cells_pml, :, 1]
                                                + Inv_dx * ce_x_f[:-1, :] * (
                                                            Hz[-n_cells_pml:-1, :] - Hz[-n_cells_pml - 1:-2, :]))

                Ey[1:n_cells_pml, :-1] -= e_coeffy[1:n_cells_pml, :-1] * psi_Eyx[1:n_cells_pml, :-1, 0]
                Ey[-n_cells_pml:-1, :-1] -= e_coeffy[-n_cells_pml:-1, :-1] * psi_Eyx[1:n_cells_pml, :-1, 1]

                # add electric field source
                for s in self.source:
                    source_dir = s["direction"]
                    imin = s["imin"]
                    imax = s["imax"]
                    jmin = s["jmin"]
                    jmax = s["jmax"]
                    if source_dir == "+y":
                        Ex[imin:imax, jmin] = Ex[imin:imax, jmin] - s["Hz"] * s["signal"]((n + 1) * dt) * e_coeffx[
                                                                                                          imin:imax,
                                                                                                          jmin] / dy
                    elif source_dir == "-y":
                        Ex[imin:imax, jmin] = Ex[imin:imax, jmin] + s["Hz"] * s["signal"]((n + 1) * dt) * e_coeffx[
                                                                                                          imin:imax,
                                                                                                          jmin] / dy
                    elif source_dir == "+x":
                        Ey[imin, jmin:jmax] = Ey[imin, jmin:jmax] + s["Hz"] * s["signal"]((n + 1) * dt) * e_coeffy[imin,
                                                                                                          jmin:jmax] / dx
                    elif source_dir == "-x":
                        Ey[imin, jmin:jmax] = Ey[imin, jmin:jmax] - s["Hz"] * s["signal"]((n + 1) * dt) * e_coeffy[imin,
                                                                                                          jmin:jmax] / dx

                for d, dft in enumerate(self.dft_region):
                    for ww, w in enumerate(dft["omegas"]):
                        exp_term = np.exp(-1j * time * w)
                        dft["Ex"][ww, :, :] += Ex[dft["iminx"]:dft["imaxx"], dft["jminx"]:dft["jmaxx"]] * exp_term
                        dft["Ey"][ww, :, :] += Ey[dft["iminy"]:dft["imaxy"], dft["jminy"]:dft["jmaxy"]] * exp_term

                for fr in self.flux_region:
                    for ww, w in enumerate(fr["omegas"]):
                        exp_e = np.exp(-1j * time * w)
                        exp_h = np.exp(-1j * time_ * w)
                        if fr["direction_"] == "y":
                            fr["Ex"][ww, :] += Ex[fr["imin"]:fr["imax"], fr["jmin"]] * exp_e
                            fr["Ez"][ww, :] += Ez[fr["imin"]:fr["imax"], fr["jmin"]] * exp_e
                            fr["Hx"][ww, :] += 0.5 * (Hx[fr["imin"]:fr["imax"], fr["jmin"]]
                                                      + Hx[fr["imin"]:fr["imax"], fr["jmin"] - 1]) * exp_h
                            fr["Hz"][ww, :] += 0.5 * (Hz[fr["imin"]:fr["imax"], fr["jmin"]]
                                                      + Hz[fr["imin"]:fr["imax"], fr["jmin"] - 1]) * exp_h
                        if fr["direction_"] == "x":
                            fr["Ey"][ww, :] += Ey[fr["imin"], fr["jmin"]:fr["jmax"]] * exp_e
                            fr["Ez"][ww, :] += Ez[fr["imin"], fr["jmin"]:fr["jmax"]] * exp_e
                            fr["Hy"][ww, :] += 0.5 * (Hy[fr["imin"], fr["jmin"]:fr["jmax"]]
                                                      + Hy[fr["imin"] - 1, fr["jmin"]:fr["jmax"]]) * exp_h
                            fr["Hz"][ww, :] += 0.5 * (Hz[fr["imin"], fr["jmin"]:fr["jmax"]]
                                                      + Hz[fr["imin"] - 1, fr["jmin"]:fr["jmax"]]) * exp_h

                mf = np.sqrt(np.max(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2))
                if mf > max_field:
                    max_field = mf

                # shutoff simulation early
                if time > min_simulation_time:
                    if mf < self.cutoff * max_field:
                        print("Electric field below threshold, breaking")
                        break

                if n % 100 == 0:
                    print(n, np.max(np.abs(Ex)), np.max(np.abs(Ey)), np.max(np.abs(Ez)), max_field, mf / max_field)

                if n % self.movie_update == 0:
                    E_movie.append(Ex ** 2 + Ey ** 2 + Ez ** 2)

        end = time_mod.time()
        print(f"Simulation took {end - begin} seconds")
        print(f"Simulation took {(end - begin) / n} seconds per step")
        # hupdate_time = hupdate_time[2:]
        # print(f"H-update mean: {np.mean(hupdate_time)}")
        # print(f"H-update max: {np.max(hupdate_time)}")
        # print(f"H-update min: {np.min(hupdate_time)}")
        # plt.plot(hupdate_time)
        # plt.show()

        for fr in self.flux_region:
            fr["pulses"] = []
            for so, s in enumerate(self.source):
                for so, s in enumerate(self.source):
                    pulse = s["signal"](np.arange(N_time_steps) * dt)
                    pulsew = Discrete_Fourier_Transform(pulse, np.arange(N_time_steps) * dt, fr["omegas"])
                    fr["pulses"].append(pulsew.copy())

            if "mode" in fr:
                imin = fr["imin"]
                imax = fr["imax"]
                jmin = fr["jmin"]
                jmax = fr["jmax"]

                if fr["size"][0] == 0:
                    epsyy_ = epszz[imin:imax, jmin:jmax].transpose()
                    epszz_ = epsxx[imin:imax, jmin:jmax].transpose()
                    epsxx_ = epsyy[imin:imax, jmin:jmax].transpose()
                else:
                    epsxx_ = epsxx[imin:imax, jmin:jmax]  # .transpose()
                    epsyy_ = epszz[imin:imax, jmin:jmax]  # .transpose()
                    epszz_ = epsyy[imin:imax, jmin:jmax]  # .transpose()

                E1s = []
                E2s = []
                H1s = []
                H2s = []
                ns = []

                if "mode Ey" not in fr:
                    for w, wl in enumerate(fr["wavelengths"]):
                        n, E1, E2, H1, H2 = Mode_Solver(epsxx_, epsyy_, epszz_, dx, dy, wl, fr["mode"])
                        E1s.append(E1)
                        E2s.append(E2)
                        H1s.append(H1)
                        H2s.append(H2)
                        ns.append(n)
                        print(n)
                    fr["mode index"] = np.array(ns)
                    if fr["direction"] == "+x" or fr["direction"] == "-x":
                        fr["mode Ey"] = np.array(E1s)[:, :, 0]
                        fr["mode Ez"] = np.array(E2s)[:, :, 0]
                        fr["mode Hy"] = np.array(H1s)[:, :, 0]
                        fr["mode Hz"] = np.array(H2s)[:, :, 0]
                    if fr["direction"] == "+y" or fr["direction"] == "-y":
                        fr["mode Ex"] = np.array(E1s)[:, :, 0]
                        fr["mode Ez"] = np.array(E2s)[:, :, 0]
                        fr["mode Hx"] = -np.array(H1s)[:, :, 0]
                        fr["mode Hz"] = -np.array(H2s)[:, :, 0]

                # plt.plot(np.abs(fr["Hz"][5,40:60]) / np.abs(fr["mode Hz"][5,40:60]))
                # plt.show()

                if fr['direction'] == "+x" or fr["direction"] == "-x":
                    fr['mode amplitude +x'] = GetModeOverlap(fr['Ey'], fr['Ez'], fr['Hy'], fr['Hz'],
                                                             fr['mode Ey'], fr['mode Ez'], fr['mode Hy'],
                                                             fr['mode Hz'])
                    fr['mode amplitude -x'] = GetModeOverlap(fr['Ey'], fr['Ez'], fr['Hy'], fr['Hz'],
                                                             fr['mode Ey'], fr['mode Ez'], -fr['mode Hy'],
                                                             -fr['mode Hz'])
                    fr['Power norm +x'] = GetModeOverlap(fr['mode Ey'], fr['mode Ez'], fr['mode Hy'], fr['mode Hz'],
                                                         fr['mode Ey'], fr['mode Ez'], fr['mode Hy'],
                                                         fr['mode Hz'])
                    fr['Power norm -x'] = GetModeOverlap(fr['mode Ey'], fr['mode Ez'], fr['mode Hy'], fr['mode Hz'],
                                                         fr['mode Ey'], fr['mode Ez'], -fr['mode Hy'],
                                                         -fr['mode Hz'])
                if fr['direction'] == "+y" or fr["direction"] == "-y":
                    # plt.plot(np.abs(fr['Hz'][5,40:60]) / np.abs(fr['mode Hz'][5,40:60]))
                    # plt.show()
                    fr['mode amplitude +y'] = GetModeOverlap(fr['Ez'], fr['Ex'], fr['Hz'], fr['Hx'],
                                                             fr['mode Ez'], fr['mode Ex'], fr['mode Hz'],
                                                             fr['mode Hx'])
                    fr['mode amplitude -y'] = GetModeOverlap(fr['Ez'], fr['Ex'], fr['Hz'], fr['Hx'],
                                                             fr['mode Ez'], fr['mode Ex'], -fr['mode Hz'],
                                                             -fr['mode Hx'])
                    fr['Power norm +y'] = GetModeOverlap(fr['mode Ez'], fr['mode Ex'], fr['mode Hz'], fr['mode Hx'],
                                                         fr['mode Ez'], fr['mode Ex'], fr['mode Hz'],
                                                         fr['mode Hx'])
                    fr['Power norm -y'] = GetModeOverlap(fr['mode Ez'], fr['mode Ex'], fr['mode Hz'], fr['mode Hx'],
                                                         fr['mode Ez'], fr['mode Ex'], -fr['mode Hz'],
                                                         -fr['mode Hx'])
        if len(design_grid) > 0:
            return E_movie, self.flux_region, self.dft_region, design_grid
        else:
            return E_movie, self.flux_region, self.dft_region

#
# def FDTD_2D(simulation_size, step_size,
#             geometry = [],
#             source = [],
#             simulation_time = 2000e-15,
#             cutoff = 1e-4 ,
#             dft_region = [],
#             flux_region = [],
#             courant = 0.5,
#             movie_update = 10,
#             n_cells_pml = 10,
#             complex_sim = False,
#             TE = False,
#             staircasing = True
#             ):
#
#     if complex_sim:
#         type = complex_type
#     else:
#         type = real_type
#
#     dx = step_size
#     dy = step_size
#     Sx, Sy = simulation_size
#     x_axis = np.arange(-Sx/2,Sx/2,dx)
#     y_axis = np.arange(-Sy/2,Sy/2,dy)
#     x_axis_sub = np.arange(-Sx/2,Sx/2,dx/2)
#     y_axis_sub = np.arange(-Sy/2,Sy/2,dy/2)
#     # print(x_axis,x_axis_sub)
#     x_axis_sub_ = np.arange(len(x_axis_sub)) * dx / 2 - Sx / 2
#     # print(Sx / 2, dx / 2)
#     x_axis_x = x_axis_sub[1::2]
#     y_axis_x = y_axis_sub[::2]
#     Xx, Yx = np.meshgrid(x_axis_x,y_axis_x,indexing='ij')
#     x_axis_y = x_axis_sub[::2]
#     y_axis_y = y_axis_sub[1::2]
#     Xy, Yy = np.meshgrid(x_axis_y,y_axis_y,indexing='ij')
#     x_axis_z = x_axis_sub[::2]
#     y_axis_z = y_axis_sub[::2]
#     Xz, Yz = np.meshgrid(x_axis_z,y_axis_z,indexing='ij')
#
#     X_axis, Y_axis = np.meshgrid(x_axis,y_axis,indexing='ij')
#     X_axis_sub, Y_axis_sub = np.meshgrid(x_axis_sub,y_axis_sub,indexing='ij')
#     Nx = x_axis.shape[0]
#     Ny = y_axis.shape[0]
#     Domain_shape = (Nx,Ny)
#     Nxs = x_axis_sub.shape[0]
#     Nys = y_axis_sub.shape[0]
#     Domain_shape_sub = (Nxs,Nys)
#     Inv_dx = 1./dx
#     Inv_dy = 1./dy
#     dt = courant * min([dx,dy]) / c0
#
#     epsxx = np.ones(Domain_shape)
#     epsyy = np.ones(Domain_shape)
#     epszz = np.ones(Domain_shape)
#     eps = np.ones(Domain_shape)
#     eps_sub = np.ones(Domain_shape_sub)
#     sigma_exx = np.zeros(Domain_shape)
#     sigma_eyy = np.zeros(Domain_shape)
#     sigma_ezz = np.zeros(Domain_shape)
#     sigma_e = np.zeros(Domain_shape)
#     sigma_h = 0
#
#     design_grid = []
#
#     for g in geometry:
#         if g["type"] == "rectangle":
#             ce = g["center"]
#             si = g["size"]
#             if staircasing:
#                 dx_ = 0.25 * dx
#                 dy_ = 0.25 * dy
#                 rectx = ((np.abs(X_axis + dx_ - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis + dy_ - ce[1]) <= si[1]/2))
#                 epsxx[rectx] = g["refractive index"] ** 2
#                 recty = ((np.abs(X_axis + dx_ - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis + dy_ - ce[1]) <= si[1]/2))
#                 epsyy[recty] = g["refractive index"] ** 2
#                 rectz = ((np.abs(X_axis + dx_ - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis + dy_ - ce[1]) <= si[1]/2))
#                 epszz[rectz] = g["refractive index"] ** 2
#
#                 rect = ((np.abs(X_axis - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis - ce[1]) <= si[1]/2))
#                 eps[rect] = g["refractive index"] ** 2
#
#                 rects = ((np.abs(X_axis_sub - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis_sub - ce[1]) <= si[1]/2))
#                 eps_sub[rects] = g["refractive index"] ** 2
#             else:
#                 print("here")
#                 rectx = ((np.abs(Xx - ce[0]) <= si[0]/2)
#                          & (np.abs(Yx  - ce[1]) <= si[1]/2))
#                 epsxx[rectx] = g["refractive index"] ** 2
#
#                 recty = ((np.abs(Xy - ce[0]) <= si[0]/2)
#                          & (np.abs(Yy - ce[1]) <= si[1]/2))
#                 epsyy[recty] = g["refractive index"] ** 2
#                 rectz = ((np.abs(Xz - ce[0]) <= si[0]/2)
#                          & (np.abs(Yz - ce[1]) <= si[1]/2))
#                 epszz[rectz] = g["refractive index"] ** 2
#
#                 rect = ((np.abs(X_axis - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis - ce[1]) <= si[1]/2))
#                 eps[rect] = g["refractive index"] ** 2
#
#                 rects = ((np.abs(X_axis_sub - ce[0]) <= si[0]/2)
#                          & (np.abs(Y_axis_sub - ce[1]) <= si[1]/2))
#                 eps_sub[rects] = g["refractive index"] ** 2
#
#         if g["type"] == "design":
#             ce = g["center"]
#             si = g["size"]
#
#             if staircasing:
#                 dx_ = 0.25 * dx
#                 dy_ = 0.25 * dy
#                 nx, ny = g["grid"].shape
#                 imin = np.argmin(np.abs(x_axis + dx_ - ce[0] + si[0]/2))
#                 imax = imin + int(round(si[0]/dx))
#                 nx_ = int(np.floor((imax - imin)/nx))
#                 i_list = [0]
#                 remainder = (imax - imin) - nx * nx_
#                 print(imax - imin, nx_ * nx, remainder)
#                 for i in range(nx):
#                     if i < remainder:
#                         i_list.append(i_list[-1] + nx_ + 1)
#                     else:
#                         i_list.append(i_list[-1] + nx_)
#
#                 jmin = np.argmin(np.abs(y_axis + dy_ - ce[1] + si[1]/2))
#                 jmax = jmin + int(round(si[1]/dy))
#                 ny_ = int(np.floor((jmax - jmin)/ny))
#                 j_list = [0]
#                 remainder = (jmax - jmin) - ny * ny_
#                 print(jmax - jmin, ny_ * ny, remainder)
#                 for j in range(ny):
#                     if j < remainder:
#                         j_list.append(j_list[-1] + ny_ + 1)
#                     else:
#                         j_list.append(j_list[-1] + ny_)
#                 g["i_list"] = copy.deepcopy(i_list)
#                 g["j_list"] = copy.deepcopy(j_list)
#                 i_list = [il + imin for il in i_list]
#                 j_list = [jl + jmin for jl in j_list]
#                 if "loss" in g:
#                     sigma_exx[imin:imax,jmin:jmax] = g["loss"]
#                     sigma_eyy[imin:imax,jmin:jmax] = g["loss"]
#                     sigma_ezz[imin:imax,jmin:jmax] = g["loss"]
#
#                 for i in range(1,len(i_list)):
#                     for j in range(1,len(j_list)):
#                         epsxx[i_list[i-1]:i_list[i],j_list[j-1]:j_list[j]] = density_to_perm(g["grid"][i-1,j-1], g["ri min"] ** 2,g["ri max"] ** 2)
#                         epsyy[i_list[i-1]:i_list[i],j_list[j-1]:j_list[j]] = density_to_perm(g["grid"][i-1,j-1], g["ri min"] ** 2,g["ri max"] ** 2)
#                         epszz[i_list[i-1]:i_list[i],j_list[j-1]:j_list[j]] = density_to_perm(g["grid"][i-1,j-1], g["ri min"] ** 2,g["ri max"] ** 2)
#                 design_grid += [g]
#             else:
#                 nx, ny = g["grid"].shape
#                 imin = 0
#                 imax = 10000
#                 jmin = 0
#                 jmax = 10000
#                 for i in range(1,len(x_axis_sub)-2):
#                     if x_axis_sub[i] - ce[0] >= -si[0] / 2 and x_axis_sub[i-1] - ce[0] < -si[0] / 2:
#                         imin = i
#
#                     if x_axis_sub[i] - ce[0] <= si[0] / 2 and x_axis_sub[i+1] - ce[0] > si[0] / 2:
#                         imax = i
#                         break
#                 for i in range(1,len(y_axis_sub)-2):
#                     if y_axis_sub[i] - ce[1] >= -si[1] / 2 and y_axis_sub[i-1] - ce[1] < -si[1] / 2:
#                         jmin = i
#                     if y_axis_sub[i] - ce[1] <= si[1] / 2 and y_axis_sub[i+1] - ce[1] > si[1] / 2:
#                         jmax = i
#                         break
#
#                 print(imin, imax, jmin, jmax)
#                 x_g = x_axis_sub[imin:imax+1]
#                 y_g = y_axis_sub[jmin:jmax+1]
#                 Xg,Yg = np.meshgrid(x_g,y_g, indexing = 'ij')
#
#                 grid = g["grid"]
#                 eps_ = density_to_perm(grid, g["ri min"] ** 2, g["ri max"] ** 2).reshape(nx, ny)
#                 interp = RegularGridInterpolator((x_g,y_g),eps_, method = "nearest",fill_value=None)
#                 print(np.min(x_g), np.max(x_g), np.min(y_g), np.max(y_g), y_axis_sub[jmin-1])
#                 eps_sub[imin:imax+1,jmin:jmax+1] = interp(np.array([Xg.flatten(),Yg.flatten()]).transpose()).reshape(len(x_g),len(y_g))
#
#                 for i in range(1,len(x_axis)-2):
#                     if x_axis_x[i] - ce[0] >= -si[0] / 2 and x_axis_x[i-1] - ce[0] < -si[0] / 2:
#                         imin = i
#                     if x_axis_x[i] - ce[0] <= si[0] / 2 and x_axis_x[i+1] - ce[0] > si[0] / 2:
#                         imax = i
#                         break
#                 for i in range(1,len(y_axis)-2):
#                     if y_axis_x[i] - ce[1] >= -si[1] / 2 and y_axis_x[i-1] - ce[1] < -si[1] / 2:
#                         jmin = i
#                     if y_axis_x[i] - ce[1] <= si[1] / 2 and y_axis_x[i+1]  - ce[1] > si[1] / 2:
#                         jmax = i
#                         break
#                 x_gx = x_axis_x[imin:imax+1]
#                 y_gx = y_axis_x[jmin:jmax+1]
#                 Xgx,Ygx = np.meshgrid(x_gx,y_gx, indexing = 'ij')
#                 print(np.min(x_gx), np.max(x_gx), np.min(y_gx), np.max(y_gx))
#                 def gridderx(grid):
#                     interp = RegularGridInterpolator((x_g,y_g),grid,method = "nearest",fill_value=None,
#                                                      bounds_error=False)
#                     inputs = np.array([Xgx.flatten(),Ygx.flatten()]).transpose()
#                     vals = interp(inputs)
#                     return vals
#
#                 valsx, jacx = vjp(gridderx,eps_)
#                 print("valsx")
#                 # valsx = gridderx(eps_)
#                 # jacx = jacobian(gridderx,0)(eps_)
#                 epsxx[imin:imax+1,jmin:jmax+1] = valsx.reshape(len(x_gx),len(y_gx))
#                 g["iminx"] = imin
#                 g["imaxx"] = imax + 1
#                 g["jminx"] = jmin
#                 g["jmaxx"] = jmax + 1
#                 g["jacx"] = jacx
#                 print((len(x_gx),len(y_gx)),valsx.shape)
#
#                 for i in range(1,len(x_axis)-2):
#                     if x_axis_y[i] - ce[0] >= -si[0] / 2 and x_axis_y[i-1] - ce[0] < -si[0] / 2:
#                         imin = i
#                     if x_axis_y[i]  - ce[0] <= si[0] / 2 and x_axis_y[i+1] - ce[0] > si[0] / 2:
#                         imax = i
#                         break
#                 for i in range(1,len(y_axis)-2):
#                     if y_axis_y[i] - ce[1] >= -si[1] / 2 and y_axis_y[i-1] - ce[1] < -si[1] / 2:
#                         jmin = i
#                     if y_axis_y[i] - ce[1] <= si[1] / 2 and y_axis_y[i+1] - ce[1] > si[1] / 2:
#                         jmax = i
#                         break
#                 x_gy = x_axis_y[imin:imax+1]
#                 y_gy = y_axis_y[jmin:jmax+1]
#                 Xgy,Ygy = np.meshgrid(x_gy,y_gy, indexing = 'ij')
#
#                 def griddery(grid):
#                     interp = RegularGridInterpolator((x_g, y_g), grid, method="nearest", fill_value=None,
#                                                      bounds_error=False)
#                     inputs = np.array([Xgy.flatten(), Ygy.flatten()]).transpose()
#                     vals = interp(inputs)
#                     return vals
#                 valsy, jacy = vjp(griddery,eps_)#
#                 # valsy = griddery(eps_)
#                 epsyy[imin:imax+1,jmin:jmax+1] = valsy.reshape(len(x_gy),len(y_gy))
#                 # jacy = jacobian(griddery,0)(eps_)
#
#                 print((len(x_gy),len(y_gy)))
#                 g["iminy"] = imin
#                 g["imaxy"] = imax + 1
#                 g["jminy"] = jmin
#                 g["jmaxy"] = jmax+ 1
#                 g["jacy"] = jacy
#
#                 for i in range(1,len(x_axis)-2):
#                     if x_axis_z[i] - ce[0] >= -si[0] / 2 and x_axis_z[i-1] - ce[0] < -si[0] / 2:
#                         imin = i
#                     if x_axis_z[i]  - ce[0] <= si[0] / 2 and x_axis_z[i+1] - ce[0] > si[0] / 2:
#                         imax = i
#                         break
#                 for i in range(1,len(y_axis)-2):
#                     if y_axis_z[i] - ce[1] >= -si[1] / 2 and y_axis_z[i-1]- ce[1] < -si[1] / 2:
#                         jmin = i
#                     if y_axis_z[i] - ce[1] <= si[1] / 2 and y_axis_z[i+1] - ce[1] > si[1] / 2:
#                         jmax = i
#                         break
#                 x_gz = x_axis_z[imin:imax+1]
#                 y_gz = y_axis_z[jmin:jmax+1]
#                 print((len(x_gz),len(y_gz)))
#                 Xgz,Ygz = np.meshgrid(x_gz,y_gz, indexing = 'ij')
#                 def gridderz(grid):
#                     interp = RegularGridInterpolator((x_g, y_g), grid, method="nearest", fill_value=None,
#                                                      bounds_error=False)
#                     inputs = np.array([Xgz.flatten(), Ygz.flatten()]).transpose()
#                     vals = interp(inputs)
#                     return vals
#
#                 valsz, jacz = vjp(gridderz,eps_)
#                 # valsz = gridderz(eps_)
#                 epszz[imin:imax+1,jmin:jmax+1] = valsz.reshape(len(x_gz),len(y_gz))
#                 # jacz = jacobian(gridderz,0)(eps_)
#                 g["iminz"] = imin
#                 g["imaxz"] = imax + 1
#                 g["jminz"] = jmin
#                 g["jmaxz"] = jmax + 1
#                 g["jacz"] = jacz
#
#                 if "loss" in g:
#                     sigma_exx[g["iminx"]:g["imaxx"],g["jminx"]:g["imaxx"]] = g["loss"]
#                     sigma_eyy[g["iminy"]:g["imaxy"],g["jminy"]:g["imaxy"]] = g["loss"]
#                     sigma_ezz[g["iminz"]:g["imaxz"],g["jminz"]:g["imaxz"]] = g["loss"]
#
#                 design_grid += [g]
#
#     plt.contourf(x_axis_sub * 1e6,y_axis_sub * 1e6,eps_sub.transpose(),200)
#     plt.xlabel("x [um]")
#     plt.ylabel("y [um]")
#     plt.savefig("sim.png")
#     plt.close()
#     plt.contourf(x_axis * 1e6,y_axis * 1e6,epsxx.transpose(),200)
#     plt.xlabel("x [um]")
#     plt.ylabel("y [um]")
#     plt.savefig("simxx.png")
#     plt.close()
#     plt.contourf(x_axis * 1e6,y_axis * 1e6,epsyy.transpose(),200)
#     plt.xlabel("x [um]")
#     plt.ylabel("y [um]")
#     plt.savefig("simyy.png")
#     plt.close()
#     plt.contourf(x_axis * 1e6,y_axis * 1e6,epszz.transpose(),200)
#     plt.xlabel("x [um]")
#     plt.ylabel("y [um]")
#     plt.savefig("simzz.png")
#     plt.close()
#
#     plt.imshow(epsxx - np.flip(epsxx, axis = 1))
#     plt.colorbar()
#     plt.savefig("simzxxflip.png")
#     plt.close()
#
#
#     for s in source:
#         if not "pulse width" in s:
#             pulse_width = 10e-15
#             s["pulse width"] = pulse_width
#         if not "amplitude" in s:
#             amplitude = 1.0
#             s["amplitude"] = amplitude
#         if not "phase" in s:
#             pulse_phase = 0.0
#             s["phase"] = pulse_phase
#         if not "delay" in s:
#             pulse_delay_fact = 4
#             s["delay"] = pulse_delay_fact * s["pulse width"]
#         if "pulse" not in s:
#             pulse = None
#             s["pulse"] = None
#         if s["pulse"] is None:
#             omega0 = 2 * np.pi * c0 / s["wavelength"]
#
#             signal = partial(gaussian, pulse_width=s["pulse width"],
#                              pulse_delay=s["delay"],
#                              omega0=omega0,
#                              phase=s["phase"],
#                              amplitude=1)
#
#             pulse = signal(np.arange(int(simulation_time/dt)) * dt)
#             pulsew = Discrete_Fourier_Transform(pulse, np.arange(int(simulation_time/dt)) * dt, np.array([omega0]))
#
#             s["signal"] = partial(gaussian, pulse_width=s["pulse width"],
#                              pulse_delay=s["delay"],
#                              omega0=omega0,
#                              phase=s["phase"],
#                              amplitude=s["amplitude"] / np.abs(pulsew)[0])
#
#             min_simulation_time = s["delay"] * 5 + s["pulse width"]
#         else:
#             if type == "complex128" or type == "complex64":
#                 signal = partial(complex_source, srcre=s["pulse"][0], srcim=s["pulse"][1])
#             else:
#                 signal = s["pulse"][0]
#             min_simulation_time = s["pulse"][2] * dt
#
#         if s["type"] == "mode" or s["type"] == "adjmode":
#             si = s["size"]
#             ce = s["center"]
#             imin = np.argmin(np.abs(x_axis - (ce[0] - si[0]/2)))
#             imax = np.argmin(np.abs(x_axis - (ce[0] + si[0]/2)))
#             jmin = np.argmin(np.abs(y_axis - (ce[1] - si[1]/2)))
#             jmax = np.argmin(np.abs(y_axis - (ce[1] + si[1]/2)))
#
#             if imin == imax:
#                 imax = imin + 1
#                 epsyy_ = epszz[imin:imax, jmin:jmax].transpose()
#                 epszz_ = epsxx[imin:imax, jmin:jmax].transpose()
#                 epsxx_ = epsyy[imin:imax, jmin:jmax].transpose()
#                 if "direction" not in s:
#                     s["direction"] = "+x"
#             elif jmin == jmax:
#                 jmax = jmax + 1
#                 epsxx_ = epsxx[imin:imax, jmin:jmax]#.transpose()
#                 epsyy_ = epszz[imin:imax, jmin:jmax]#.transpose()
#                 epszz_ = epsyy[imin:imax, jmin:jmax]#.transpose()
#                 if "direction" not in s:
#                     s["direction"] = "+y"
#             s["imin"] = imin
#             s["imax"] = imax
#             s["jmin"] = jmin
#             s["jmax"] = jmax
#
#             n, E1, E2, H1, H2 = Mode_Solver(epsxx_, epsyy_, epszz_, dx, dy, s["wavelength"], s["mode"])
#
#             if type.startswith("float"):
#                 E1 = E1.real
#                 E2 = E2.real
#                 H1 = H1.real
#                 H2 = H2.real
#
#             norm = 1.0
#             s["mode index"] = n
#             if s["direction"] == "+x":
#                 s["Ey"] = E1.copy()[:,0] / norm
#                 s["Ez"] = E2.copy()[:,0] / norm
#                 s["Hy"] = H1.copy()[:,0] / norm
#                 s["Hz"] = H2.copy()[:,0] / norm
#             elif s["direction"] == "-x":
#                 s["Ey"] = E1.copy()[:,0] / norm
#                 s["Ez"] = E2.copy()[:,0] / norm
#                 s["Hy"] = -H1.copy()[:,0] / norm
#                 s["Hz"] = -H2.copy()[:,0] / norm
#             elif s["direction"] == "+y":
#                 s["Ex"] = E1.copy()[:,0] / norm
#                 s["Ez"] = E2.copy()[:,0] / norm
#                 s["Hx"] = -H1.copy()[:,0] / norm
#                 s["Hz"] = -H2.copy()[:,0] / norm
#             elif s["direction"] == "-y":
#                 s["Ex"] = E1.copy()[:,0] / norm
#                 s["Ez"] = E2.copy()[:,0] / norm
#                 s["Hx"] = H1.copy()[:,0] / norm
#                 s["Hz"] = H2.copy()[:,0] / norm
#
#             s["t_offset"] = n * step_size / (2 * c0)
#             s["Z"] = imp0 / n
#
#     Ex = np.zeros(Domain_shape, dtype = type)
#     Ey = np.zeros(Domain_shape, dtype = type)
#     Ez = np.zeros(Domain_shape, dtype = type)
#     Hx = np.zeros(Domain_shape, dtype = type)
#     Hy = np.zeros(Domain_shape, dtype = type)
#     Hz = np.zeros(Domain_shape, dtype = type)
#
#     psi_Hxy = np.zeros((Nx,n_cells_pml,2), dtype = type)
#     psi_Hyx = np.zeros((n_cells_pml,Ny,2), dtype = type)
#     psi_Hzx = np.zeros((n_cells_pml,Ny,2), dtype = type)
#     psi_Hzy = np.zeros((Nx,n_cells_pml,2), dtype = type)
#     psi_Exy = np.zeros((Nx,n_cells_pml,2), dtype = type)
#     psi_Eyx = np.zeros((n_cells_pml,Ny,2), dtype = type)
#     psi_Ezx = np.zeros((n_cells_pml,Ny,2), dtype = type)
#     psi_Ezy = np.zeros((Nx,n_cells_pml,2), dtype = type)
#
#
#     simulation_time = max([simulation_time,min_simulation_time])
#     N_time_steps = int(simulation_time / dt)
#     print(f"there are {N_time_steps} FDTD time steps")
#
#
#     E_movie = []
#
#
#     # pulse = signal((np.arange(N_time_steps)+1)*dt)
#
#     # setup pmls
#     cpml_exp = 3  # should be 3 or 4
#     sigma_max = -(cpml_exp + 1) * 0.8 / (imp0 * dx)
#     alpha_max = 0.05
#     kappa_max = 5
#
#     # setup electric field PMLs
#     sigma = sigma_max * ((n_cells_pml - 1 - np.arange(n_cells_pml)) / n_cells_pml) ** cpml_exp
#     alpha = alpha_max * ((np.arange(n_cells_pml) - 1) / n_cells_pml) ** 1
#     kappa = 1 + (kappa_max - 1) * ((n_cells_pml - 1 - np.arange(n_cells_pml)) / n_cells_pml) ** cpml_exp
#
#     # setup electric field PMLs
#     sigmah = sigma_max * ((n_cells_pml - 1 - (np.arange(n_cells_pml) + 0.5)) / n_cells_pml) ** cpml_exp
#     alpha_h = alpha_max * ((np.arange(n_cells_pml) + 0.5 - 1) / n_cells_pml) ** 1
#     kappa_h = 1 + (kappa_max - 1) * ((n_cells_pml - 1 - (np.arange(n_cells_pml) + 0.5)) / n_cells_pml) ** cpml_exp
#
#     bh_x = np.exp((sigmah / kappa_h + alpha_h) * dt / eps0)
#     bh_x_f = np.flip(bh_x)
#     bh_y = np.exp((sigmah / kappa_h + alpha_h) * dt / eps0)
#     bh_y_f = np.flip(bh_y)
#     ch_x = sigmah * (bh_x - 1.0) / (sigmah + kappa_h * alpha_h) / kappa_h
#     ch_x_f = np.flip(ch_x)
#     ch_y = sigmah * (bh_y - 1.0) / (sigmah + kappa_h * alpha_h) / kappa_h
#     ch_y_f = np.flip(ch_y)
#
#     be_x = np.exp((sigma / kappa + alpha) * dt / eps0)
#     be_x_f = np.flip(be_x)
#     be_y = np.exp((sigma / kappa + alpha) * dt / eps0)
#     be_y_f = np.flip(be_y)
#     ce_x = sigma * (be_x - 1.0) / (sigma + kappa * alpha) / kappa
#     ce_x_f = np.flip(ce_x)
#     ce_y = sigma * (be_y - 1.0) / (sigma + kappa * alpha) / kappa
#     ce_y_f = np.flip(ce_y)
#
#     kappa_e_x = np.ones(eps.shape[0])
#     kappa_e_y = np.ones(eps.shape[1])
#     kappa_h_x = np.ones(eps.shape[0])
#     kappa_h_y = np.ones(eps.shape[1])
#
#     kappa_e_x[:len(kappa)] = kappa
#     kappa_e_x[-len(kappa):] = np.flip(kappa)
#     kappa_e_y[:len(kappa)] = kappa
#     kappa_e_y[-len(kappa):] = np.flip(kappa)
#     kappa_h_x[:len(kappa_h)] = kappa_h
#     kappa_h_x[-len(kappa_h) - 1:-1] = np.flip(kappa_h)
#     kappa_h_y[:len(kappa_h)] = kappa_h
#     kappa_h_y[-len(kappa_h) - 1:-1] = np.flip(kappa_h)
#
#     kappa_e_x = np.expand_dims(kappa_e_x, axis=1)
#     kappa_e_y = np.expand_dims(kappa_e_y, axis=0)
#     ce_x = np.expand_dims(ce_x, axis=1)
#     ce_y = np.expand_dims(ce_y, axis=0)
#     be_x = np.expand_dims(be_x, axis=1)
#     be_y = np.expand_dims(be_y, axis=0)
#     be_x_f = np.expand_dims(be_x_f, axis=1)
#     be_y_f = np.expand_dims(be_y_f, axis=0)
#     ce_x_f = np.expand_dims(ce_x_f, axis=1)
#     ce_y_f = np.expand_dims(ce_y_f, axis=0)
#
#     kappa_h_x = np.expand_dims(kappa_h_x, axis=1)
#     kappa_h_y = np.expand_dims(kappa_h_y, axis=0)
#     ch_x = np.expand_dims(ch_x, axis=1)
#     ch_y = np.expand_dims(ch_y, axis=0)
#     bh_x = np.expand_dims(bh_x, axis=1)
#     bh_y = np.expand_dims(bh_y, axis=0)
#     ch_x_f = np.expand_dims(ch_x_f, axis=1)
#     ch_y_f = np.expand_dims(ch_y_f, axis=0)
#     bh_x_f = np.expand_dims(bh_x_f, axis=1)
#     bh_y_f = np.expand_dims(bh_y_f, axis=0)
#
#     # Electric field update coefficients
#     denominatorx = eps0 * epsxx / dt + sigma_exx / 2
#     e_coeff_1x = (eps0 * epsxx / dt - sigma_exx / 2) / denominatorx
#     denominatory = eps0 * epsyy / dt + sigma_eyy / 2
#     e_coeff_1y = (eps0 * epsyy / dt - sigma_eyy / 2) / denominatory
#     denominatorz = eps0 * epszz / dt + sigma_ezz / 2
#     e_coeff_1z = (eps0 * epszz / dt - sigma_ezz / 2) / denominatorz
#
#     e_coeffx = 1.0 / denominatorx
#     e_coeffy = 1.0 / denominatory
#     e_coeffz = 1.0 / denominatorz
#
#     e_coeff_yx = e_coeffy / (dx * kappa_e_x)
#     e_coeff_xy = e_coeffx / (dy * kappa_e_y)
#     e_coeff_zx = e_coeffz / (dx * kappa_e_x)
#     e_coeff_zy = e_coeffz / (dy * kappa_e_y)
#
#     # Magnetic field update coefficients
#     denominator_h = mu0 / dt + sigma_h / 2
#     h_coeff_1 = (mu0 / dt - sigma_h / 2) / denominator_h
#     h_coeff = 1.0 / (denominator_h)
#     h_coeff_x = h_coeff/(dx * kappa_h_x)
#     h_coeff_y = h_coeff/(dx * kappa_h_y)
#
#
#     probe = []
#
#     for dft in dft_region:
#         ce = dft["center"]
#         si = dft["size"]
#         dft["omegas"] = 2 * np.pi * c0 / dft["wavelengths"]
#
#         dx_ = 0.25 * dx
#         dy_ = 0.25 * dy
#
#         if dft["type"] == "design" and not staircasing:
#             iminx = design_grid[0]["iminx"]
#             imaxx = design_grid[0]["imaxx"]
#             jminx = design_grid[0]["jminx"]
#             jmaxx = design_grid[0]["jmaxx"]
#             iminy = design_grid[0]["iminy"]
#             imaxy = design_grid[0]["imaxy"]
#             jminy = design_grid[0]["jminy"]
#             jmaxy = design_grid[0]["jmaxy"]
#             iminz = design_grid[0]["iminz"]
#             imaxz = design_grid[0]["imaxz"]
#             jminz = design_grid[0]["jminz"]
#             jmaxz = design_grid[0]["jmaxz"]
#
#             dft["iminx"] = iminx
#             dft["imaxx"] = imaxx
#             dft["jminx"] = jminx
#             dft["jmaxx"] = jmaxx
#             dft["iminy"] = iminy
#             dft["imaxy"] = imaxy
#             dft["jminy"] = jminy
#             dft["jmaxy"] = jmaxy
#             dft["iminz"] = iminz
#             dft["imaxz"] = imaxz
#             dft["jminz"] = jminz
#             dft["jmaxz"] = jmaxz
#             print(imaxx - iminx)
#
#             dft["Ex"] = np.zeros([len(dft["wavelengths"]),imaxx-iminx,jmaxx-jminx], dtype = complex_type)
#             dft["Ey"] = np.zeros([len(dft["wavelengths"]),imaxy-iminy,jmaxy-jminy], dtype = complex_type)
#             dft["Ez"] = np.zeros([len(dft["wavelengths"]),imaxz-iminz,jmaxz-jminz], dtype = complex_type)
#
#         else:
#             imin = np.argmin(np.abs(x_axis + dx_ - ce[0] + si[0] / 2))
#             imax = imin + int(round(si[0] / dx))
#             jmin = np.argmin(np.abs(y_axis + dy_ - ce[1] + si[1] / 2))
#             jmax = jmin + int(round(si[1] / dy))
#             dft["iminx"] = imin
#             dft["imaxx"] = imax
#             dft["jminx"] = jmin
#             dft["jmaxx"] = jmax
#             dft["iminy"] = imin
#             dft["imaxy"] = imax
#             dft["jminy"] = jmin
#             dft["jmaxy"] = jmax
#             dft["iminz"] = imin
#             dft["imaxz"] = imax
#             dft["jminz"] = jmin
#             dft["jmaxz"] = jmax
#             dft["Ex"] = np.zeros([len(dft["wavelengths"]),imax-imin,jmax-jmin], dtype = complex_type)
#             dft["Ey"] = np.zeros([len(dft["wavelengths"]),imax-imin,jmax-jmin], dtype = complex_type)
#             dft["Ez"] = np.zeros([len(dft["wavelengths"]),imax-imin,jmax-jmin], dtype = complex_type)
#
#     for fr in flux_region:
#         si = fr["size"]
#         ce = fr["center"]
#         imin = np.argmin(np.abs(x_axis - (ce[0] - si[0] / 2)))
#         imax = np.argmin(np.abs(x_axis - (ce[0] + si[0] / 2)))
#         jmin = np.argmin(np.abs(y_axis - (ce[1] - si[1] / 2)))
#         jmax = np.argmin(np.abs(y_axis - (ce[1] + si[1] / 2)))
#
#         if imin == imax:
#             imax = imin + 1
#             if "direction" not in fr:
#                 fr["direction"] = "+x"
#         if jmin == jmax:
#             jmax = jmin + 1
#             if "direction" not in fr:
#                 fr["direction"] = "+y"
#
#         fr["imin"] = imin
#         fr["imax"] = imax
#         fr["jmin"] = jmin
#         fr["jmax"] = jmax
#
#         if fr["direction"] == "+y" or fr["direction"] == "-y":
#             fr["Ex"] = np.zeros([len(fr["wavelengths"]),abs(fr["imax"]-fr["imin"])],dtype = "complex128")
#             fr["Ez"] = np.zeros([len(fr["wavelengths"]),abs(fr["imax"]-fr["imin"])],dtype = "complex128")
#             fr["Hx"] = np.zeros([len(fr["wavelengths"]),abs(fr["imax"]-fr["imin"])],dtype = "complex128")
#             fr["Hz"] = np.zeros([len(fr["wavelengths"]),abs(fr["imax"]-fr["imin"])],dtype = "complex128")
#             fr["direction_"] = "y"
#         if fr["direction"] == "+x" or fr["direction"] == "-x":
#             fr["Ey"] = np.zeros([len(fr["wavelengths"]),abs(fr["jmax"]-fr["jmin"])],dtype = "complex128")
#             fr["Ez"] = np.zeros([len(fr["wavelengths"]),abs(fr["jmax"]-fr["jmin"])],dtype = "complex128")
#             fr["Hy"] = np.zeros([len(fr["wavelengths"]),abs(fr["jmax"]-fr["jmin"])],dtype = "complex128")
#             fr["Hz"] = np.zeros([len(fr["wavelengths"]),abs(fr["jmax"]-fr["jmin"])],dtype = "complex128")
#             fr["direction_"] = "x"
#
#         fr["omegas"] = 2 * np.pi * c0 / fr["wavelengths"]
#
#
#     max_field = 0.0
#     # FDTD algorithm
#     for n in range(N_time_steps):
#         time = (n+1) * dt
#         time_ = (n+0.5) * dt
#
#         Hx_prev = Hx.copy()
#         Hy_prev = Hy.copy()
#         Hz_prev = Hz.copy()
#         Ex_prev = Ex.copy()
#         Ey_prev = Ey.copy()
#         Ez_prev = Ez.copy()
#
#         # update magnetic field at n+1/2
#         if TE:
#             Hx[1:-1,:-1] = (h_coeff_1 * Hx_prev[1:-1,:-1]
#                           - h_coeff_y[:,:-1] * (Ez[1:-1,1:] - Ez[1:-1,:-1]))
#
#             Hy[:-1,1:-1] = (h_coeff_1 * Hy_prev[:-1,1:-1]
#                           + h_coeff_x[:-1,:] * (Ez[1:,1:-1] - Ez[:-1,1:-1]))
#         else:
#             Hz[:-1,:-1] = (h_coeff_1 * Hz_prev[:-1,:-1]
#                           + h_coeff_y[:,:-1]  * (Ex[:-1,1:] - Ex[:-1,:-1])
#                           - h_coeff_x[:-1,:] * (Ey[1:,:-1] - Ey[:-1,:-1]))
#
#         # Hx PML along y-direction
#         if TE:
#             psi_Hxy[:, :n_cells_pml - 1, 0] = (psi_Hxy[:, :n_cells_pml - 1, 0] * bh_y[:, :-1]
#                                                +  ch_y[:, :-1] * Inv_dy * (Ez[:, 1:n_cells_pml] - Ez[:, :n_cells_pml - 1]))
#
#             psi_Hxy[:, :n_cells_pml - 1, 1] = (psi_Hxy[:, :n_cells_pml - 1, 1] * bh_y_f[:, 1:]
#                                                + ch_y_f[:, 1:] * Inv_dy * (Ez[:, -n_cells_pml + 1:] - Ez[:, -n_cells_pml:-1]))
#
#             Hx[:-1, :n_cells_pml -1] -= h_coeff * psi_Hxy[:-1, :n_cells_pml - 1,0]
#             Hx[:-1, -n_cells_pml:-1] -= h_coeff * psi_Hxy[:-1, :n_cells_pml - 1,1]
#
#             # Hy PML along x-direction
#             psi_Hyx[:n_cells_pml - 1, :, 0] = (psi_Hyx[:n_cells_pml - 1, :, 0] * bh_x[:-1, :]
#                                                +  ch_x[:-1, :] * Inv_dx * (Ez[1:n_cells_pml,:] - Ez[:n_cells_pml-1,:]))
#
#             psi_Hyx[:n_cells_pml - 1, :, 1] = (psi_Hyx[:n_cells_pml - 1, :, 1] * bh_x_f[1:, :]
#                                                +  ch_x_f[1:, :] * Inv_dy * (Ez[-n_cells_pml + 1:,:] - Ez[-n_cells_pml:-1,:]))
#
#             Hy[:n_cells_pml -1,:-1] += h_coeff * psi_Hyx[:n_cells_pml - 1,:-1,0]
#             Hy[-n_cells_pml:-1,:-1] += h_coeff * psi_Hyx[:n_cells_pml - 1,:-1,1]
#         else:
#             # Hz PML along x-direction
#             psi_Hzx[:n_cells_pml - 1, :, 0] =  (psi_Hzx[:n_cells_pml - 1, :, 0] * bh_x[:-1, :]
#                                                  + (Ey[1:n_cells_pml,:] - Ey[:n_cells_pml - 1,:]) * ch_x[:-1, :] * Inv_dx)
#
#             psi_Hzx[:n_cells_pml - 1, :, 1] = (psi_Hzx[:n_cells_pml - 1, :, 1] * bh_x_f[1:, :]
#                                                  + (Ey[-n_cells_pml + 1:, :] - Ey[-n_cells_pml:-1, :]) * ch_x_f[1:, :] * Inv_dx)
#
#             Hz[:n_cells_pml - 1, :-1] -= h_coeff * psi_Hzx[:n_cells_pml - 1,:-1,0]
#             Hz[-n_cells_pml:-1, :-1] -= h_coeff * psi_Hzx[:n_cells_pml - 1, :-1, 1]
#
#             # Hz PML along y-direction
#             psi_Hzy[:, :n_cells_pml - 1, 0] = (psi_Hzy[:, :n_cells_pml - 1, 0] * bh_y[:, :-1]
#                                                + (Ex[:, 1:n_cells_pml] - Ex[:, :n_cells_pml - 1]) * ch_y[:, :-1] * Inv_dy)
#
#             psi_Hzy[:, :n_cells_pml - 1, 1] = (psi_Hzy[:, :n_cells_pml - 1, 1] * bh_y_f[:, 1:]
#                                                + ch_y_f[:, 1:,] * Inv_dy * (Ex[:,-n_cells_pml + 1:] - Ex[:,-n_cells_pml:-1]))
#
#             Hz[:-1,  :n_cells_pml-1] += h_coeff * psi_Hzy[:-1,:n_cells_pml - 1,0]
#             Hz[:-1, -n_cells_pml:-1] += h_coeff * psi_Hzy[:-1,:n_cells_pml - 1,1]
#
#         # add magnetic field source
#         if not TE:
#             for s in source:
#                 source_dir = s["direction"]
#                 imin = s["imin"]
#                 imax = s["imax"]
#                 jmin = s["jmin"]
#                 jmax = s["jmax"]
#                 t_offset = s["t_offset"]
#                 Z = 1 #s["Z"]
#                 # print(s["Hz"].shape, Hz[imin:imax,jmin-1].shape, (s["Hz"] - Hz[imin:imax,jmin-1]).shape)
#                 if source_dir == "+y":
#                     Hz[imin:imax,jmin-1] = Hz[imin:imax,jmin-1] - h_coeff * s["Ex"]  * s["signal"]((n + 0.5)*dt - t_offset) / dy
#                 elif source_dir == "-y":
#                     Hz[imin:imax,jmin-1] = Hz[imin:imax,jmin-1] + h_coeff * s["Ex"]  * s["signal"]((n + 0.5)*dt + t_offset) / dy
#                 elif source_dir == "-x":
#                     Hz[imin-1,jmin:jmax] = Hz[imin-1,jmin:jmax] - h_coeff * s["Ey"]  * s["signal"]((n + 0.5) * dt + t_offset) / dx
#                 elif source_dir == "+x":
#                     Hz[imin-1,jmin:jmax] = Hz[imin-1,jmin:jmax] + h_coeff * s["Ey"]  * s["signal"]((n + 0.5) * dt - t_offset) / dx
#         else:
#             for s in source:
#                 source_dir = s["direction"]
#                 imin = s["imin"]
#                 imax = s["imax"]
#                 jmin = s["jmin"]
#                 jmax = s["jmax"]
#                 t_offset = s["t_offset"]
#                 Z = 1 #s["Z"]
#                 # print(s["Hz"].shape, Hz[imin:imax,jmin-1].shape, (s["Hz"] - Hz[imin:imax,jmin-1]).shape)
#                 if source_dir == "+y":
#                     Hx[imin:imax,jmin-1] = Hx[imin:imax,jmin-1] + h_coeff * s["Ez"]  * s["signal"]((n + 0.5)*dt - t_offset) / dy
#                 elif source_dir == "-y":
#                     Hx[imin:imax,jmin-1] = Hx[imin:imax,jmin-1] - h_coeff * s["Ez"]  * s["signal"]((n + 0.5)*dt + t_offset) / dy
#                 elif source_dir == "-x":
#                     Hy[imin-1,jmin:jmax] = Hy[imin-1,jmin:jmax] + h_coeff * s["Ez"]  * s["signal"]((n + 0.5) * dt + t_offset) / dx
#                 elif source_dir == "+x":
#                     Hy[imin-1,jmin:jmax] = Hy[imin-1,jmin:jmax] - h_coeff * s["Ez"]  * s["signal"]((n + 0.5) * dt - t_offset) / dx
#                     # pass
#         # update electric field at n+1
#         if not TE:
#             Ex[:-1,1:-1] = (e_coeff_1x[:-1,1:-1] * Ex_prev[:-1,1:-1]
#                            + e_coeff_xy[:-1,1:-1] * (Hz[:-1,1:-1] - Hz[:-1,:-2]))
#
#             Ey[1:-1,:-1] = (e_coeff_1y[1:-1,:-1] * Ey_prev[1:-1,:-1]
#                            - e_coeff_yx[1:-1,:-1] * (Hz[1:-1,:-1] - Hz[:-2,:-1]))
#         else:
#             Ez[1:-1,1:-1] = (e_coeff_1z[1:-1,1:-1] * Ez_prev[1:-1,1:-1]
#                            + e_coeff_zx[1:-1,1:-1] * (Hy[1:-1,1:-1] - Hy[:-2,1:-1])
#                            - e_coeff_zy[1:-1,1:-1] * (Hx[1:-1,1:-1] - Hx[1:-1,:-2]))
#
#         if not TE:
#             # Ex PML along y-direction
#             psi_Exy[:, 1:n_cells_pml, 0] = (be_y[:, 1:] * psi_Exy[:, 1:n_cells_pml, 0]
#                                             + ce_y[:, 1:] * Inv_dy * (Hz[:, 1:n_cells_pml] - Hz[:, :n_cells_pml - 1]))
#
#             psi_Exy[:, 1:n_cells_pml, 1] = (be_y_f[:, :-1] * psi_Exy[:, 1:n_cells_pml, 1]
#                                             + ce_y_f[:, :-1] * Inv_dy * (Hz[:, -n_cells_pml:-1] - Hz[:, -n_cells_pml - 1:-2]))
#
#             Ex[:-1, 1:n_cells_pml] += e_coeffx[:-1, 1:n_cells_pml] * psi_Exy[:-1, 1:n_cells_pml, 0]
#             Ex[:-1, -n_cells_pml:-1] += e_coeffx[:-1, -n_cells_pml:-1] * psi_Exy[:-1, 1:n_cells_pml, 1]
#
#             # Ey PML along x-direction
#             psi_Eyx[1:n_cells_pml, :, 0] = (be_x[1:, :] * psi_Eyx[1:n_cells_pml, :, 0] +
#                                                  Inv_dx * ce_x[1:, :] * (Hz[1:n_cells_pml, :] - Hz[:n_cells_pml - 1, :]))
#             psi_Eyx[1:n_cells_pml, :, 1] = (be_x_f[:-1, :] * psi_Eyx[1:n_cells_pml, :, 1]
#                                             + Inv_dx * ce_x_f[:-1, :] * (Hz[-n_cells_pml:-1, :] - Hz[-n_cells_pml - 1:-2, :]))
#
#             Ey[1:n_cells_pml, :-1] -= e_coeffy[1:n_cells_pml, :-1] * psi_Eyx[1:n_cells_pml,:-1,0]
#             Ey[-n_cells_pml:-1, :-1] -= e_coeffy[-n_cells_pml:-1, :-1] * psi_Eyx[1:n_cells_pml, :-1, 1]
#         else:
#             # Ez PML along x-direction
#             psi_Ezx[1:n_cells_pml, :, 0] = (be_x[1:, :] * psi_Ezx[1:n_cells_pml, :, 0] +
#                                             Inv_dx * ce_x[1:, :] * (Hy[1:n_cells_pml, :] - Hy[:n_cells_pml - 1, :]))
#             psi_Ezx[1:n_cells_pml, :, 1] = (be_x_f[:-1, :] * psi_Ezx[1:n_cells_pml, :, 1] +
#                                             Inv_dx * ce_x_f[:-1, :] * (Hy[-n_cells_pml:-1, :] - Hy[-n_cells_pml - 1:-2, :]))
#
#             Ez[1:n_cells_pml, 1:-1] += e_coeffz[1:n_cells_pml, 1:-1] * psi_Ezx[1:n_cells_pml, 1:-1, 0]
#             Ez[-n_cells_pml:-1, 1:-1] += e_coeffz[-n_cells_pml:-1, 1:-1] * psi_Ezx[1:n_cells_pml, 1:-1, 1]
#
#             # Ez PML along y-direction
#             psi_Ezy[:, 1:n_cells_pml, 0] = (be_y[:, 1:] * psi_Ezy[:, 1:n_cells_pml, 0] +
#                                             ce_y[:, 1:] * Inv_dy * (Hx[:, 1:n_cells_pml] - Hx[:, :n_cells_pml - 1]))
#             psi_Ezy[:, 1:n_cells_pml, 1] = (be_y_f[:, :-1] * psi_Ezy[:, 1:n_cells_pml, 1] +
#                                             ce_y_f[:, :-1] * Inv_dy * (Hx[:, -n_cells_pml:-1] - Hx[:, -n_cells_pml - 1:-2]))
#
#             Ez[1:-1, 1:n_cells_pml] -= e_coeffz[1:-1, 1:n_cells_pml,] * psi_Ezy[1:-1, 1:n_cells_pml, 0]
#             Ez[1:-1, -n_cells_pml:-1] -= e_coeffz[1:-1, -n_cells_pml:-1] * psi_Ezy[1:-1, 1:n_cells_pml, 1]
#
#         # add electric field source
#         if not TE:
#             for s in source:
#                 source_dir = s["direction"]
#                 imin = s["imin"]
#                 imax = s["imax"]
#                 jmin = s["jmin"]
#                 jmax = s["jmax"]
#                 if source_dir == "+y":
#                     Ex[imin:imax,jmin] = Ex[imin:imax,jmin] - s["Hz"] * s["signal"]((n + 1)*dt) * e_coeffx[imin:imax,jmin]/dy
#                 elif source_dir == "-y":
#                     Ex[imin:imax,jmin] = Ex[imin:imax,jmin] + s["Hz"] * s["signal"]((n + 1)*dt) * e_coeffx[imin:imax,jmin]/dy
#                 elif source_dir == "+x":
#                     Ey[imin,jmin:jmax] = Ey[imin,jmin:jmax] + s["Hz"] * s["signal"]((n + 1)*dt) * e_coeffy[imin,jmin:jmax]/dx
#                 elif source_dir == "-x":
#                     Ey[imin,jmin:jmax] = Ey[imin,jmin:jmax] - s["Hz"] * s["signal"]((n + 1)*dt) * e_coeffy[imin,jmin:jmax]/dx
#
#         else:
#             pass
#             for s in source:
#                 source_dir = s["direction"]
#                 imin = s["imin"]
#                 imax = s["imax"]
#                 jmin = s["jmin"]
#                 jmax = s["jmax"]
#                 if source_dir == "+y":
#                     Ez[imin:imax,jmin] = Ez[imin:imax,jmin] + s["Hx"] * s["signal"]((n + 1)*dt) * e_coeffz[imin:imax,jmin]/dy
#                 elif source_dir == "-y":
#                     Ez[imin:imax,jmin] = Ez[imin:imax,jmin] - s["Hx"] * s["signal"]((n + 1)*dt) * e_coeffz[imin:imax,jmin]/dy
#                 elif source_dir == "+x":
#                     Ez[imin,jmin:jmax] = Ez[imin,jmin:jmax] - s["Hy"] * s["signal"]((n + 1)*dt) * e_coeffz[imin,jmin:jmax]/dx
#                 elif source_dir == "-x":
#                     Ez[imin,jmin:jmax] = Ez[imin,jmin:jmax] + s["Hy"] * s["signal"]((n + 1)*dt) * e_coeffz[imin,jmin:jmax]/dx
#
#
#         for d, dft in enumerate(dft_region):
#             for ww, w in enumerate(dft["omegas"]):
#                 exp_term = np.exp(-1j * time * w)
#                 if not TE:
#                     dft["Ex"][ww,:,:] += Ex[dft["iminx"]:dft["imaxx"],dft["jminx"]:dft["jmaxx"]] * exp_term
#                     dft["Ey"][ww,:,:] += Ey[dft["iminy"]:dft["imaxy"],dft["jminy"]:dft["jmaxy"]] * exp_term
#                 else:
#                     dft["Ez"][ww,:,:] += Ez[dft["iminz"]:dft["imaxz"],dft["jminz"]:dft["jmaxz"]] * exp_term
#
#         for fr in flux_region:
#             for ww,w in enumerate(fr["omegas"]):
#                 exp_e =  np.exp(-1j * time * w)
#                 exp_h =  np.exp(-1j * time_ * w)
#                 if fr["direction_"] == "y":
#                     fr["Ex"][ww,:] += Ex[fr["imin"]:fr["imax"],fr["jmin"]] * exp_e
#                     fr["Ez"][ww,:] += Ez[fr["imin"]:fr["imax"],fr["jmin"]] * exp_e
#                     fr["Hx"][ww,:] += 0.5 * (Hx[fr["imin"]:fr["imax"],fr["jmin"]]
#                                              + Hx[fr["imin"]:fr["imax"],fr["jmin"]-1]) * exp_h
#                     fr["Hz"][ww,:] += 0.5 * (Hz[fr["imin"]:fr["imax"],fr["jmin"]]
#                                              + Hz[fr["imin"]:fr["imax"],fr["jmin"]-1]) * exp_h
#                 if fr["direction_"] == "x":
#                     fr["Ey"][ww,:] += Ey[fr["imin"],fr["jmin"]:fr["jmax"]] * exp_e
#                     fr["Ez"][ww,:] += Ez[fr["imin"],fr["jmin"]:fr["jmax"]] * exp_e
#                     fr["Hy"][ww,:] += 0.5 * (Hy[fr["imin"],fr["jmin"]:fr["jmax"]]
#                                              + Hy[fr["imin"]-1,fr["jmin"]:fr["jmax"]]) * exp_h
#                     fr["Hz"][ww,:] += 0.5 * (Hz[fr["imin"],fr["jmin"]:fr["jmax"]]
#                                              + Hz[fr["imin"]-1,fr["jmin"]:fr["jmax"]]) * exp_h
#
#
#         mf = np.sqrt(np.max(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2))
#         if mf > max_field:
#             max_field = mf
#
#         # shutoff simulation early
#         if time > min_simulation_time:
#             if mf < cutoff * max_field:
#                 print("Electric field below threshold, breaking")
#                 break
#
#         if n % 100 == 0:
#             print(n,np.max(np.abs(Ex)), np.max(np.abs(Ey)), np.max(np.abs(Ez)),max_field,mf/max_field)
#
#         if n % movie_update == 0:
#             E_movie.append(Ex**2 + Ey**2 + Ez**2)
#
#
#     for so, s in enumerate(source):
#         pulse = s["signal"](np.arange(N_time_steps) * dt)
#         pulsew = Discrete_Fourier_Transform(pulse, np.arange(N_time_steps) * dt, fr["omegas"])
#         s["pulsew"] = pulsew.copy()
#
#     for fr in flux_region:
#         fr["pulses"] = []
#         for so, s in enumerate(source):
#             fr["pulses"].append(s["pulsew"])
#
#         if "mode" in fr:
#             imin = fr["imin"]
#             imax = fr["imax"]
#             jmin = fr["jmin"]
#             jmax = fr["jmax"]
#
#             if fr["size"][0] == 0:
#                 epsyy_ = epszz[imin:imax, jmin:jmax].transpose()
#                 epszz_ = epsxx[imin:imax, jmin:jmax].transpose()
#                 epsxx_ = epsyy[imin:imax, jmin:jmax].transpose()
#             else:
#                 epsxx_ = epsxx[imin:imax, jmin:jmax]#.transpose()
#                 epsyy_ = epszz[imin:imax, jmin:jmax]#.transpose()
#                 epszz_ = epsyy[imin:imax, jmin:jmax]#.transpose()
#
#             E1s = []
#             E2s = []
#             H1s = []
#             H2s = []
#             ns = []
#
#             if "mode Ey" not in fr:
#                 for w,wl in enumerate(fr["wavelengths"]):
#                     n, E1, E2, H1, H2 = Mode_Solver(epsxx_, epsyy_, epszz_, dx, dy, wl, fr["mode"])
#                     E1s.append(E1)
#                     E2s.append(E2)
#                     H1s.append(H1)
#                     H2s.append(H2)
#                     ns.append(n)
#                     print(n)
#                 fr["mode index"] = np.array(ns)
#                 if fr["direction"] == "+x" or fr["direction"] == "-x":
#                     fr["mode Ey"] = np.array(E1s)[:,:,0]
#                     fr["mode Ez"] = np.array(E2s)[:,:,0]
#                     fr["mode Hy"] = np.array(H1s)[:,:,0]
#                     fr["mode Hz"] = np.array(H2s)[:,:,0]
#                 if fr["direction"] == "+y" or fr["direction"] == "-y":
#                     fr["mode Ex"] = np.array(E1s)[:,:,0]
#                     fr["mode Ez"] = np.array(E2s)[:,:,0]
#                     fr["mode Hx"] = -np.array(H1s)[:,:,0]
#                     fr["mode Hz"] = -np.array(H2s)[:,:,0]
#
#             # plt.plot(np.abs(fr["Hz"][5,40:60]) / np.abs(fr["mode Hz"][5,40:60]))
#             # plt.show()
#
#             if fr['direction'] == "+x" or fr["direction"] == "-x":
#                 fr['mode amplitude +x'] = GetModeOverlap(fr['Ey'], fr['Ez'], fr['Hy'], fr['Hz'],
#                                                           fr['mode Ey'], fr['mode Ez'], fr['mode Hy'],
#                                                           fr['mode Hz'])
#                 fr['mode amplitude -x'] = GetModeOverlap(fr['Ey'], fr['Ez'], fr['Hy'], fr['Hz'],
#                                                           fr['mode Ey'], fr['mode Ez'], -fr['mode Hy'],
#                                                           -fr['mode Hz'])
#                 fr['Power norm +x'] = GetModeOverlap(fr['mode Ey'], fr['mode Ez'], fr['mode Hy'], fr['mode Hz'],
#                                                         fr['mode Ey'], fr['mode Ez'], fr['mode Hy'],
#                                                         fr['mode Hz'])
#                 fr['Power norm -x'] = GetModeOverlap(fr['mode Ey'], fr['mode Ez'], fr['mode Hy'], fr['mode Hz'],
#                                                         fr['mode Ey'], fr['mode Ez'], -fr['mode Hy'],
#                                                         -fr['mode Hz'])
#             if fr['direction'] == "+y" or fr["direction"] == "-y":
#                 # plt.plot(np.abs(fr['Hz'][5,40:60]) / np.abs(fr['mode Hz'][5,40:60]))
#                 # plt.show()
#                 fr['mode amplitude +y'] = GetModeOverlap(fr['Ez'], fr['Ex'], fr['Hz'], fr['Hx'],
#                                                           fr['mode Ez'], fr['mode Ex'], fr['mode Hz'],
#                                                           fr['mode Hx'])
#                 fr['mode amplitude -y'] = GetModeOverlap(fr['Ez'], fr['Ex'], fr['Hz'], fr['Hx'],
#                                                           fr['mode Ez'], fr['mode Ex'], -fr['mode Hz'],
#                                                           -fr['mode Hx'])
#                 fr['Power norm +y'] = GetModeOverlap(fr['mode Ez'], fr['mode Ex'], fr['mode Hz'], fr['mode Hx'],
#                                                           fr['mode Ez'], fr['mode Ex'], fr['mode Hz'],
#                                                           fr['mode Hx'])
#                 fr['Power norm -y'] = GetModeOverlap(fr['mode Ez'], fr['mode Ex'], fr['mode Hz'], fr['mode Hx'],
#                                                           fr['mode Ez'], fr['mode Ex'], -fr['mode Hz'],
#                                                           -fr['mode Hx'])
#     if len(design_grid) > 0:
#         return E_movie, flux_region, dft_region, design_grid
#     else:
#         return E_movie, flux_region, dft_region

