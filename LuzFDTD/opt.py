import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import functools
from jax import grad as grad_jax
from .FDTD2D import c0, eps0, FDTD_2D
from .FDTD2D_jax import FDTD_2D as FDTD_2D_jax
from .FDTD3D_jax import FDTD_3D as FDTD_3D_jax
import inspect

def update_parameters_with_adam(x, grads, s, v,
                                t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999,
                                epsilon=1e-8):
    s = beta1 * s + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * grads ** 2
    s_hat = s / (1.0 - beta1 ** (t + 1))
    v_hat = v / (1.0 - beta2 ** (t + 1))
    x = x - learning_rate * s_hat / (np.sqrt(v_hat) + epsilon)
    return x, s, v

class LuzOpt:
    def __init__(self, sim, objective, design_region, design_wavelengths, pw_adj = 10e-15, with_jax = False):

        self.ThreeD = False      
        self.adj_sim = None
        self.adj_source = []
        self.sim = sim
        design_region[0]["wavelengths"] = design_wavelengths        
        self.design_region = design_region
        self.sim.geometry += design_region
        self.design_wavelengths = design_wavelengths

        self.sim.dft_region += copy.deepcopy(design_region)
        self.adj_design_region = copy.deepcopy(design_region)
        self.design_freqs = c0 / design_wavelengths
        self.pw_adj = pw_adj
        if len(self.design_wavelengths) > 1:
            dfs = np.abs(np.diff(self.design_freqs))
            min_df = np.min(dfs)
            self.pw_adj = 2./min_df
            print(self.pw_adj)

        if len(sim.simulation_size) > 2:
            if sim.simulation_size[2] > 0:
                self.ThreeD = True

        self.objective = objective

    def opt_step(self, grid):
        self.sim.geometry[-1]['grid'] = grid.copy()
        num_args = len(inspect.signature(self.objective).parameters)
        argnums = tuple([i for i in range(num_args)])
        gradient_function = grad_jax(self.objective, argnums=argnums)

        E_movie, flux_region, dft_fwd, design_grid = self.sim.run()

        dt = self.sim.dt
        dy = self.sim.dy
        dx = self.sim.dx

        amps = []
        for fr in flux_region:
            neff = fr["mode index"]
            if fr["direction"] == "+y":
                trans = fr["mode amplitude +y"]
                fact = 1
                new_dir = "-y"
                phase_corr = - 0.5 * neff * dy * fr["omegas"] / c0

            if fr["direction"] == "-y":
                trans = fr["mode amplitude -y"]
                fact = -1
                new_dir = "+y"
                phase_corr =  0.5 * neff *  dy * fr["omegas"] / c0

            if fr["direction"] == "+x":
                trans = fr["mode amplitude +x"]
                fact = 1
                new_dir = "-x"
                phase_corr = - 0.5 * neff *  dx * fr["omegas"] / c0

            if fr["direction"] == "-x":
                trans = fr["mode amplitude -x"]
                fact = -1
                new_dir = "+x"
                phase_corr =  0.5 * neff *  dx * fr["omegas"] / c0
            amps.append(trans)

        print(amps)
        adj_amps = gradient_function(*amps)
        adj_amps = np.nan_to_num(adj_amps)

        print(amps)
        print(adj_amps)

        FOM = self.objective(*amps)
        print(FOM)

        if self.adj_sim is None:
            for f,fr in enumerate(flux_region):
                for w,wl in enumerate(self.design_wavelengths):
                    ome = 2 * np.pi * c0 / wl
                    neff = fr["mode index"][w]
                    if fr["direction"] == "+y":
                        fact = 1
                        new_dir = "-y"
                        phase_corr = - 0.5 * neff * dy * ome/ c0

                    if fr["direction"] == "-y":
                        fact = -1
                        new_dir = "+y"
                        phase_corr =  0.5 * neff *  dy * ome / c0

                    if fr["direction"] == "+x":
                        fact = 1
                        new_dir = "-x"
                        phase_corr = - 0.5 * neff *  dx * ome / c0

                    if fr["direction"] == "-x":
                        fact = -1
                        new_dir = "+x"
                        phase_corr =  0.5 * neff *  dx * ome / c0

                    adj_amp = np.abs(adj_amps[f,w]) * fact
                    adj_phs = np.angle(adj_amps[f,w]) + phase_corr
                    adj_source_center = fr["center"]
                    adj_source_size = fr["size"]
                    print(wl)
                    self.adj_source.append({"type":"adjmode","size":adj_source_size,"center":adj_source_center, "mode": fr["mode"],
                                   "wavelength":wl, "pulse width": self.pw_adj,
                                   "direction":new_dir, "amplitude":adj_amp, "phase": adj_phs})
        else:
            id = 0
            for f,fr in enumerate(flux_region):
                for w,wl in enumerate(self.design_wavelengths):
                    ome = 2 * np.pi * c0 / wl
                    neff = fr["mode index"][w]
                    if fr["direction"] == "+y":
                        fact = 1
                        new_dir = "-y"
                        phase_corr = - 0.5 * neff * dy * ome/ c0

                    if fr["direction"] == "-y":
                        fact = -1
                        new_dir = "+y"
                        phase_corr =  0.5 * neff *  dy * ome / c0

                    if fr["direction"] == "+x":
                        fact = 1
                        new_dir = "-x"
                        phase_corr = - 0.5 * neff *  dx * ome / c0

                    if fr["direction"] == "-x":
                        fact = -1
                        new_dir = "+x"
                        phase_corr =  0.5 * neff *  dx * ome / c0

                    adj_amp = np.abs(adj_amps[f,w]) * fact
                    adj_phs = np.angle(adj_amps[f,w]) + phase_corr
                    self.adj_sim.source[id]["amplitude"] = adj_amp
                    self.adj_sim.source[id]["phase"] = adj_phs

        if self.adj_sim is None:
            if self.ThreeD:
                print("Adjoint sim")
                self.adj_sim = FDTD_3D_jax(self.sim.simulation_size,
                                   self.sim.step_size,
                                   source=self.adj_source,
                                   geometry = self.sim.geometry,
                                   flux_region=[],
                                   dft_region = copy.deepcopy(self.adj_design_region),
                                   cutoff = self.sim.cutoff,
                                   movie_update=100, staircasing=self.sim.staircasing)

            else:
                if with_jax:
                    print("Adjoint sim")
                    self.adj_sim = FDTD_2D_jax(self.sim.simulation_size,
                                       self.sim.step_size,
                                       source=self.adj_source,
                                       geometry = self.sim.geometry,
                                       flux_region=[],
                                       dft_region = copy.deepcopy(self.adj_design_region),
                                       cutoff = sim.cutoff,
                                       movie_update=100, TE = TE, staircasing=sim.staircasing)
                else:
                    self.adj_sim = FDTD_2D(sim.simulation_size,
                                       self.sim.step_size,
                                       source=self.adj_source,
                                       geometry = self.sim.geometry,
                                       flux_region=[],
                                       dft_region = copy.deepcopy(self.adj_design_region),
                                       cutoff = self.sim.cutoff,
                                       movie_update=100, TE = TE, staircasing=self.sim.staircasing)
        
        self.adj_sim.geometry[-1]["grid"] = grid.copy()
        E_movie, _, dft_adj, design_grid = self.adj_sim.run()

        omega = dft_fwd[0]["omegas"]
        Exg = dft_fwd[0]["Ex"]
        Eyg = dft_fwd[0]["Ey"]
        Ezg = dft_fwd[0]["Ez"]
        Exga = dft_adj[0]["Ex"]
        Eyga = dft_adj[0]["Ey"]
        Ezga = dft_adj[0]["Ez"]

        if not self.sim.staircasing:
            jacx = design_grid[0]["jacx"]
            jacy = design_grid[0]["jacy"]
            jacz = design_grid[0]["jacz"]

            gradient = 0
            for w,wl in enumerate(self.design_wavelengths):
                omega_ = 2 * np.pi * c0 / wl
                phase_fact = np.exp(-1j * dt * omega_ * 0.5)
                if len(Exg[w].shape) == 2:
                    gradient_x = -(1j * omega_ * phase_fact * eps0 * Exg[w] * Exga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2))
                    gradient_y = -(1j * omega_ * phase_fact * eps0 * Eyg[w] * Eyga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2))
                    gradient_z = -(1j * omega_ * phase_fact * eps0 * Ezg[w] * Ezga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2))
                else:
                    gradient_x = -np.sum(1j * omega_ * phase_fact * eps0 * Exg[w] * Exga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis=2)
                    gradient_y = -np.sum(1j * omega_ * phase_fact * eps0 * Eyg[w] * Eyga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis=2)
                    gradient_z = -np.sum(1j * omega_ * phase_fact * eps0 * Ezg[w] * Ezga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis=2)
                gx = jacx(gradient_x.flatten().real)[0]
                gy = jacy(gradient_y.flatten().real)[0]
                gz = jacz(gradient_z.flatten().real)[0]
                gradient += (gx + gy + gz).reshape(self.design_region[0]["grid"].shape)

        else:
            i_list = design_grid[0]["i_list"]
            j_list = design_grid[0]["j_list"]
            gradient = 0

            for w,wl in enumerate(self.design_wavelengths):
                omega_ = 2 * np.pi * c0 / wl
                phase_fact = np.exp(-1j * dt * omega_ * 0.5)
                if len(Exg[w].shape) == 2:
                    gradient = -(1j * omega_ * phase_fact * eps0 * (Exg[w] * Exga[w] + Eyg[w] * Eyga[w] + Ezg[w] * Ezga[w])
                                 * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2))
                else:
                    gradient = -np.sum(1j * omega_ * phase_fact * eps0 * (Exg[w] * Exga[w] + Eyg[w] * Eyga[w] + Ezg[w] * Ezga[w])
                                 * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis = 2)

            gradient_tot = np.zeros(design_grid[0]["grid"].shape,dtype = "complex64")

            for i in range(1,len(i_list)):
                for j in range(1, len(j_list)):
                    gradient_tot[i-1,j-1] = np.sum(gradient[i_list[i-1]:i_list[i],j_list[j-1]:j_list[j]])
            gradient = gradient_tot.copy()
        return FOM, gradient

def fd_step(sim, design_region, design_wavelengths,
              d_d = 0.001, objective = None):

    sim.geometry += design_region
    design_region[0]["wavelengths"] = design_wavelengths
    E_movie, flux_region, dft_fwd, design_grid = sim.run()

    amps0 = []
    for fr in flux_region:
        trans = fr["mode amplitude " + fr["direction"]]
        amps0.append(trans)
    print(amps0)
    FOM0 = objective(*amps0)

    grad_tot = []
    grid_ = design_grid[0]["grid"].flatten()
    grid_shape = design_grid[0]["grid"].shape
    for i in range(grid_.shape[0]):
        grid_this = grid_.copy()
        grid_this[i] += d_d
        sim.geometry[-1]["grid"] = grid_this.reshape(grid_shape)
        E_movie, flux_region, dft_fwd, design_grid = sim.run()
        amps = []
        for fr in flux_region:
            trans = fr["mode amplitude " + fr["direction"]]
            amps.append(trans)

        FOM = objective(*amps)
        grad_tot.append((FOM - FOM0) / d_d)
        print(FOM,FOM0)

    grad_tot = np.array(grad_tot)

    return grad_tot.reshape(grid_shape)