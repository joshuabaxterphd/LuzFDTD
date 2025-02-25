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
import jax.numpy as jnp

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
    def __init__(self, sim, objective, design_region, design_wavelengths, pw_adj = 10e-15, with_jax = False, adj_sim_per_wl = False):

        self.ThreeD = False
        self.with_jax = with_jax
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
        if len(self.design_wavelengths) > 1 and not adj_sim_per_wl:
            dfs = np.abs(np.diff(self.design_freqs))
            min_df = np.min(dfs)
            self.pw_adj = 2./min_df
            print(self.pw_adj)

        if len(sim.simulation_size) > 2:
            if sim.simulation_size[2] > 0:
                self.ThreeD = True

        self.objective = objective
        self.adj_sim_per_wl = adj_sim_per_wl

    def opt_step(self, grid, loss = 0):
        self.sim.geometry[-1]['grid'] = grid.copy()
        self.sim.geometry[-1]['loss'] = loss
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

        adj_amps = gradient_function(*amps)
        adj_amps = np.nan_to_num(adj_amps)

        FOM = self.objective(*amps)

        if self.adj_sim is None:
            self.adj_source = []
            self.adj_source_wl = []
            for w,wl in enumerate(self.design_wavelengths): 
                adj_source_ = []
                for f,fr in enumerate(flux_region):
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
                    adj_source_.append({"type":"adjmode","size":adj_source_size,"center":adj_source_center, "mode": fr["mode"],
                                   "wavelength":wl, "pulse width": self.pw_adj,
                                   "direction":new_dir, "amplitude":adj_amp, "phase": adj_phs})
                self.adj_source_wl.append(adj_source_)

            adj_source_ = None
            if not self.adj_sim_per_wl:
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
                    if self.with_jax:
                        print("Adjoint sim")
                        self.adj_sim = FDTD_2D_jax(self.sim.simulation_size,
                                           self.sim.step_size,
                                           source=self.adj_source,
                                           geometry = self.sim.geometry,
                                           flux_region=[],
                                           dft_region = copy.deepcopy(self.adj_design_region),
                                           cutoff = self.sim.cutoff,
                                           movie_update=100, TE = self.sim.TE, staircasing=self.sim.staircasing)
                    else:
                        self.adj_sim = FDTD_2D(self.sim.simulation_size,
                                           self.sim.step_size,
                                           source=self.adj_source,
                                           geometry = self.sim.geometry,
                                           flux_region=[],
                                           dft_region = copy.deepcopy(self.adj_design_region),
                                           cutoff = self.sim.cutoff,
                                           movie_update=100, TE = self.sim.TE, staircasing=self.sim.staircasing)
            else:
                self.adj_sim = []   
                for w, wl in enumerate(self.design_wavelengths):
                    adj_dr = self.adj_design_region
                    adj_dr[0]["wavelengths"] = np.array([wl])
                    if self.ThreeD:             
                        print("Adjoint sim")
                        self.adj_sim.append(FDTD_3D_jax(self.sim.simulation_size,
                                           self.sim.step_size,
                                           source=self.adj_source_wl[w],
                                           geometry = self.sim.geometry,
                                           flux_region=[],
                                           dft_region = copy.deepcopy(adj_dr),
                                           cutoff = self.sim.cutoff,
                                           movie_update=100, staircasing=self.sim.staircasing))

                    else:
                        if self.with_jax:
                            print("Adjoint sim")
                            self.adj_sim.append(FDTD_2D_jax(self.sim.simulation_size,
                                               self.sim.step_size,
                                               source=self.adj_source_wl[w],
                                               geometry = self.sim.geometry,
                                               flux_region=[],
                                               dft_region = copy.deepcopy(adj_dr),
                                               cutoff = self.sim.cutoff,
                                               movie_update=100, TE = self.sim.TE, staircasing=self.sim.staircasing))
                        else:
                            self.adj_sim.append(FDTD_2D(self.sim.simulation_size,
                                               self.sim.step_size,
                                               source=self.adj_source_wl[w],
                                               geometry = self.sim.geometry,
                                               flux_region=[],
                                               dft_region = copy.deepcopy(adj_dr),
                                               cutoff = self.sim.cutoff,
                                               movie_update=100, TE = self.sim.TE, staircasing=self.sim.staircasing))
        else:
            if not self.adj_sim_per_wl:
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
                        id += 1
            else:
                for w,wl in enumerate(self.design_wavelengths): 
                    id = 0              
                    for f,fr in enumerate(flux_region):
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
                        self.adj_sim[w].source[id]["amplitude"] = adj_amp
                        self.adj_sim[w].source[id]["phase"] = adj_phs
                        id += 1


        # if self.adj_sim is None:
        
        if not self.adj_sim_per_wl:
            self.adj_sim.geometry[-1]["grid"] = grid.copy()
            E_movie, _, dft_adj, design_grid = self.adj_sim.run()
        else:
            dft_adj = []
            for ads in self.adj_sim:
                ads.geometry[-1]["grid"] = grid.copy()
                E_movie, _, dft_adj_, design_grid = ads.run()
                dft_adj.append(dft_adj_)

            dft_adj[0][0]["Ex"] = np.array(dft_adj[0][0]["Ex"]).repeat(2,0)
            dft_adj[0][0]["Ey"] = np.array(dft_adj[0][0]["Ey"]).repeat(2,0)
            dft_adj[0][0]["Ez"] = np.array(dft_adj[0][0]["Ez"]).repeat(2,0)
            dft_adj[0][0]["Ex"][1] = np.array(dft_adj[1][0]["Ex"][0])
            dft_adj[0][0]["Ey"][1] = np.array(dft_adj[1][0]["Ey"][0])
            dft_adj[0][0]["Ez"][1] = np.array(dft_adj[1][0]["Ez"][0])
            dft_adj = dft_adj[0]


        Exg = dft_fwd[0]["Ex"]
        Eyg = dft_fwd[0]["Ey"]
        Ezg = dft_fwd[0]["Ez"]
        Exga = dft_adj[0]["Ex"]
        Eyga = dft_adj[0]["Ey"]
        Ezga = dft_adj[0]["Ez"]

        # plt.imshow(Eyga[0].real)
        # plt.show()

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
                    print("calculating 3D gradient")
                    gradient_x = -np.sum(1j * omega_ * phase_fact * eps0 * Exg[w] * Exga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis=2)
                    gradient_y = -np.sum(1j * omega_ * phase_fact * eps0 * Eyg[w] * Eyga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis=2)
                    gradient_z = -np.sum(1j * omega_ * phase_fact * eps0 * Ezg[w] * Ezga[w] * self.sim.step_size * (self.design_region[0]["ri max"] ** 2 - self.design_region[0]["ri min"] ** 2), axis=2)
                    
                    # plt.imshow(gradient_x.real)
                    # plt.savefig("gx.png")
                    # plt.close()
                    # plt.imshow(gradient_y.real)
                    # plt.savefig("gy.png")
                    # plt.close()
                    # plt.imshow(gradient_z.real)
                    # plt.savefig("gz.png")
                    # plt.close()

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


def indicator_solid(x, c, filter_f, threshold_f, resolution):
    '''Calculates the indicator function for the void phase needed for minimum length optimization [1].

    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_e : float
        Erosion threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.

    Returns
    -------
    array_like
        Indicator value

    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = jnp.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0] * resolution) ** 2 + (gradient_filtered_field[1] * resolution) ** 2
    if grad_mag.ndim != 2:
        raise ValueError("The gradient fields must be 2 dimensional. Check input array and filter functions.")
    I_s = design_field * jnp.exp(-c * grad_mag)
    return I_s

def constraint_solid(x, c, eta_e, filter_f, threshold_f, resolution):
    '''Calculates the constraint function of the solid phase needed for minimum length optimization [1].

    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_e : float
        Erosion threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.

    Returns
    -------
    float
        Constraint value

    Example
    -------
    >> g_s = constraint_solid(x,c,eta_e,filter_f,threshold_f) # constraint
    >> g_s_grad = grad(constraint_solid,0)(x,c,eta_e,filter_f,threshold_f) # gradient

    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x)
    I_s = indicator_solid(x.reshape(filtered_field.shape), c, filter_f, threshold_f, resolution).flatten()
    return jnp.mean(I_s * jnp.minimum(filtered_field.flatten() - eta_e, 0) ** 2)


def indicator_void(x, c, filter_f, threshold_f, resolution):
    '''Calculates the indicator function for the void phase needed for minimum length optimization [1].

    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_d : float
        Dilation threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.

    Returns
    -------
    array_like
        Indicator value

    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x).reshape(x.shape)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = jnp.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0] * resolution) ** 2 + (gradient_filtered_field[1] * resolution) ** 2
    if grad_mag.ndim != 2:
        raise ValueError("The gradient fields must be 2 dimensional. Check input array and filter functions.")
    return (1 - design_field) * jnp.exp(-c * grad_mag)


def constraint_void(x, c, eta_d, filter_f, threshold_f, resolution):
    '''Calculates the constraint function of the void phase needed for minimum length optimization [1].

    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_d : float
        Dilation threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.

    Returns
    -------
    float
        Constraint value

    Example
    -------
    >> g_v = constraint_void(p,c,eta_d,filter_f,threshold_f) # constraint
    >> g_v_grad = tensor_jacobian_product(constraint_void,0)(p,c,eta_d,filter_f,threshold_f,g_s) # gradient

    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''

    filtered_field = filter_f(x)
    I_v = indicator_void(x.reshape(filtered_field.shape), c, filter_f, threshold_f, resolution).flatten()
    return jnp.mean(I_v * jnp.minimum(eta_d - filtered_field.flatten(), 0) ** 2)
