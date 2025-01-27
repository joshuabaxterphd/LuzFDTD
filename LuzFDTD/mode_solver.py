import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
np.random.seed(1)
eps0 = 8.8541878128e-12 # permittivity of free space
mu0 = 1.256637062e-6 # permeability of free space
c0 = 2.99792458e8 # speed of light in vacuum
imp0 = np.sqrt(mu0/eps0) # impedance of free space

def Mode_Solver(Epsxx, Epsyy, Epszz, dx, dy, wavelength, mode_number, pad_factor=0.1, tol = 1e-10):
    eta0 = imp0

    def Derivative_Matrices_X(Nx, Ny):
        return sp.diags(-np.ones(Nx * Ny)) + sp.diags(np.ones(Nx * Ny - 1), 1)

    def Derivative_Matrices_Y(Nx, Ny):
        return sp.diags(-np.ones(Nx * Ny)) + sp.diags(np.ones(Nx * Ny - Nx), Nx)

    def Dirichlet(A, Nx, Ny):
        for i in range(1, Ny):
            A[i * Nx - 1, i * Nx] = 0
        return A

    Nx = Epsxx.shape[0]
    Ny = Epsxx.shape[1]

    num_padx = round(pad_factor * Nx)
    if Ny > 1:
        num_pady = round(pad_factor * Ny)
    else:
        num_pady = 0
    Epsxx = np.pad(Epsxx.copy(), ((num_padx, num_padx), (num_pady, num_pady)), mode='edge')
    Epsyy = np.pad(Epsyy.copy(), ((num_padx, num_padx), (num_pady, num_pady)), mode='edge')
    Epszz = np.pad(Epszz.copy(), ((num_padx, num_padx), (num_pady, num_pady)), mode='edge')

    Nx = Epsxx.shape[0]
    Ny = Epsxx.shape[1]

    omega = 2 * np.pi * c0 / wavelength
    wavevector = omega / c0
    # print(wavevector,dx,dy)
    dx_prime = dx * wavevector
    dy_prime = dy * wavevector
    inv_dx_prime = 1.0 / dx_prime
    inv_dy_prime = 1.0 / dy_prime

    Dxe = Derivative_Matrices_X(Nx, Ny).tocsr() * inv_dx_prime
    Dxe = Dirichlet(Dxe, Nx, Ny)
    if Ny > 1:
        Dye = Derivative_Matrices_Y(Nx, Ny).tocsr() * inv_dy_prime
    else:
        Dye = Dxe.copy() * 0
    Dxh = -Dxe.copy().transpose()
    Dyh = -Dye.copy().transpose()

    Epsxxdiag = sp.diags(Epsxx.transpose().flatten(), format='csc')
    Epsyydiag = sp.diags(Epsyy.transpose().flatten(), format='csc')
    Epszzdiag = sp.diags(Epszz.transpose().flatten(), format='csc')
    mudiag = sp.diags(np.ones(Nx * Ny), format='csc')
    muinv = la.inv(mudiag)
    Epszzinv = la.inv(Epszzdiag)
    # print(wavevector,dx,dy)

    P = sp.bmat([[Dxe @ Epszzinv @ Dyh, -(Dxe @ Epszzinv @ Dxh + mudiag)],
                 [(Dye @ Epszzinv @ Dyh + mudiag), -Dye @ Epszzinv @ Dxh]], format='csc')
    Q = sp.bmat(
        [[Dxh @ muinv @ Dye, -(Dxh @ muinv @ Dxe + Epsyydiag)], [(Dyh @ muinv @ Dye + Epsxxdiag), -Dyh @ muinv @ Dxe]],
        format='csc')
    OmSq = P @ Q
    v0 = np.random.rand(Nx * Ny * 2)
    vals, vecs = la.eigs(OmSq, which='SR', v0=v0, tol=tol)
    gamma_t = np.sqrt(vals)
    gamma = wavevector * gamma_t
    neff = gamma / (1j * wavevector)

    Efield = vecs[:, mode_number] / np.max(np.abs(vecs[:, mode_number]))

    # get sign of field max.
    arg = np.argmax(np.abs(Efield))
    Efield *= np.sign(np.real(Efield[arg]))
    Hfield = -Q * Efield / gamma_t[mode_number] / (1j * eta0)
    N = Nx * Ny
    Ex = Efield[:N].reshape(Nx, Ny, order='F')
    Ey = Efield[N:].reshape(Nx, Ny, order='F')
    Hx = Hfield[:N].reshape(Nx, Ny, order='F')
    Hy = Hfield[N:].reshape(Nx, Ny, order='F')

    # plt.imshow((np.abs(Ex)**2 + np.abs(Ey)**2).transpose(),cmap='jet');plt.colorbar(label = '$|E|^2$');plt.title('Mode Intensity')
    # plt.show()
    # plt.plot((np.abs(Ex)**2 + np.abs(Ey)**2))
    # plt.show()

    print("Mode index: " + str(np.real(neff[mode_number])))
    if Ny > 1:
        return np.real(neff[mode_number]), Ex[num_padx:-num_padx, num_pady:-num_pady], Ey[num_padx:-num_padx,num_pady:-num_pady], Hx[num_padx:-num_padx, num_pady:-num_pady], Hy[num_padx:-num_padx,num_pady:-num_pady]
    else:
        return np.real(neff[mode_number]), Ex[num_padx:-num_padx,:], Ey[num_padx:-num_padx,:], Hx[num_padx:-num_padx,:], Hy[num_padx:-num_padx,:]


if __name__ == "__main__":
    Sx = 3e-6
    Sy = 2.5e-6
    dx = 0.02e-6
    x = np.arange(-Sx/2, Sx/2, dx)
    y = np.arange(-Sy/2, Sy/2, dx)
    Nx = len(x)
    Ny = len(y)

    Epsxx = np.ones((Nx,Ny))
    Epsyy = np.ones((Nx,Ny))
    Epszz = np.ones((Nx,Ny))

    rect_x = 0.5e-6
    rect_y = 0.24e-6

    X,Y = np.meshgrid(x,y,indexing='ij')

    rect = (np.abs(X + 0.5 * dx) < rect_x/2) & (np.abs(Y) < rect_y/2)
    Epsxx[rect] *= 3.5 ** 2
    rect = (np.abs(X) < rect_x/2) & (np.abs(Y + 0.5 * dx) < rect_y/2)
    Epsyy[rect] *= 3.5 ** 2
    rect = (np.abs(X) < rect_x/2) & (np.abs(Y) < rect_y/2)
    Epszz[rect] *= 3.5 ** 2

    Mode_Solver(Epsxx, Epsyy, Epszz, dx, dx, 1.55e-6, 0, tol = 1e-5)