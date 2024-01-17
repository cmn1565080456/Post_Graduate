import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
import cv2
import math

# several sub-functions goes here
def EvenFlip(A):
    temp = np.concatenate((A,np.flip(A)),axis=1)
    AA = np.concatenate((temp,np.flip(temp)),axis=0)
    return AA

def ZeroPadding(A):
    AA = np.concatenate((np.concatenate((A,np.zeros_like(A)),axis=1),np.concatenate((np.zeros_like(A),np.zeros_like(A)),axis=1)),axis=0)

# ----------------------------------------------------------------------------------------------- #
def TIE_solver_meanI(dIdz, I0z, pixelsize, k, r, thr, method):
    # existing codes in matlab goes here
    if thr is None:
        thr = 0.01

    if r is None:
        r = np.finfo(float).eps

    # Check method
    J = -k * dIdz

    if method.lower() == 'dct':
        # EvenFlip Matrices
        J = EvenFlip(J)
        I0z = EvenFlip(I0z)
    elif method.lower() == 'zeropaddingfft':
        # MeanPad Matrices
        J = ZeroPadding(J)
        I0z = ZeroPadding(I0z)
    elif method.lower() == 'fft':
        # Do nothing
        pass
    else:
        raise ValueError('Method can only be "FFT", "DCT", or "ZeroPaddingFFT" for now')

    # Coordinates in frequency domain
    Ny, Nx = J.shape
    nx = np.arange(1, Nx + 1)
    ny = np.arange(1, Ny + 1)
    u = ((nx - 1) - Nx / 2) / (pixelsize * Nx)
    v = ((ny - 1) - Ny / 2) / (pixelsize * Ny)
    fx, fy = np.meshgrid(u, v)
    fx = fftshift(fx)
    fy = fftshift(fy)

    # Terms for derivative calculation in frequency domain
    Cx = 2 * 1j * np.pi * fx
    Cy = 2 * 1j * np.pi * fy

    # Calculate Psi(optional), the main purpose is to calculate FPsi
    FJ = fft2(J)
    FPsi = FJ * (Cx * Cx + Cy * Cy) / (r + (Cx * Cx + Cy * Cy) ** 2)
    Psi = np.real(ifft2(FPsi))
    if method.lower() == 'dct':
        Psi = Psi[:Ny // 2, :Nx // 2]  # optional result
    elif method.lower() == 'zeropaddingfft':
        Psi = Psi[:Ny // 2, :Nx // 2]  # optional result

    # Calculate phi
    # Set the intensity out of aperture as the mean value of intensity inside.
    OutAper = I0z < thr * np.max(np.max(I0z))
    Im = I0z.copy()
    Im[OutAper] = np.mean(I0z[~OutAper])

    # From (dPsidx, dPsidy) to (dphidx, dphidy)
    dPsidx = np.real(ifft2(FPsi * Cx))
    dPsidy = np.real(ifft2(FPsi * Cy))
    dphidx = dPsidx / Im
    dphidy = dPsidy / Im

    # Calculate the laplace of phi
    Fdphidx = fft2(dphidx)
    Fdphidy = fft2(dphidy)

    Fphi = (Fdphidx * Cx + Fdphidy * Cy) * (Cx * Cx + Cy * Cy) / (r + (Cx * Cx + Cy * Cy) ** 2)
    phi = np.real(ifft2(Fphi))
    if method.lower() == 'dct':
        phi = phi[:Ny // 2, :Nx // 2]  # result
    elif method.lower() == 'zeropaddingfft':
        phi = phi[:Ny // 2, :Nx // 2]  # result

    Fdphidx = Fphi * Cx
    Fdphidy = Fphi * Cy
    dphidx = np.real(ifft2(Fdphidx))
    dphidy = np.real(ifft2(Fdphidy))

    dPsidx = I0z * dphidx
    dPsidy = I0z * dphidy

    FdPsidx = fft2(dPsidx)
    FdPsidy = fft2(dPsidy)

    Fd2Psidx2 = FdPsidx * Cx
    Fd2Psidy2 = FdPsidy * Cy

    d2Psidx2 = np.real(ifft2(Fd2Psidx2))
    d2Psidy2 = np.real(ifft2(Fd2Psidy2))

    laplacePsi = d2Psidx2 + d2Psidy2

    dIdz_est = laplacePsi / (-k)
    if method.lower() == 'dct':
        dIdz_est = dIdz_est[:Ny // 2, :Nx // 2]  # optional result
    elif method.lower() == 'zeropaddingfft':
        dIdz_est = dIdz_est[:Ny // 2, :Nx // 2]

# ---------------------------------------------------------------------------- #
def Numerical_Propagation(U0, z, pixelsize, wavelength, method):
    # existing code for Numerical_Propagation function in MATLAB goes here.
    N,M = U0.shape
    x = np.arange(1, M + 1)
    y = np.arange(1, N + 1)

    L0X = pixelsize * M
    L0Y = pixelsize * N

    k = 2 * np.pi / wavelength

    u = wavelength * (-M / L0X / 2 + 1 / L0X * (x - 1))
    v = wavelength * (-N / L0Y / 2 + 1 / L0Y * (y - 1))

    uu, vv = np.meshgrid(u, v)

    FU0 = np.fft.fftshift(np.fft.fft2(U0))

    if method == 'Angular Spectrum':
        H = np.exp(1j * k * z * np.sqrt(1 - uu ** 2 - vv ** 2))  # Angular Spectrum method
    elif method == 'Fresnel':
        H = np.exp(1j * k * z * (1 - (uu ** 2 + vv ** 2) / 2))  # Fresnel method
    else:
        raise ValueError('Type of transfer function must be <Angular Spectrum> or <Fresnel>')

    Uz = np.fft.ifft2(np.fft.fftshift(FU0 * H))

    return Uz

# ----------------------------------------------------------------------- #
def Estimate_dIdz(U0, z, pixelsize, wavelength, method):
    # existing code for Estimate_dIdz goes here
    N,M = U0.shape

    x = np.arange(1, M + 1)
    y = np.arange(1, N + 1)

    L0X = pixelsize * M
    L0Y = pixelsize * N

    k = 2 * np.pi / wavelength

    u = wavelength * (-M / L0X / 2 + 1 / L0X * (x - 1))
    v = wavelength * (-N / L0Y / 2 + 1 / L0Y * (y - 1))

    uu, vv = np.meshgrid(u, v)

    FU0 = np.fft.fftshift(np.fft.fft2(U0))

    if method == 'Angular Spectrum':
        H = np.exp(1j * k * z * np.sqrt(1 - uu ** 2 - vv ** 2))  # Angular Spectrum method
    elif method == 'Fresnel':
        H = np.exp(1j * k * z * (1 - (uu ** 2 + vv ** 2) / 2))  # Fresnel method
    else:
        raise ValueError('Type of transfer function must be <Angular Spectrum> or <Fresnel>')

    Uz = np.fft.ifft2(np.fft.fftshift(FU0 * H))

    return Uz


def TIEiter(dIdz, I0z, pixelsize, k, RegPara, IntThr, method, MaxIterNum, JudgeFlag, StdThr, NumerProp=None):
    if JudgeFlag is None:
        JudgeFlag = 'delta_dIdzStd'

    # TIE_solver_meanI
    phi_DCT = TIE_solver_meanI(dIdz, I0z, pixelsize, k, RegPara, IntThr, method)

    # Iterative Compensation
    IterNum = np.arange(1, MaxIterNum + 1)
    delta_dIdzStd = np.full_like(IterNum, np.nan)
    phi_compStd = np.full_like(IterNum, np.nan)

    if NumerProp is not None:
        Ny, Nx = I0z.shape
        dz = NumerProp['dz']
        wavelength = NumerProp['lambda']

    for num in IterNum:
        if NumerProp is not None:
            # Expand the Matrices for Numerical Propagation.
            Indy = slice(Ny//2, Ny//2*3)
            Indx = slice(Nx//2, Nx//2*3)

            I0z_est = np.zeros((2*Ny, 2*Nx))
            I0z_est[Indy, Indx] = I0z

            phi_DCT_est = np.zeros((2*Ny, 2*Nx))
            phi_DCT_est[Indy, Indx] = phi_DCT

            # Estimated complex amplitude of optical field in focus.
            U0z_est = np.sqrt(I0z_est) * np.exp(1j * phi_DCT_est)

            # Numerical propagation for defocused images.
            Unz_est = Numerical_Propagation(U0z_est, -dz, pixelsize, wavelength, 'Angular Spectrum')
            Upz_est = Numerical_Propagation(U0z_est, dz, pixelsize, wavelength, 'Angular Spectrum')

            Inz_est = np.abs(Unz_est)**2
            Ipz_est = np.abs(Upz_est)**2

            # Crop the Matrices.
            Inz_est = Inz_est[Indy, Indx]
            Ipz_est = Ipz_est[Indy, Indx]

            # Estimated dIdz.
            dIdz_est = (Ipz_est - Inz_est) / (2 * dz)
        else:
            # Estimate dIdz with TIE relations.
            dIdz_est = Estimate_dIdz(phi_DCT, I0z, pixelsize, k)

        # Update delta_dIdz.
        delta_dIdz = dIdz - dIdz_est
        # Check Std.
        delta_dIdzStd[num - 1] = np.std(delta_dIdz[~np.isnan(delta_dIdz)])

        # TIE_solver_meanI
        phi_comp = TIE_solver_meanI(delta_dIdz, I0z, pixelsize, k, RegPara, IntThr, method)
        # Check Std.
        phi_compStd[num - 1] = np.std(phi_comp[~np.isnan(phi_comp)])

        if JudgeFlag.lower() == 'delta_dIdzstd':
            Judge = delta_dIdzStd[num - 1] < StdThr
        elif JudgeFlag.lower() == 'phi_compstd':
            Judge = phi_compStd[num - 1] < StdThr
        else:
            raise ValueError('Currently only allow "delta_dIdzStd" or "phi_compStd".')

        if Judge:
            break
        else:
            # Update to phi_DCT.
            phi_DCT = phi_DCT + phi_comp

    return phi_DCT, num, delta_dIdzStd, phi_compStd

# Add the main function to activate the formula





