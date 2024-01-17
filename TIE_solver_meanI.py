import numpy as np

def TIE_solver_meanI(dIdz, I0z, pixelsize, k, r, thr, method):
    # Check nargin
    if thr is None:
        thr = 0.01

    if r is None:
        r = np.finfo(float).eps

    if method is None:
        method = 'DCT'

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
        pass  # Do nothing
    else:
        raise ValueError("Method can only be 'FFT', 'DCT', or 'ZeroPadding' for now")

    # Coordinates in frequency domain
    Ny, Nx = J.shape
    nx = np.arange(1, Nx + 1)
    ny = np.arange(1, Ny + 1)
    u = ((nx - 1) - Nx / 2) / (pixelsize * Nx)
    v = ((ny - 1) - Ny / 2) / (pixelsize * Ny)
    fx, fy = np.meshgrid(u, v)
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftshift(fy)

    # Terms for derivative calculation in frequency domain
    Cx = 2j * np.pi * fx
    Cy = 2j * np.pi * fy

    # Calculate Psi(optional), the main purpose is to calculate FPsi
    FJ = np.fft.fft2(J)
    FPsi = FJ * (Cx * Cx + Cy * Cy) / (r + (Cx * Cx + Cy * Cy)**2)
    Psi = np.real(np.fft.ifft2(FPsi))
    if method.lower() == 'dct':
        Psi = Psi[0:Ny // 2, 0:Nx // 2]  # optional result
    elif method.lower() == 'zeropaddingfft':
        Psi = Psi[0:Ny // 2, 0:Nx // 2]  # optional result

    # Calculate phi
    OutAper = I0z < thr * np.max(np.max(I0z))
    Im = I0z.copy()
    Im[OutAper] = np.mean(I0z[~OutAper])

    dPsidx = np.real(np.fft.ifft2(FPsi * Cx))
    dPsidy = np.real(np.fft.ifft2(FPsi * Cy))
    dphidx = dPsidx / Im
    dphidy = dPsidy / Im

    Fdphidx = np.fft.fft2(dphidx)
    Fdphidy = np.fft.fft2(dphidy)
    Fphi = (Fdphidx * Cx + Fdphidy * Cy) * (Cx * Cx + Cy * Cy) / (r + (Cx * Cx + Cy * Cy)**2)
    phi = np.real(np.fft.ifft2(Fphi))
    if method.lower() == 'dct':
        phi = phi[0:Ny // 2, 0:Nx // 2]  # result
    elif method.lower() == 'zeropaddingfft':
        phi = phi[0:Ny // 2, 0:Nx // 2]  # result

    Fdphidx = Fphi * Cx
    Fdphidy = Fphi * Cy
    dphidx = np.real(np.fft.ifft2(Fdphidx))
    dphidy = np.real(np.fft.ifft2(Fdphidy))

    dPsidx = I0z * dphidx
    dPsidy = I0z * dphidy

    FdPsidx = np.fft.fft2(dPsidx)
    FdPsidy = np.fft.fft2(dPsidy)

    Fd2Psidx2 = FdPsidx * Cx
    Fd2Psidy2 = FdPsidy * Cy

    d2Psidx2 = np.real(np.fft.ifft2(Fd2Psidx2))
    d2Psidy2 = np.real(np.fft.ifft2(Fd2Psidy2))

    laplacePsi = d2Psidx2 + d2Psidy2

    dIdz_est = laplacePsi / (-k)
    if method.lower() == 'dct':
        dIdz_est = dIdz_est[0:Ny // 2, 0:Nx // 2]  # optional result
    elif method.lower() == 'zeropaddingfft':
        dIdz_est = dIdz_est[0:Ny // 2, 0:Nx // 2]  # optional result

    return phi, dIdz_est, Psi


def EvenFlip(A):
    temp = np.concatenate((A, np.fliplr(A)), axis=1)
    AA = np.concatenate((temp, np.flipud(temp)), axis=0)
    return AA


def ZeroPadding(A):
    AA = np.zeros((2 * A.shape[0], 2 * A.shape[1]))
    AA[:A.shape[0], :A.shape[1]] = A
    return AA