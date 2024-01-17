import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def numerical_propagation(U0, z, pixelsize, wavelength, method):
    """
    Numerical propagation of a complex field to another plane at a given distance.

    Parameters:
        U0 (ndarray): Original complex field.
        z (float): Propagation distance.
        pixelsize (float): Pixelsize (mm).
        wavelength (float): Wavelength (mm).
        method (str): Type of transfer function used ('Angular Spectrum' or 'Fresnel').

    Returns:
        Uz (ndarray): Complex field after propagation.
    """
    # Get the size of the input field
    N, M = U0.shape

    # Generate spatial frequency coordinates
    x = np.arange(1, M + 1)
    y = np.arange(1, N + 1)

    L0X = pixelsize * M
    L0Y = pixelsize * N

    k = 2 * np.pi / wavelength

    u = wavelength * (-M / (2 * L0X) + 1 / L0X * (x - 1))
    v = wavelength * (-N / (2 * L0Y) + 1 / L0Y * (y - 1))

    uu, vv = np.meshgrid(u, v)

    # Perform FFT on the input field and shift the zero frequency component to the center
    FU0 = fftshift(fft2(U0))

    # Generate the transfer function based on the selected method
    if method == 'Angular Spectrum':
        H = np.exp(1j * k * z * np.sqrt(1 - uu**2 - vv**2))
    elif method == 'Fresnel':
        H = np.exp(1j * k * z * (1 - (uu**2 + vv**2) / 2))
    else:
        raise ValueError('Type of transfer function must be <Angular Spectrum> or <Fresnel>')

    # Perform inverse FFT to obtain the propagated complex field
    Uz = ifft2(fftshift(FU0 * H))

    return Uz

