import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def DCT_TIE_solver(dIdz, I0, pixelsize, k, r, threshold, reduced_region_size):
    dIdz_double = np.hstack((dIdz, np.fliplr(dIdz)))
    dIdz_double = np.vstack((dIdz_double, np.flipud(dIdz_double)))

    I0_double = np.hstack((I0, np.fliplr(I0)))
    I0_double = np.vstack((I0_double, np.flipud(I0_double)))

    M, N = dIdz_double.shape
    n = np.arange(1, N+1)
    m = np.arange(1, M+1)
    L0X = pixelsize * M
    L0Y = pixelsize * N

    v = (-M / (2 * L0X) + 1 / L0X * (m - 1))
    u = (-N / (2 * L0Y) + 1 / L0Y * (n - 1))

    uu, vv = np.meshgrid(u, v)

    kdIdz_double = -k * dIdz_double

    Fleft = fft2(kdIdz_double)

    Fphi = fftshift(Fleft) * (-4 * np.pi ** 2 * (uu ** 2 + vv ** 2)) / (r + (-4 * np.pi ** 2 * (uu ** 2 + vv ** 2)) ** 2)
    bigphi = np.real(ifft2(fftshift(Fphi)))

    if reduced_region_size == 0:
        Fbigphi = fft2(bigphi)

        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * uu)
        dxbigphi = np.real(ifft2(fftshift(Fphi)))

        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * vv)
        dybigphi = np.real(ifft2(fftshift(Fphi)))

        I0_double[I0_double < threshold * np.max(I0_double)] = threshold * np.max(I0_double)

        dxbigphi = dxbigphi / I0_double
        dybigphi = dybigphi / I0_double

        Fbigphi = fft2(dxbigphi)
        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * uu)
        dxdxbigphi = np.real(ifft2(fftshift(Fphi)))

        Fbigphi = fft2(dybigphi)
        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * vv)
        dydybigphi = np.real(ifft2(fftshift(Fphi)))

        ddphi = dxdxbigphi + dydybigphi

        Fleft = fft2(ddphi)
        Fphi = fftshift(Fleft) * (-4 * np.pi ** 2 * (uu ** 2 + vv ** 2)) / (
                r + (-4 * np.pi ** 2 * (uu ** 2 + vv ** 2)) ** 2)

        phi = np.real(ifft2(fftshift(Fphi)))
        phi = phi[:M // 2, :N // 2]

    else:
        bigphi = bigphi[:M // 2, :N // 2]
        bigphi = bigphi[reduced_region_size:-reduced_region_size, reduced_region_size:-reduced_region_size]

        bigphi_double = np.hstack((bigphi, np.fliplr(bigphi)))
        bigphi_double = np.vstack((bigphi_double, np.flipud(bigphi_double)))

        I0 = I0[reduced_region_size:-reduced_region_size, reduced_region_size:-reduced_region_size]
        I0_double = np.hstack((I0, np.fliplr(I0)))
        I0_double = np.vstack((I0_double, np.flipud(I0_double)))

        M, N = I0_double.shape
        n = np.arange(1, N + 1)
        m = np.arange(1, M + 1)
        L0X = pixelsize * M
        L0Y = pixelsize * N

        v = (-M / (2 * L0X) + 1 / L0X * (m - 1))
        u = (-N / (2 * L0Y) + 1 / L0Y * (n - 1))

        uu, vv = np.meshgrid(u, v)

        Fbigphi = fft2(bigphi_double)

        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * uu)
        dxbigphi = np.real(ifft2(fftshift(Fphi)))

        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * vv)
        dybigphi = np.real(ifft2(fftshift(Fphi)))

        I0[I0 < threshold * np.max(I0)] = threshold * np.max(I0)

        dxbigphi = dxbigphi / I0_double
        dybigphi = dybigphi / I0_double

        Fbigphi = fft2(dxbigphi)
        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * uu)
        dxdxbigphi = np.real(ifft2(fftshift(Fphi)))

        Fbigphi = fft2(dybigphi)
        Fphi = fftshift(Fbigphi) * (2 * 1j * np.pi * vv)
        dydybigphi = np.real(ifft2(fftshift(Fphi)))

        ddphi = dxdxbigphi + dydybigphi

        Fleft = fft2(ddphi)
        Fphi = fftshift(Fleft) * (-4 * np.pi ** 2 * (uu ** 2 + vv ** 2)) / (
                r + (-4 * np.pi ** 2 * (uu ** 2 + vv ** 2)) ** 2)

        phi = np.real(ifft2(fftshift(Fphi)))
        phi = phi[:M // 2, :N // 2]

    return phi

# Debug usage:
# Provide appropriate values for the parameters
dIdz = np.random.rand(256, 256)  # Replace with actual data
I0 = np.random.rand(256, 256)  # Replace with actual data
pixelsize = 0.1
k = 2 * np.pi / 0.5  # Replace with actual value
r = 1e-6  # Replace with actual value
threshold = 0.01
reduced_region_size = 10

phi_result = DCT_TIE_solver(dIdz, I0, pixelsize, k, r, threshold, reduced_region_size)
print(phi_result)
