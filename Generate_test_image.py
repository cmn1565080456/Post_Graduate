import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def generate_maps(pixelsize, aperture_left, aperture_right,aperture_up, aperture_down):
    Intensity = np.zeros((256, 256))
    Phase = np.zeros((256, 256))

    # Generate phase map using a complex quadratic phase function
    for ii in range(256):
        for jj in range(256):
            x = (ii - 129) * pixelsize
            y = (jj - 129) * pixelsize
            Phase[ii, jj] = 0.82 + (-0.7 * x + x**2 * 10 - y**2 * 10 + 2 * y)

    # Generate intensity map using a Gaussian distribution
    for ii in range(256):
        for jj in range(256):
            x = (ii - 129) * pixelsize
            y = (jj - 129) * pixelsize
            sigma = 0.18
            Intensity[ii, jj] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Create an aperture (rectangular light stop) for DCT algorithm
    I = np.zeros((256, 256))
    I[aperture_up:aperture_down+1, aperture_left:aperture_right+1] = 1.0



    return Intensity, Phase, I

# # Example of how to call the function and plot the results
# intensity_map, phase_map, aperture = generate_maps(pixelsize=1.0)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(intensity_map, cmap='gray')
# plt.title('Intensity Map')
# plt.colorbar()
#
# plt.subplot(1, 3, 2)
# plt.imshow(phase_map, cmap='hsv')
# plt.title('Phase Map')
# plt.colorbar()
#
# plt.subplot(1, 3, 3)
# plt.imshow(aperture, cmap='gray')
# plt.title('Aperture')
# plt.colorbar()
#
# plt.show()
