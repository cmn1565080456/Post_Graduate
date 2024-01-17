import numpy as np
import matplotlib.pyplot as plt
import cv2
from DCT_TIE_solver import DCT_TIE_solver
from Generate_test_image import generate_maps
from Numerical_Propagation import numerical_propagation
from TIEiter import TIEiter



# -------------------------Intensity & Phase------------------------------ #
pixelsize_0 = 2e-3
lambda_0 = 633e-6
k_0 = 2*(np.pi)/lambda_0
a_left = 79
a_right = 178
a_up = 79
a_down = 178

intensity_map, phase_map, aperture = generate_maps(pixelsize=pixelsize_0,aperture_left=a_left,aperture_right=a_right,aperture_up=a_up,aperture_down=a_down)

plt.figure(figsize=(10, 5))
# figure_size_appropriated
plt.subplot(1, 3, 1)
plt.imshow(intensity_map, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Intensity Map')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(phase_map, cmap='hsv')
plt.xticks([])
plt.yticks([])
plt.title('Phase Map')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(aperture, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Aperture')
plt.colorbar()

plt.show()
intensity_map = aperture
#--------------------------Numerical Propagation to Generate Defocus Image------------------------------#
# Numerical propagation to generate defocused images
U0 = np.sqrt(intensity_map) * np.exp(1j * phase_map)
dz = 50e-3  # defocus distance 10um

Uz = numerical_propagation(U0, dz, pixelsize_0, lambda_0, 'Angular Spectrum')
Iz = np.abs(Uz)**2
U_z = numerical_propagation(U0, -dz, pixelsize_0, lambda_0, 'Angular Spectrum')
I_z = np.abs(U_z)**2

STD = 0.0005
Intensity = intensity_map + np.random.randn(*intensity_map.shape) * STD
Iz = Iz + np.random.randn(*Iz.shape) * STD
I_z = I_z + np.random.randn(*I_z.shape) * STD

dIdz = (Iz - I_z) / (2 * dz)  # axial intensity derivative

# Visualization
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
fig.tight_layout()
axs[0].imshow(I_z, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('I_z')

axs[1].imshow(Intensity, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('I')

axs[2].imshow(Iz, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('Iz')

axs[3].imshow(dIdz, cmap='gray')
axs[3].set_title('dIdz')

plt.show()

# ----------------------------------Solve TIE with iteration function----------------------------- #
# Solve TIE with iteration function.

RegPara = np.finfo(float).eps
IntThr = 0.01
method = 'dct'
JudgeFlag = 'delta_dIdzStd'
StdThr = 0.0001
MaxIterNum = 10  # 原始的最大迭代次数为5

NumerProp = {'dz': dz, 'lambda': lambda_0}

I0z = Intensity

phi_iter, num_iter, delta_dIdzStd, phi_compStd = TIEiter(dIdz, I0z, pixelsize_0, k_0, RegPara, IntThr, method, MaxIterNum, JudgeFlag, StdThr, NumerProp)

# Error Calculation.
phiErr_iter = phi_iter - phase_map
phiErr_iter = phiErr_iter - np.nanmean(phiErr_iter)

# -----------------------------------Crop each Image according to the Aperture Edge-------------------- #
more = 0

Iz = Iz[ a_up - more :  a_down + more,  a_left - more :  a_right + more]
I_z = I_z[ a_up - more :  a_down + more,  a_left - more :  a_right + more]
I = Intensity[ a_up - more :  a_down + more,  a_left - more :  a_right + more]
dIdz = dIdz[ a_up - more :  a_down + more,  a_left - more :  a_right + more]

phase_map =  phase_map[ a_up :  a_down,  a_left :  a_right]

phi_iter = phi_iter[0][ a_up - more :  a_down + more,  a_left - more :  a_right + more]
phiErr_iter = phiErr_iter[ a_up - more :  a_down + more,  a_left - more :  a_right + more]
phiErr_iter = phiErr_iter - np.nanmean(phiErr_iter[~np.isnan(phiErr_iter)])

plt.figure()
plt.imshow(phi_iter, cmap='jet')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(phase_map, cmap='jet')
plt.colorbar()
plt.show()

# -----------------------------Solve TIE Function-----------------------------------#
Phi_DCT = DCT_TIE_solver(dIdz, I, pixelsize_0, k_0, np.finfo(float).eps, 0.01, more)
# Phi_FFT = FFT_TIE_solver(dIdz, I, pixelsize, k, np.finfo(float).eps, 0.01)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs[0, 0].imshow(Phi_DCT, cmap='jet')
axs[0, 0].set_title('Phase retrieved by DCT-TIE')
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])

# axs[0, 1].imshow(Phi_FFT, cmap='jet')
axs[0, 1].set_title('Phase retrieved by Iteration-TIE')
axs[0, 1].imshow(phi_iter, cmap='jet')
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])

ax0 = axs[1, 0]
# ax0.imshow(phase_map, cmap='jet')
im0 = ax0.imshow(phase_map, cmap='jet')
ax0.set_title('True Phase')
ax0.set_xticks([])
ax0.set_yticks([])
cbar0 = plt.colorbar(im0, ax=ax0)
cbar0.set_label('Parameter')
# err = Phi_DCT - phase_map
# axs[1, 1].imshow(err - np.nanmean(err), cmap='jet')
# axs[1, 1].set_title('DCT-TIE error')

# 创建第二个子图
err = Phi_DCT - phase_map
ax = axs[1, 1]
im = ax.imshow(err - np.nanmean(err), cmap='jet')
ax.set_title('DCT-TIE error')
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
# 添加 colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Error')

plt.show()




