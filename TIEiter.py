# 迭代DCT算法的实现
import numpy as np
from TIE_solver_meanI import TIE_solver_meanI
from Numerical_Propagation import numerical_propagation
from TIE_Functions import Estimate_dIdz

def TIEiter(dIdz, I0z, pixelsize, k, RegPara, IntThr, method, MaxIterNum, JudgeFlag='delta_dIdzStd', StdThr=None, NumerProp=None):
    # Check nargin.
    if JudgeFlag is None:
        JudgeFlag = 'delta_dIdzStd'

    # TIE_solver_meanI
    phi_DCT = TIE_solver_meanI(dIdz, I0z, pixelsize, k, RegPara, IntThr, method)
    # Now the phi_DCT is a tuple
    phi_DCT = np.array(phi_DCT)

    # Iterative Compensation
    IterNum = np.arange(1, MaxIterNum + 1)
    delta_dIdzStd = np.full_like(IterNum, np.nan)
    phi_compStd = np.full_like(IterNum, np.nan)

    if NumerProp is not None:
        Ny, Nx = I0z.shape
        dz = NumerProp['dz']
        lambda_val = NumerProp['lambda']

    for num in IterNum:
        if NumerProp is not None:
            # Expand the Matrices for Numerical Propagation.
            Indy = slice(Ny//2, Ny//2*3)
            Indx = slice(Nx//2, Nx//2*3)

            I0z_est = np.zeros((2*Ny, 2*Nx))
            I0z_est[Indy, Indx] = I0z

            phi_DCT_est = np.zeros((2*Ny, 2*Nx))
            # phi_DCT_est[Indy, Indx] = phi_DCT[Indy, Indx]
            phi_DCT_est[Indy, Indx] = phi_DCT[0]
            # phi_DCT_est = phi_DCT


            # Estimated complex amplitude of optical field in focus.
            U0z_est = np.sqrt(I0z_est) * np.exp(1j * phi_DCT_est)

            # Numerical propagation for defocused images.
            Unz_est = numerical_propagation(U0z_est, -dz, pixelsize, lambda_val, 'Angular Spectrum')
            Upz_est = numerical_propagation(U0z_est, +dz, pixelsize, lambda_val, 'Angular Spectrum')

            Inz_est = np.abs(Unz_est)**2
            Ipz_est = np.abs(Upz_est)**2

            # Crop the Matrices.
            Inz_est = Inz_est[Indy, Indx]
            Ipz_est = Ipz_est[Indy, Indx]

            # Estimated dIdz.
            dIdz_est = (Ipz_est - Inz_est) / 2 / dz
        else:
            # Estimate dIdz with TIE relations.
            dIdz_est = Estimate_dIdz(phi_DCT, I0z, pixelsize, k)

        # Update delta_dIdz.
        delta_dIdz = dIdz - dIdz_est
        # Check Std.
        # delta_dIdzStd[num - 1] = np.std(delta_dIdz[~np.isnan(delta_dIdz)])
        if not np.any(np.isnan(delta_dIdz)):
            delta_dIdzStd[num - 1] = np.std(delta_dIdz)
        else:
            # 处理包含NaN的情况，可以选择赋予一个特殊值或执行其他操作
            delta_dIdzStd[num - 1] = 0  # 或者选择一个合适的默认值

        # TIE_solver_meanI
        phi_comp = TIE_solver_meanI(delta_dIdz, I0z, pixelsize, k, RegPara, IntThr, method)
        phi_comp = np.array(phi_comp)
        # Check Std.
        # phi_compStd[num - 1] = np.std(phi_comp[~np.isnan(phi_comp)])
        if not np.any(np.isnan(phi_compStd)):
            phi_compStd[num - 1] = np.std(phi_compStd)
        else:
            phi_compStd[num - 1] = 0


        # if JudgeFlag.lower() == 'delta_dIdzStd':
        if JudgeFlag == 'delta_dIdzStd':
            Judge = delta_dIdzStd[num - 1] < StdThr
        # elif JudgeFlag.lower() == 'phi_compStd':
        elif JudgeFlag == 'phi_compStd':
            Judge = phi_compStd[num - 1] < StdThr
        else:
            raise ValueError("Currently only allow 'delta_dIdzStd' or 'phi_compStd'.")

        if Judge:
            break
        else:
            # update to phi_DCT.
            phi_DCT = phi_DCT + phi_comp

    return phi_DCT, num, delta_dIdzStd, phi_compStd
