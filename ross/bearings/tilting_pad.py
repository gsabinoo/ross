# from cmath import sin
import numpy as np
import scipy 

from scipy.optimize import fmin, Bounds, minimize

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import os, sys
from pathlib import Path

sys.path.append(str(Path(os.path.realpath(__file__)).parent.parent.parent))

#from lmest_rotor.surrogates.models.tilting_SR import Tilting_SR

import time

from timeit import default_timer 

class Tilting:
    
    """ This class calculates the pressure and temperature fields, equilibrium
    position of a tilting-pad thrust bearing. It is also possible to obtain the
    stiffness and damping coefficients.
    
    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    Rs : float
    Rotor radius. Default unit is meter
    Rp : float
    Individual Pad radius. Default unit is meter.
    npad : integer
    Number of pads.
    tpad : float  
    Pad thickness. Default unit is meter.
    betha_p : float
    Individual Pad angle. Default unit is degrees.
    rp_pad : float
    Pivot offset.
    L : float
    Bearing length. Default unit is meter.
    Cr : float
    Radial clearance. Default unit is meter.
    
    Operating conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operating conditions of the bearing
    speed : float
    Rotor rotating speed. Default unit is rad/s
    load : Float
    Axial load. The unit is Newton.
    Tcub : Float
    Oil tank temperature. The unit is °C
    x0  : array
    Initial Equilibrium Position

    
    Fluid properties
    ^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    
    lubricant : str
    Lubricant type. Can be:
    - 'ISOVG32'
    - 'ISOVG46'
    - 'ISOVG68'
    With it, we get:
    rho :  Fluid specific mass. Default unit is kg/m^3.
    kt  :  Fluid thermal conductivity. The unit is J/(s*m*°C).
    cp  :  Fluid specific heat. The unit is J/(kg*°C).
        
        
    k1, k2, and k3 : float
    Oil coefficients for the viscosity interpolation.

    Mesh discretization
    ^^^^^^^^^^^^^^^^^^^
    Describes the discretization of the fluid film
    nX : integer
    Number of volumes along the X direction.
    nZ : integer
    Number of volumes along the Z direction.
    Z1 : float
    Initial dimensionless X.
    Z2 : float
    Initial dimensionless Z.
    
    
    Returns
    -------
    Pmax : float
    Maximum pressure. The unit is Pa.
    Tmax : float
    Maximum temperature. The unit is °C.
    h0 : float
    oil film thickness at the pivot point. The unit is m.
    hmax : float
    maximum oil film thickness. The unit is m.
    hmin : float
    minimum oil film thickness. The unit is m.
    K : float
    bearing stiffness coefficient. The unit is N/m.
    C : float
    bearing damping coefficient. The unit is N.s/m.
    PPdim : array
    pressure field. The unit is Pa.
    XH,YH : array
    mesh grid. The uni is m.
    TT : array
    temperature field. The unit is °C.
    ecc : float
    Eccentricity.

    References
    ----------
    .. [1] BARBOSA, J.S. Analise de Modelos Termohidrodinamicos para Mancais de unidades geradoras Francis. 2016. Dissertacao de Mestrado. Universidade Federal de Uberlandia, Uberlandia. ..
    .. [2] HEINRICHSON, N.; SANTOS, I. F.; FUERST, A., The Influence of Injection Pockets on the Performance of Tilting Pad Thrust Bearings Part I Theory. Journal of Tribology, 2007. .. 
    .. [3] NICOLETTI, R., Efeitos Termicos em Mancais Segmentados Hibridos Teoria e Experimento. 1999. Dissertacao de Mestrado. Universidade Estadual de Campinas, Campinas. ..
    .. [4] LUND, J. W.; THOMSEN, K. K. A calculation method and data for the dynamic coefficients of oil lubricated journal bearings. Topics in fluid film bearing and rotor bearing system design and optimization, n. 1000118, 1978. ..
    Attributes
    ----------
    """

    def __init__(
        self,
        Rs,    # Rotor radius
        npad,  # Number of pads
        Rp,    # Pad radius
        tpad,  # Pad thickness
        betha_p, # Pad angle
        rp_pad, # Pivot offset
        L,     # Bearing length
        lubricant,
        Tcub,  # Oil tank temperature
        nX, # n° volumes x
        nZ, # n° volumes z
        Cr, # Radial Clearance [m]
        Cr_ref,
        sigma,  # Array
        speed,
        choice_CAIMP, # Method (calculating or imposing equilibrium position)
        Coefs_D=None,
    ):

        self.Rs = Rs 
        self.npad = npad
        self.Rp = Rp
        self.tpad = tpad
        self.betha_p = betha_p * (np.pi / 180)   # Pad angle [rad]
        self.rp_pad = rp_pad 
        self.Cr = Cr 
        self.Cr_ref = np.array(Cr_ref)
        self.L = L
        self.Tcub = Tcub
        T0 = Tcub
        self.T0 = self.Tcub # Reference temperature [Celsius]
        self.lubricant = lubricant

        self.nX = nX    # circumferential direction
        self.nZ = nZ    # axial direction
        self.Z1 = -0.5
        self.Z2 = 0.5
        self.sigma = np.array(sigma) * (np.pi/180) #rad
        TETA1= - ( self.rp_pad ) * self.betha_p # initial coordinate in the TETA direction
        TETA2= ( 1-self.rp_pad ) * self.betha_p # final coordinate in the TETA direction
        self.TETA1 = TETA1
        self.TETA2 = TETA2
        self.dTETA = ( self.TETA2 - self.TETA1 ) / self.nX
        self.dZ = ( self.Z2 - self.Z1 ) / self.nZ
        self.dX = self.dTETA / self.betha_p
        self.XZ = np.zeros(self.nZ)
        self.XZ[0] = self.Z1 + 0.5 * self.dZ
        self.XTETA = np.zeros(self.nX)
        self.XTETA[0] = self.TETA1 + 0.5 * self.dTETA

        self.speed = speed * np.pi/30 # Convert rpm to rad/s
        self.choice_CAIMP = choice_CAIMP
        self.op_key = [*choice_CAIMP][0]

        self.Coefs_D = Coefs_D

        ##### Center shaft speed
        xpt = 0  
        ypt = 0
        self.xpt = self.ypt = xpt
        # --------------------------------------------------------------------------
        
        # Interpolation coefficients
        lubricant_properties = self.lub_selector()
        T_1 = lubricant_properties["temp1"]
        T_2 = lubricant_properties["temp2"]
        mi_1 = lubricant_properties["viscosity1"]
        mi_2 = lubricant_properties["viscosity2"]
        self.rho = lubricant_properties["lube_density"]
        self.Cp = lubricant_properties["lube_cp"]
        self.kt = lubricant_properties["lube_conduct"]

        self.b_b = np.log(mi_1/mi_2)*1/(T_1-T_2)
        self.a_a = mi_1/(np.exp(T_1*self.b_b))

        self.mi0 = self.a_a*np.exp(self.T0*self.b_b) #reference viscosity

        for ii in range(1, self.nZ):
            self.XZ[ii] = self.XZ[ii-1] + self.dZ

        for jj in range(1, self.nX):
            self.XTETA[jj] = self.XTETA[jj-1] +self.dTETA
        pass
        
        self.dimForca = 1/ ( self.Cr_ref ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L))
    
    def run(self):

        ############### Define parameters ##############

        self.PPdim = np.zeros((self.nZ,self.nX,self.npad))
        self.H0 = np.zeros((self.nZ,self.nX,self.npad))
        self.Hdim = np.zeros((self.nZ,self.nX,self.npad))
        self.H = np.zeros((self.nZ,self.nX))
        self.P = np.zeros((self.nZ,self.nX))
        self.PP = np.zeros((self.nZ, self.nX, self.npad))
        self.h_pivot = np.zeros((self.npad))
        self.TT_i = self.T0 * np.ones((self.nZ,self.nX,self.npad)) # Initial 3D - temperature field
        self.dPdX = np.zeros((self.nZ,self.nX))
        self.dPdZ = np.zeros((self.nZ,self.nX))
        self.Reyn = np.zeros((self.nZ,self.nX,self.npad))
        self.mi_turb = 1.3*np.ones((self.nZ,self.nX,self.npad)) # Turbulence coefficient
        self.Fx = np.zeros((self.npad))
        self.Fy = np.zeros((self.npad))
        self.Mj = np.zeros((self.npad))
        self.Fx_dim = np.zeros((self.npad))
        self.Fy_dim = np.zeros((self.npad))
        self.Mj_dim = np.zeros((self.npad)) 
        self.Mj_new = np.zeros((self.npad))
        self.F1 = np.zeros((self.npad))
        self.F1_new = np.zeros((self.npad))
        self.Fj_dim = np.zeros((self.npad))
        self.F2 = np.zeros((self.npad))
        self.F2_new = np.zeros((self.npad))
        self.P_bef = np.zeros((self.nZ,self.nX))
        

        if "print" in [*self.choice_CAIMP[self.op_key]] and "progress" in self.choice_CAIMP[self.op_key]["print"]:
            
            self.progress = True
                
        else:

            self.progress = False

        if "calc_EQ" in self.op_key:

            self.x0 = self.choice_CAIMP["calc_EQ"]["init_guess"]
            self.Wx = self.choice_CAIMP["calc_EQ"]["load"][0]
            self.Wy = self.choice_CAIMP["calc_EQ"]["load"][1]
            self.WX = self.Wx * ( self.Cr_ref[n_p] ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L)) # Loading - X direction [dimensionless]
            self.WY = self.Wy * ( self.Cr_ref[n_p] ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L)) # Loading - Y direction [dimensionless]

        elif "impos_EQ" in self.op_key:
            
            self.x0 = (self.choice_CAIMP["impos_EQ"]["ent_angle"])
            self.WX = self.WY = 0
    
        maxP, medP, maxT, medT, h_pivot0, ecc = self.p_and_t_solution()
        
        description = [
                    f"\n>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<\n",
                    f"      Pmax: {maxP}\n",
                    f"      Tmax: {maxT}\n",
                    f"      Eccentricity: {ecc}\n",
                    f"      h pivot: {h_pivot0}\n",
                    f">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<\n",
                ]   
                        
        for line in description:
            print(line[:-1])    

        if self.Coefs_D is not None:
        
            self.coeffs_din()

            if "show_coef" in self.Coefs_D and self.Coefs_D["show_coef"]:
                description = [
                            f"\n>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<\n",
                            f"      Kzz: {self.Kxx}\n",
                            f"      Kzx: {self.Kxy}\n",
                            f"      Kxz: {self.Kyx}\n",
                            f"      Kxx: {self.Kyy}\n",
                            f" =========================================\n",
                            f"      Czz: {self.Cxx}\n",
                            f"      Czx: {self.Cxy}\n",
                            f"      Cxz: {self.Cyx}\n",
                            f"      Cxx: {self.Cyy}\n",
                            f">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<\n",
                        ]   
                            
                for line in description:
                    print(line[:-1])   
       
    def p_and_t_solution(self):
        if "impos_EQ" in self.op_key:
            ang_rot = np.zeros(self.npad)
            momen_rot = np.zeros(self.npad)
            self.alpha_max_chut = np.zeros(self.npad)
            self.alpha_min_chut = np.zeros(self.npad)

            for n_pad in range(self.npad):
                eq_0 = self.choice_CAIMP["impos_EQ"]["pos_EQ"][0]
                eq_1 = self.choice_CAIMP["impos_EQ"]["pos_EQ"][1]

                xx_alpha = eq_0 * self.Cr * np.cos(eq_1)
                yy_alpha = eq_0 * self.Cr * np.sin(eq_1)

                xryr_alpha = np.dot(
                    [
                        [np.cos(self.sigma[n_pad]), np.sin(self.sigma[n_pad])],
                        [-np.sin(self.sigma[n_pad]), np.cos(self.sigma[n_pad])],
                    ],
                    [[xx_alpha], [yy_alpha]],
                )

                self.alpha_max_chut[n_pad] = (
                    (self.Rp - self.Rs - np.cos(self.TETA2)
                    * (xryr_alpha[0, 0] + self.Rp - self.Rs - self.Cr_ref[n_pad]))
                    / (np.sin(self.TETA2) * (self.Rp + self.tpad))
                    - (xryr_alpha[1, 0]) / (self.Rp + self.tpad)
                )

                self.alpha_min_chut[n_pad] = (
                    (self.Rp - self.Rs - np.cos(self.TETA1)
                    * (xryr_alpha[0, 0] + self.Rp - self.Rs - self.Cr_ref[n_pad]))
                    / (np.sin(self.TETA1) * (self.Rp + self.tpad))
                    - (xryr_alpha[1, 0]) / (self.Rp + self.tpad)
                )

            self.x0 = 0.4 * self.alpha_max_chut

            start_time = time.time()
            for self.con_np in range(self.npad):
                idx = self.con_np
                self.score_dim = 100000

                x_opt = fmin(
                    self.hde_equilibrium,
                    self.x0[self.con_np],
                    xtol=0.1,
                    ftol=0.1,
                    maxiter=100,
                    disp=False,
                )

                ang_rot[idx] = x_opt.item() if hasattr(x_opt, "item") else x_opt # Rotation angle of each pad
                momen_rot[idx] = self.score_dim

            elapsed_time = time.time() - start_time
            print(elapsed_time)

            self.psi_pad = ang_rot
            self.FX_dim = np.sum(self.Fx_dim)
            self.FY_dim = np.sum(self.Fy_dim)

            np.set_printoptions(precision=20)

            self.xdin = np.zeros(self.npad + 2)
            self.xdin = self.choice_CAIMP["impos_EQ"]["pos_EQ"] + list(ang_rot)

            if "result" in self.choice_CAIMP["impos_EQ"]["print"]:
                print("Moment [N.m]:", momen_rot)
                print("Rotation Angle [RAD]:", ang_rot)

        psi_pad = np.zeros(self.npad)
        for k_pad in range(self.npad):
            psi_pad[k_pad] = self.xdin[k_pad + 2] # Tilting angle of each pad

        n_k = self.nX * self.nZ
        tol_t = 0.1  # Celsius degrees

        for n_p in range(self.npad):
            t_new = self.TT_i[:, :, n_p]
            t_i = 1.1 * t_new
            cont_temp = 0

            while abs((t_new - t_i).max()) >= tol_t:
                cont_temp += 1
                t_i = np.array(t_new)

                mi_i = self.a_a * np.exp(self.b_b * t_i)  # [Pa.s]
                mi = mi_i / self.mi0  # viscosidade adimensional

                k_idx = 0
                mat_coef = np.zeros((n_k, n_k))
                b_vec = np.zeros(n_k)

                # transformação de coordenadas - inercial -> pivot
                xryr, xryrpt, xr, yr, xrpt, yrpt = self.xr_fun(
                    n_p, self.xdin[0], self.xdin[1]
                )

                alpha = psi_pad[n_p]
                alphapt = 0

                for ii in range(self.nZ):
                    for jj in range(self.nX):
                        teta_e = self.XTETA[jj] + 0.5 * self.dTETA
                        teta_w = self.XTETA[jj] - 0.5 * self.dTETA

                        h_p = (
                            self.Rp
                            - self.Rs
                            - (
                                np.sin(self.XTETA[jj]) * (yr + alpha * (self.Rp + self.tpad))
                                + np.cos(self.XTETA[jj])
                                * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                            )
                        ) / self.Cr_ref[n_p]

                        h_e = (
                            self.Rp
                            - self.Rs
                            - (
                                np.sin(teta_e) * (yr + alpha * (self.Rp + self.tpad))
                                + np.cos(teta_e)
                                * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                            )
                        ) / self.Cr_ref[n_p]

                        h_w = (
                            self.Rp
                            - self.Rs
                            - (
                                np.sin(teta_w) * (yr + alpha * (self.Rp + self.tpad))
                                + np.cos(teta_w)
                                * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                            )
                        ) / self.Cr_ref[n_p]

                        h_n = h_p
                        h_s = h_p

                        h_pt = -(
                            1 / (self.Cr_ref[n_p] * self.speed)
                        ) * (
                            np.cos(self.XTETA[jj]) * xrpt
                            + np.sin(self.XTETA[jj]) * yrpt
                            + np.sin(self.XTETA[jj]) * (self.Rp + self.tpad) * alphapt
                        )

                        self.H[ii, jj] = h_p

                        # harmonic/arithmetic averages of mi on faces
                        if jj == 0 and ii == 0:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = mi[ii, jj]
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = mi[ii, jj]

                        if jj == 0 and 0 < ii < self.nZ - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = mi[ii, jj]
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if jj == 0 and ii == self.nZ - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = mi[ii, jj]
                            mi_n = mi[ii, jj]
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if ii == 0 and 0 < jj < self.nX - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = mi[ii, jj]

                        if 0 < jj < self.nX - 1 and 0 < ii < self.nZ - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = mi[ii, jj]
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if ii == 0 and jj == self.nX - 1:
                            mi_e = mi[ii, jj]
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = mi[ii, jj]

                        if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                            mi_e = mi[ii, jj]
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if jj == self.nX - 1 and ii == self.nZ - 1:
                            mi_e = mi[ii, jj]
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = mi[ii, jj]
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        c_e = (1 / (self.betha_p**2)) * (h_e**3 / (12 * mi_e)) * (self.dZ / self.dX)
                        c_w = (1 / (self.betha_p**2)) * (h_w**3 / (12 * mi_w)) * (self.dZ / self.dX)
                        c_n = (self.Rp / self.L) ** 2 * (self.dX / self.dZ) * (h_n**3 / (12 * mi_n))
                        c_s = (self.Rp / self.L) ** 2 * (self.dX / self.dZ) * (h_s**3 / (12 * mi_s))
                        c_p = -(c_e + c_w + c_n + c_s)
                        b_val = (
                            (self.Rs / (2 * self.Rp * self.betha_p)) * self.dZ * (h_e - h_w)
                            + h_pt * self.dX * self.dZ
                        )
                        b_vec[k_idx] = b_val

                        # filling mat_coef according to position in the mesh
                        if ii == 0 and jj == 0:
                            mat_coef[k_idx, k_idx] = c_p - c_s - c_w
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx + self.nX] = c_n

                        if ii == 0 and 0 < jj < self.nX - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_s
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx + self.nX] = c_n

                        if ii == 0 and jj == self.nX - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_e - c_s
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx + self.nX] = c_n

                        if jj == 0 and 0 < ii < self.nZ - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_w
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - self.nX] = c_s
                            mat_coef[k_idx, k_idx + self.nX] = c_n

                        if 0 < ii < self.nZ - 1 and 0 < jj < self.nX - 1:
                            mat_coef[k_idx, k_idx] = c_p
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nX] = c_s
                            mat_coef[k_idx, k_idx + self.nX] = c_n
                            mat_coef[k_idx, k_idx + 1] = c_e

                        if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_e
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nX] = c_s
                            mat_coef[k_idx, k_idx + self.nX] = c_n

                        if jj == 0 and ii == self.nZ - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_n - c_w
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - self.nX] = c_s

                        if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_n
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nX] = c_s

                        if ii == self.nZ - 1 and jj == self.nX - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_e - c_n
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nX] = c_s

                        k_idx += 1

                # pressure field solution
                p_vec = np.linalg.solve(mat_coef, b_vec)

                cont_lin = 0
                for i_lin in np.arange(self.nZ):
                    for j_col in np.arange(self.nX):
                        self.P[i_lin, j_col] = p_vec[cont_lin]
                        cont_lin += 1
                        if self.P[i_lin, j_col] < 0:
                            self.P[i_lin, j_col] = 0

            # If the pressure is greater than the previous pressure, update the pad_in
            if np.max(self.P) > np.max(self.P_bef):
                self.pad_in = n_p
                self.P_bef = self.P.copy()

            self.PP[:, :, n_p] = self.P
            self.H0[:, :, n_p] = self.H

            self.PPdim[:, :, n_p] = (
                self.PP[:, :, n_p] * (self.mi0 * self.speed * self.Rp**2) / (self.Cr_ref[n_p] ** 2)
            )
            self.Hdim[:, :, n_p] = self.H * self.Cr_ref[n_p]
            self.h_pivot[n_p] = (
                self.Cr_ref[n_p]
                * (
                    self.Rp
                    - self.Rs
                    - (
                        np.sin(0) * (yr + alpha * (self.Rp + self.tpad))
                        + np.cos(0) * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                    )
                )
                / self.Cr_ref[n_p]
            )

        max_p = (
            self.PP[:, :, self.pad_in].max() * (self.mi0 * self.speed * self.Rp**2) / (self.Cr_ref[n_p] ** 2)
        )
        med_p = (
            self.PP[:, :, self.pad_in].mean() * (self.mi0 * self.speed * self.Rp**2) / (self.Cr_ref[n_p] ** 2)
        )
        max_t = self.TT_i[:, :, self.pad_in].max()
        med_t = self.TT_i[:, :, self.pad_in].mean()
        h_pivot0 = self.h_pivot[self.pad_in]
        ecc = np.sqrt(xr**2 + yr**2) / self.Cr_ref[n_p]

        return max_p, med_p, max_t, med_t, h_pivot0, ecc
    
    def coeffs_din(self):
        del_fx = np.zeros(self.npad)
        del_fy = np.zeros(self.npad)
        del_mj = np.zeros(self.npad)

        k_xx = np.zeros(self.npad)
        k_tt = np.zeros(self.npad)
        k_yy = np.zeros(self.npad)
        k_xt = np.zeros(self.npad)
        k_tx = np.zeros(self.npad)
        k_yx = np.zeros(self.npad)
        k_xy = np.zeros(self.npad)
        k_yt = np.zeros(self.npad)
        k_ty = np.zeros(self.npad)
        self.K = np.zeros((self.npad, 3, 3))

        c_xx = np.zeros(self.npad)
        c_tt = np.zeros(self.npad)
        c_yy = np.zeros(self.npad)
        c_xt = np.zeros(self.npad)
        c_tx = np.zeros(self.npad)
        c_yx = np.zeros(self.npad)
        c_xy = np.zeros(self.npad)
        c_yt = np.zeros(self.npad)
        c_ty = np.zeros(self.npad)
        self.C = np.zeros((self.npad, 3, 3))

        self.Sjpt = np.zeros((self.npad, 3, 3), dtype="complex")
        self.Sjipt = np.zeros((self.npad, 3, 3), dtype="complex")
        self.Aj = np.zeros((2, 2), dtype="complex")
        self.Hj = np.zeros((2, self.npad), dtype="complex")
        self.Vj = np.zeros((self.npad, 2), dtype="complex")
        self.Bj = np.zeros((self.npad, self.npad), dtype="complex")
        self.Sj = np.zeros((self.npad, 2, 2), dtype="complex")
        self.Tj = np.zeros((self.npad, 3, 3))

        self.Sw = np.zeros((2, 2), dtype="complex")

        psi_pad = np.zeros(self.npad)
        n_k = self.nX * self.nZ

        d_e = 0.005 * self.Cr  # space perturbation
        print(f"dE: {d_e}")

        d_ev = 0.025 * self.speed * d_e  # speed perturbation
        print(f"dEv: {d_ev}")

        tol_t = 0.1  # Celsius degree

        for a_p in range(4):
            for n_p in range(self.npad):
                xx_coef = self.xdin[:2]

                for k_pad in range(self.npad):
                    psi_pad[k_pad] = self.xdin[k_pad + 2]  # tilting angles of each pad

                t_new = self.TT_i[:, :, n_p]
                t_i = 1.1 * t_new
                cont_temp = 0

                while abs((t_new - t_i).max()) >= tol_t:
                    cont_temp += 1
                    t_i = np.array(t_new)

                    mi_i = self.a_a * np.exp(self.b_b * t_i)  # [Pa.s]
                    mi = mi_i / self.mi0  # dimensionless viscosity field

                    k_idx = 0  # vectorization index
                    mat_coef = np.zeros((n_k, n_k))
                    b_vec = np.zeros(n_k)

                    # transformation of coordinates - inertial to pivot referential
                    xryr, xryrpt, xr, yr, xrpt, yrpt = self.xr_fun(
                        n_p, xx_coef[0], xx_coef[1]
                    )

                    alphapt = 0

                    if a_p == 0:
                        xr += d_e
                    elif a_p == 1:
                        yr += d_e
                    if a_p == 2:
                        xrpt += d_ev
                    if a_p == 3:
                        yrpt += d_ev

                    alpha = psi_pad[n_p]

                    for ii in range(self.nZ):
                        for jj in range(self.nX):
                            teta_e = self.XTETA[jj] + 0.5 * self.dTETA
                            teta_w = self.XTETA[jj] - 0.5 * self.dTETA

                            h_p = (
                                self.Rp
                                - self.Rs
                                - (
                                    np.sin(self.XTETA[jj]) * (yr + alpha * (self.Rp + self.tpad))
                                    + np.cos(self.XTETA[jj])
                                    * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                                )
                            ) / self.Cr_ref[n_p]

                            h_e = (
                                self.Rp
                                - self.Rs
                                - (
                                    np.sin(teta_e) * (yr + alpha * (self.Rp + self.tpad))
                                    + np.cos(teta_e)
                                    * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                                )
                            ) / self.Cr_ref[n_p]

                            h_w = (
                                self.Rp
                                - self.Rs
                                - (
                                    np.sin(teta_w) * (yr + alpha * (self.Rp + self.tpad))
                                    + np.cos(teta_w)
                                    * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                                )
                            ) / self.Cr_ref[n_p]

                            h_n = h_p
                            h_s = h_p

                            h_pt = -(
                                1 / (self.Cr_ref[n_p] * self.speed)
                            ) * (
                                np.cos(self.XTETA[jj]) * xrpt
                                + np.sin(self.XTETA[jj]) * yrpt
                                + np.sin(self.XTETA[jj]) * (self.Rp + self.tpad) * alphapt
                            )

                            self.H[ii, jj] = h_p

                            # viscosity at faces
                            if jj == 0 and ii == 0:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = mi[ii, jj]
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = mi[ii, jj]

                            if jj == 0 and 0 < ii < self.nZ - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = mi[ii, jj]
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if jj == 0 and ii == self.nZ - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = mi[ii, jj]
                                mi_n = mi[ii, jj]
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if ii == 0 and 0 < jj < self.nX - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = mi[ii, jj]

                            if 0 < jj < self.nX - 1 and 0 < ii < self.nZ - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = mi[ii, jj]
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if ii == 0 and jj == self.nX - 1:
                                mi_e = mi[ii, jj]
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = mi[ii, jj]

                            if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                                mi_e = mi[ii, jj]
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if jj == self.nX - 1 and ii == self.nZ - 1:
                                mi_e = mi[ii, jj]
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = mi[ii, jj]
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            c_e = (1 / (self.betha_p**2)) * (h_e**3 / (12 * mi_e)) * (self.dZ / self.dX)
                            c_w = (1 / (self.betha_p**2)) * (h_w**3 / (12 * mi_w)) * (self.dZ / self.dX)
                            c_n = (self.Rp / self.L) ** 2 * (self.dX / self.dZ) * (h_n**3 / (12 * mi_n))
                            c_s = (self.Rp / self.L) ** 2 * (self.dX / self.dZ) * (h_s**3 / (12 * mi_s))
                            c_p = -(c_e + c_w + c_n + c_s)
                            b_val = (
                                (self.Rs / (2 * self.Rp * self.betha_p)) * self.dZ * (h_e - h_w)
                                + h_pt * self.dX * self.dZ
                            )
                            b_vec[k_idx] = b_val

                            # fill mat_coef according to mesh location
                            if ii == 0 and jj == 0:
                                mat_coef[k_idx, k_idx] = c_p - c_s - c_w
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx + self.nX] = c_n

                            if ii == 0 and 0 < jj < self.nX - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_s
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx + self.nX] = c_n

                            if ii == 0 and jj == self.nX - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_e - c_s
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx + self.nX] = c_n

                            if jj == 0 and 0 < ii < self.nZ - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_w
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - self.nX] = c_s
                                mat_coef[k_idx, k_idx + self.nX] = c_n

                            if 0 < ii < self.nZ - 1 and 0 < jj < self.nX - 1:
                                mat_coef[k_idx, k_idx] = c_p
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nX] = c_s
                                mat_coef[k_idx, k_idx + self.nX] = c_n
                                mat_coef[k_idx, k_idx + 1] = c_e

                            if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_e
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nX] = c_s
                                mat_coef[k_idx, k_idx + self.nX] = c_n

                            if jj == 0 and ii == self.nZ - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_n - c_w
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - self.nX] = c_s

                            if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_n
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nX] = c_s

                            if ii == self.nZ - 1 and jj == self.nX - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_e - c_n
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nX] = c_s

                            k_idx += 1

                    # Pressure field solution
                    p_vec = np.linalg.solve(mat_coef, b_vec)

                    cont = 0
                    for i_lin in np.arange(self.nZ):
                        for j_col in np.arange(self.nX):
                            self.P[i_lin, j_col] = p_vec[cont]
                            cont += 1
                            if self.P[i_lin, j_col] < 0:
                                self.P[i_lin, j_col] = 0

                    # =================== Energy equation & pressure gradients ===================
                    n_k = self.nX * self.nZ
                    mat_coef_t = np.zeros((n_k, n_k))
                    b_t = np.zeros(n_k)
                    test_diag = np.zeros(n_k)  # diagnostic variable 

                    k_t_idx = 0  # vectorization temperature index
                    for ii in range(self.nZ):
                        for jj in range(self.nX):
                            if jj == 0 and ii == 0:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                            if jj == 0 and 0 < ii < self.nZ - 1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                            if jj == 0 and ii == self.nZ - 1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / self.dZ

                            if ii == 0 and 0 < jj < self.nX - 1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                            if 0 < jj < self.nX - 1 and 0 < ii < self.nZ - 1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                            if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / self.dZ

                            if ii == 0 and jj == self.nX - 1:
                                self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                            if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                                self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                            if jj == self.nX - 1 and ii == self.nZ - 1:
                                self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / self.dX
                                self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / self.dZ

                            # --------------- Turbulence (Eddy viscosity) ---------------
                            h_p_loc = self.H[ii, jj]
                            mi_p = mi[ii, jj]

                            self.Reyn[ii, jj, n_p] = (
                                self.rho
                                * self.speed
                                * self.Rs
                                * (h_p_loc / self.L)
                                * self.Cr_ref[n_p]
                                / (self.mi0 * mi_p)
                            )

                            if self.Reyn[ii, jj, n_p] <= 500:
                                delta_turb = 0
                            elif 400 < self.Reyn[ii, jj, n_p] <= 1000:
                                delta_turb = 1 - ((1000 - self.Reyn[ii, jj, n_p]) / 500) ** (1 / 8)
                            else:
                                delta_turb = 1

                            du_dx = ((h_p_loc / self.mi_turb[ii, jj, n_p]) * self.dPdX[ii, jj]) - (self.speed / h_p_loc)
                            dw_dx = (h_p_loc / self.mi_turb[ii, jj, n_p]) * self.dPdZ[ii, jj]

                            tau = self.mi_turb[ii, jj, n_p] * np.sqrt(du_dx**2 + dw_dx**2)
                            y_wall = (
                                (h_p_loc * self.Cr_ref[n_p] * 2)
                                / (self.mi0 * self.mi_turb[ii, jj, n_p] / self.rho)
                            ) * ((abs(tau) / self.rho) ** 0.5)

                            emv = 0.4 * (y_wall - (10.7 * np.tanh(y_wall / 10.7)))
                            self.mi_turb[ii, jj, n_p] = mi_p * (1 + (delta_turb * emv))
                            mi_t = self.mi_turb[ii, jj, n_p]

                            # --------------- Coefficients for the energy equation ---------------
                            aux_up = 1
                            if self.XZ[ii] < 0:
                                aux_up = 0

                            a_e = -(self.kt * h_p_loc * self.dZ) / (
                                self.rho * self.Cp * self.speed * ((self.betha_p * self.Rp) ** 2) * self.dX
                            )

                            a_w = (
                                ((h_p_loc**3) * self.dPdX[ii, jj] * self.dZ) / (12 * mi_t * (self.betha_p**2))
                                - ((self.Rs * h_p_loc * self.dZ) / (2 * self.Rp * self.betha_p))
                                - (self.kt * h_p_loc * self.dZ)
                                / (self.rho * self.Cp * self.speed * ((self.betha_p * self.Rp) ** 2) * self.dX)
                            )

                            a_n_1 = (aux_up - 1) * (
                                ((self.Rp**2) * (h_p_loc**3) * self.dPdZ[ii, jj] * self.dX)
                                / (12 * (self.L**2) * mi_t)
                            )
                            a_s_1 = (aux_up) * (
                                ((self.Rp**2) * (h_p_loc**3) * self.dPdZ[ii, jj] * self.dX)
                                / (12 * (self.L**2) * mi_t)
                            )

                            a_n_2 = -(
                                self.kt * h_p_loc * self.dX
                            ) / (self.rho * self.Cp * self.speed * (self.L**2) * self.dZ)

                            a_s_2 = -(
                                self.kt * h_p_loc * self.dX
                            ) / (self.rho * self.Cp * self.speed * (self.L**2) * self.dZ)

                            a_n = a_n_1 + a_n_2
                            a_s = a_s_1 + a_s_2
                            a_p_coef = -(a_e + a_w + a_n + a_s)

                            aux_b_t = (self.speed * self.mi0) / (self.rho * self.Cp * self.Tcub * self.Cr_ref[n_p])

                            b_t_g = (
                                self.mi0
                                * self.speed
                                * (self.Rs**2)
                                * self.dX
                                * self.dZ
                                * self.P[ii, jj]
                                * h_pt
                            ) / (self.rho * self.Cp * self.T0 * (self.Cr_ref[n_p] ** 2))

                            b_t_h = (
                                self.speed
                                * self.mi0
                                * (h_pt**2)
                                * 4
                                * mi_t
                                * self.dX
                                * self.dZ
                            ) / (self.rho * self.Cp * self.T0 * 3 * h_p_loc)

                            b_t_i = aux_b_t * (1 * mi_t * (self.Rs**2) * self.dX * self.dZ) / (
                                h_p_loc * self.Cr_ref[n_p]
                            )

                            b_t_j = aux_b_t * (
                                (self.Rp**2) * (h_p_loc**3) * (self.dPdX[ii, jj] ** 2) * self.dX * self.dZ
                            ) / (12 * self.Cr_ref[n_p] * (self.betha_p**2) * mi_t)

                            b_t_k = aux_b_t * (
                                (self.Rp**4) * (h_p_loc**3) * (self.dPdZ[ii, jj] ** 2) * self.dX * self.dZ
                            ) / (12 * self.Cr_ref[n_p] * (self.L**2) * mi_t)

                            b_t_val = b_t_g + b_t_h + b_t_i + b_t_j + b_t_k
                            b_t[k_t_idx] = b_t_val

                            # fill mat_coef_t according to mesh location
                            if ii == 0 and jj == 0:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s - a_w
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                                b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (self.Tcub / self.T0)
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (mat_coef_t[k_t_idx, k_t_idx + 1] + mat_coef_t[k_t_idx, k_t_idx + self.nX])
                                )

                            if ii == 0 and 0 < jj < self.nX - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (
                                        mat_coef_t[k_t_idx, k_t_idx + 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - 1]
                                        + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                                    )
                                )

                            if ii == 0 and jj == self.nX - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_s
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (mat_coef_t[k_t_idx, k_t_idx - 1] + mat_coef_t[k_t_idx, k_t_idx + self.nX])
                                )

                            if jj == 0 and 0 < ii < self.nZ - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef - a_w
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                                mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                                b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (self.Tcub / self.T0)
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (
                                        mat_coef_t[k_t_idx, k_t_idx + 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                        + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                                    )
                                )

                            if 0 < ii < self.nZ - 1 and 0 < jj < self.nX - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                                mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (
                                        mat_coef_t[k_t_idx, k_t_idx + 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                        + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                                    )
                                )

                            if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                                mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (
                                        mat_coef_t[k_t_idx, k_t_idx - 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                        + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                                    )
                                )

                            if jj == 0 and ii == self.nZ - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n - a_w
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                                b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (self.Tcub / self.T0)
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (mat_coef_t[k_t_idx, k_t_idx + 1] + mat_coef_t[k_t_idx, k_t_idx - self.nX])
                                )

                            if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (
                                        mat_coef_t[k_t_idx, k_t_idx + 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - 1]
                                        + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                    )
                                )

                            if ii == self.nZ - 1 and jj == self.nX - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_n
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                                test_diag[k_t_idx] = (
                                    mat_coef_t[k_t_idx, k_t_idx]
                                    - (mat_coef_t[k_t_idx, k_t_idx - 1] + mat_coef_t[k_t_idx, k_t_idx - self.nX])
                                )

                            k_t_idx += 1

                    # Solution of temperature field
                    t_vec = np.linalg.solve(mat_coef_t, b_t)
                    cont = 0
                    for i_lin in np.arange(self.nZ):
                        for j_col in np.arange(self.nX):
                            t_new[i_lin, j_col] = self.T0 * t_vec[cont]
                            cont += 1

                    # =================== Hydrodynamic forces ===================
                    p_dimen = self.P * (self.mi0 * self.speed * self.Rp**2) / (self.Cr_ref[n_p] ** 2)

                    aux_f1 = np.zeros((self.nZ, self.nX))
                    aux_f2 = np.zeros((self.nZ, self.nX))
                    for ni in np.arange(self.nZ):
                        aux_f1[ni, :] = np.cos(self.XTETA)
                        aux_f2[ni, :] = np.sin(self.XTETA)

                    y_teta_f1 = self.P * aux_f1
                    f1_teta = np.trapezoid(y_teta_f1, self.XTETA)
                    self.F1_new[n_p] = -np.trapezoid(f1_teta, self.XZ)

                    y_teta_f2 = self.P * aux_f2
                    f2_teta = np.trapezoid(y_teta_f2, self.XTETA)
                    self.F2_new[n_p] = -np.trapezoid(f2_teta, self.XZ)

                    self.Mj_new[n_p] = self.F2_new[n_p] * (self.Rp + self.tpad)

                    # =================== Dynamic Coefficients ===================
                    del_fx[n_p] = -(self.F1_new[n_p] - self.F1[n_p])
                    del_mj[n_p] = -(self.Mj_new[n_p] - self.Mj[n_p])
                    del_fy[n_p] = -(self.F2_new[n_p] - self.F2[n_p])

                    # X-axis perturbation
                    if a_p == 0:
                        k_xx[n_p] = del_fx[n_p] / d_e * self.dimForca[n_p]
                        k_yx[n_p] = del_fy[n_p] / d_e * self.dimForca[n_p]
                        self.K[n_p, 0, 0] = k_xx[n_p]
                        self.K[n_p, 1, 0] = k_yx[n_p]

                    # Angular (pad) perturbation
                    elif a_p == 1:
                        k_yy[n_p] = del_fy[n_p] / d_e * self.dimForca[n_p]
                        k_xy[n_p] = del_fx[n_p] / d_e * self.dimForca[n_p]
                        self.K[n_p, 1, 1] = k_yy[n_p]
                        self.K[n_p, 0, 1] = k_xy[n_p]

                        k_tt[n_p] = k_yy[n_p] * ((self.Rp + self.tpad) ** 2)
                        k_tx[n_p] = -k_yx[n_p] * (self.Rp + self.tpad)
                        k_xt[n_p] = -k_xy[n_p] * (self.Rp + self.tpad)
                        k_ty[n_p] = -k_yy[n_p] * (self.Rp + self.tpad)
                        k_yt[n_p] = -k_yy[n_p] * (self.Rp + self.tpad)
                        self.K[n_p, 2, 2] = k_tt[n_p]
                        self.K[n_p, 0, 2] = k_xt[n_p]
                        self.K[n_p, 2, 0] = k_tx[n_p]
                        self.K[n_p, 1, 2] = k_yt[n_p]
                        self.K[n_p, 2, 1] = k_ty[n_p]

                    # x-axis speed perturbation
                    elif a_p == 2:
                        c_xx[n_p] = del_fx[n_p] / d_ev * self.dimForca[n_p]
                        c_yx[n_p] = del_fy[n_p] / d_ev * self.dimForca[n_p]
                        self.C[n_p, 0, 0] = c_xx[n_p]
                        self.C[n_p, 1, 0] = c_yx[n_p]

                    # Angular speed perturbation
                    elif a_p == 3:
                        c_yy[n_p] = del_fy[n_p] / d_ev * self.dimForca[n_p]
                        c_xy[n_p] = del_fx[n_p] / d_ev * self.dimForca[n_p]
                        self.C[n_p, 1, 1] = c_yy[n_p]
                        self.C[n_p, 0, 1] = c_xy[n_p]

                        c_tt[n_p] = c_yy[n_p] * ((self.Rp + self.tpad) ** 2)
                        c_tx[n_p] = -c_yx[n_p] * (self.Rp + self.tpad)
                        c_xt[n_p] = -c_xy[n_p] * (self.Rp + self.tpad)
                        c_ty[n_p] = -c_yy[n_p] * (self.Rp + self.tpad)
                        c_yt[n_p] = -c_yy[n_p] * (self.Rp + self.tpad)
                        self.C[n_p, 2, 2] = c_tt[n_p]
                        self.C[n_p, 0, 2] = c_xt[n_p]
                        self.C[n_p, 2, 0] = c_tx[n_p]
                        self.C[n_p, 1, 2] = c_yt[n_p]
                        self.C[n_p, 2, 1] = c_ty[n_p]

                        # Coefficient matrix reduction (per-pad contribution)
                        self.Sjpt[n_p] = self.K[n_p] + self.C[n_p] * self.speed * 1j
                        self.Tj[n_p] = np.array(
                            [
                                [
                                    np.cos(psi_pad[n_p] + self.sigma[n_p]),
                                    np.sin(psi_pad[n_p] + self.sigma[n_p]),
                                    0,
                                ],
                                [
                                    -np.sin(psi_pad[n_p] + self.sigma[n_p]),
                                    np.cos(psi_pad[n_p] + self.sigma[n_p]),
                                    0,
                                ],
                                [0, 0, 1],
                            ]
                        )
                        self.Sjipt[n_p] = (
                            self.Tj[n_p].T @ self.Sjpt[n_p] @ self.Tj[n_p]
                        )

                        # add 2x2 block in Aj and gyro term in Bj
                        self.Aj += np.array(
                            [
                                [self.Sjipt[n_p, 0, 0], self.Sjipt[n_p, 0, 1]],
                                [self.Sjipt[n_p, 1, 0], self.Sjipt[n_p, 1, 1]],
                            ]
                        )
                        self.Bj[n_p, n_p] = self.Sjipt[n_p, 2, 2]

        self.Hj = np.array(
            [
                [self.Sjipt[m, 0, 2] for m in range(self.npad)],
                [self.Sjipt[m, 1, 2] for m in range(self.npad)],
            ],
            dtype="complex",
        )
        self.Vj = np.array(
            [[self.Sjipt[m, 2, 0], self.Sjipt[m, 2, 1]] for m in range(self.npad)],
            dtype="complex",
        )

        # final reduction: Sw = Aj - Hj * Bj^{-1} * Vj
        self.Sw = self.Aj - (self.Hj @ np.linalg.inv(self.Bj) @ self.Vj)

        k_r = np.real(self.Sw)
        c_r = np.imag(self.Sw) / self.speed
        self.Kxx, self.Kyy, self.Kxy, self.Kyx = k_r[0, 0], k_r[1, 1], 2 * k_r[0, 1], 2 * k_r[1, 0]
        self.Cxx, self.Cyy, self.Cxy, self.Cyx = c_r[0, 0], c_r[1, 1], c_r[0, 1], c_r[1, 0]

        return self.Kxx, self.Kyy, self.Kxy, self.Kyx, self.Cxx, self.Cyy, self.Cxy, self.Cyx

    def hde_equilibrium(self, x):
        n_p = self.con_np
        t_new = self.T0 * np.ones((self.nZ, self.nX))

        # limits for x based on alpha_min/max
        if x > 0.9 * self.alpha_max_chut[n_p]:
            x = 0.8 * self.alpha_max_chut[n_p]
        if x <= 0.8 * self.alpha_min_chut[n_p]:
            x = 0.8 * self.alpha_min_chut[n_p]

        # equilibrium imposition condition
        if "impos_EQ" in self.op_key:
            eq_0 = self.choice_CAIMP[self.op_key]["pos_EQ"][0]
            eq_1 = self.choice_CAIMP[self.op_key]["pos_EQ"][1]

            psi_pad = np.zeros(self.npad)

            # ensure x is scalar
            psi_pad[n_p] = x.item() if hasattr(x, "item") else x

        n_k = self.nX * self.nZ
        tol_t = 0.1  # Celsius degrees

        t_i = 1.1 * t_new
        cont_temp = 0

        while abs((t_new - t_i).max()) >= tol_t:
            cont_temp += 1
            t_i = np.array(t_new)

            mi_i = self.a_a * np.exp(self.b_b * t_i)  # [Pa.s]
            mi = mi_i / self.mi0  # dimensionless viscosity

            k_idx = 0  # vectorization index
            mat_coef = np.zeros((n_k, n_k))
            b_vec = np.zeros(n_k)

            # coordinate transformation (inertial -> pivot)
            xryr, xryrpt, xr, yr, xrpt, yrpt = self.xr_fun(n_p, eq_0, eq_1)

            alpha = psi_pad[n_p]
            alphapt = 0

            # ===================== Assembly of the Reynolds equation ======================
            for ii in range(self.nZ):
                for jj in range(self.nX):
                    teta_e = self.XTETA[jj] + 0.5 * self.dTETA
                    teta_w = self.XTETA[jj] - 0.5 * self.dTETA

                    h_p = (
                        self.Rp
                        - self.Rs
                        - (
                            np.sin(self.XTETA[jj]) * (yr + alpha * (self.Rp + self.tpad))
                            + np.cos(self.XTETA[jj]) * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                        )
                    ) / self.Cr_ref[n_p]

                    h_e = (
                        self.Rp
                        - self.Rs
                        - (
                            np.sin(teta_e) * (yr + alpha * (self.Rp + self.tpad))
                            + np.cos(teta_e) * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                        )
                    ) / self.Cr_ref[n_p]

                    h_w = (
                        self.Rp
                        - self.Rs
                        - (
                            np.sin(teta_w) * (yr + alpha * (self.Rp + self.tpad))
                            + np.cos(teta_w) * (xr + self.Rp - self.Rs - self.Cr_ref[n_p])
                        )
                    ) / self.Cr_ref[n_p]

                    h_n = h_p
                    h_s = h_p

                    h_pt = -(
                        1 / (self.Cr_ref[n_p] * self.speed)
                    ) * (
                        np.cos(self.XTETA[jj]) * xrpt
                        + np.sin(self.XTETA[jj]) * yrpt
                        + np.sin(self.XTETA[jj]) * (self.Rp + self.tpad) * alphapt
                    )

                    self.H[ii, jj] = h_p

                    # viscosity on faces
                    if jj == 0 and ii == 0:
                        mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                        mi_w = mi[ii, jj]
                        mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                        mi_s = mi[ii, jj]

                    if jj == 0 and 0 < ii < self.nZ - 1:
                        mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                        mi_w = mi[ii, jj]
                        mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                        mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                    if jj == 0 and ii == self.nZ - 1:
                        mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                        mi_w = mi[ii, jj]
                        mi_n = mi[ii, jj]
                        mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                    if ii == 0 and 0 < jj < self.nX - 1:
                        mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                        mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                        mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                        mi_s = mi[ii, jj]

                    if 0 < jj < self.nX - 1 and 0 < ii < self.nZ - 1:
                        mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                        mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                        mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                        mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                    if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                        mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                        mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                        mi_n = mi[ii, jj]
                        mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                    if ii == 0 and jj == self.nX - 1:
                        mi_e = mi[ii, jj]
                        mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                        mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                        mi_s = mi[ii, jj]

                    if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                        mi_e = mi[ii, jj]
                        mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                        mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                        mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                    if jj == self.nX - 1 and ii == self.nZ - 1:
                        mi_e = mi[ii, jj]
                        mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                        mi_n = mi[ii, jj]
                        mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                    c_e = (1 / (self.betha_p**2)) * (h_e**3 / (12 * mi_e)) * (self.dZ / self.dX)
                    c_w = (1 / (self.betha_p**2)) * (h_w**3 / (12 * mi_w)) * (self.dZ / self.dX)
                    c_n = (self.Rp / self.L) ** 2 * (self.dX / self.dZ) * (h_n**3 / (12 * mi_n))
                    c_s = (self.Rp / self.L) ** 2 * (self.dX / self.dZ) * (h_s**3 / (12 * mi_s))
                    c_p = -(c_e + c_w + c_n + c_s)

                    b_val = (
                        (self.Rs / (2 * self.Rp * self.betha_p)) * self.dZ * (h_e - h_w)
                        + h_pt * self.dX * self.dZ
                    )
                    b_vec[k_idx] = b_val

                    # filling mat_coef according to position in the mesh
                    if ii == 0 and jj == 0:
                        mat_coef[k_idx, k_idx] = c_p - c_s - c_w
                        mat_coef[k_idx, k_idx + 1] = c_e
                        mat_coef[k_idx, k_idx + self.nX] = c_n

                    if ii == 0 and 0 < jj < self.nX - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_s
                        mat_coef[k_idx, k_idx + 1] = c_e
                        mat_coef[k_idx, k_idx - 1] = c_w
                        mat_coef[k_idx, k_idx + self.nX] = c_n

                    if ii == 0 and jj == self.nX - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_e - c_s
                        mat_coef[k_idx, k_idx - 1] = c_w
                        mat_coef[k_idx, k_idx + self.nX] = c_n

                    if jj == 0 and 0 < ii < self.nZ - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_w
                        mat_coef[k_idx, k_idx + 1] = c_e
                        mat_coef[k_idx, k_idx - self.nX] = c_s
                        mat_coef[k_idx, k_idx + self.nX] = c_n

                    if 0 < ii < self.nZ - 1 and 0 < jj < self.nX - 1:
                        mat_coef[k_idx, k_idx] = c_p
                        mat_coef[k_idx, k_idx - 1] = c_w
                        mat_coef[k_idx, k_idx - self.nX] = c_s
                        mat_coef[k_idx, k_idx + self.nX] = c_n
                        mat_coef[k_idx, k_idx + 1] = c_e

                    if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_e
                        mat_coef[k_idx, k_idx - 1] = c_w
                        mat_coef[k_idx, k_idx - self.nX] = c_s
                        mat_coef[k_idx, k_idx + self.nX] = c_n

                    if jj == 0 and ii == self.nZ - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_n - c_w
                        mat_coef[k_idx, k_idx + 1] = c_e
                        mat_coef[k_idx, k_idx - self.nX] = c_s

                    if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_n
                        mat_coef[k_idx, k_idx + 1] = c_e
                        mat_coef[k_idx, k_idx - 1] = c_w
                        mat_coef[k_idx, k_idx - self.nX] = c_s

                    if ii == self.nZ - 1 and jj == self.nX - 1:
                        mat_coef[k_idx, k_idx] = c_p - c_e - c_n
                        mat_coef[k_idx, k_idx - 1] = c_w
                        mat_coef[k_idx, k_idx - self.nX] = c_s

                    k_idx += 1

            # pressure field solution
            p_vec = np.linalg.solve(mat_coef, b_vec)

            cont = 0
            for i_lin in np.arange(self.nZ):
                for j_col in np.arange(self.nX):
                    self.P[i_lin, j_col] = p_vec[cont]
                    cont += 1
                    if self.P[i_lin, j_col] < 0:
                        self.P[i_lin, j_col] = 0

            # ===================== Energy equation / pressure gradients =====================
            n_k = self.nX * self.nZ
            mat_coef_t = np.zeros((n_k, n_k))
            b_t = np.zeros(n_k)
            test_diag = np.zeros(n_k)  # diagnostic variable

            k_t_idx = 0  # vectorization index (temperature)
            for ii in range(self.nZ):
                for jj in range(self.nX):
                    if jj == 0 and ii == 0:
                        self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                    if jj == 0 and 0 < ii < self.nZ - 1:
                        self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                    if jj == 0 and ii == self.nZ - 1:
                        self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / self.dZ

                    if ii == 0 and 0 < jj < self.nX - 1:
                        self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                    if 0 < jj < self.nX - 1 and 0 < ii < self.nZ - 1:
                        self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                    if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                        self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / self.dZ

                    if ii == 0 and jj == self.nX - 1:
                        self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                    if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                        self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / self.dZ

                    if jj == self.nX - 1 and ii == self.nZ - 1:
                        self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / self.dX
                        self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / self.dZ

                    # ---------------- Turbulence (Eddy viscosity) ----------------
                    h_p_loc = self.H[ii, jj]
                    mi_p = mi[ii, jj]

                    self.Reyn[ii, jj, n_p] = (
                        self.rho
                        * self.speed
                        * self.Rs
                        * (h_p_loc / self.L)
                        * self.Cr_ref[n_p]
                        / (self.mi0 * mi_p)
                    )

                    if self.Reyn[ii, jj, n_p] <= 500:
                        delta_turb = 0
                    elif 400 < self.Reyn[ii, jj, n_p] <= 1000:
                        delta_turb = 1 - ((1000 - self.Reyn[ii, jj, n_p]) / 500) ** (1 / 8)
                    else:
                        delta_turb = 1

                    du_dx = ((h_p_loc / self.mi_turb[ii, jj, n_p]) * self.dPdX[ii, jj]) - (self.speed / h_p_loc)
                    dw_dx = (h_p_loc / self.mi_turb[ii, jj, n_p]) * self.dPdZ[ii, jj]

                    tau = self.mi_turb[ii, jj, n_p] * np.sqrt(du_dx**2 + dw_dx**2)
                    y_wall = (
                        (h_p_loc * self.Cr_ref[n_p] * 2)
                        / (self.mi0 * self.mi_turb[ii, jj, n_p] / self.rho)
                    ) * ((abs(tau) / self.rho) ** 0.5)

                    emv = 0.4 * (y_wall - (10.7 * np.tanh(y_wall / 10.7)))
                    self.mi_turb[ii, jj, n_p] = mi_p * (1 + (delta_turb * emv))
                    mi_t = self.mi_turb[ii, jj, n_p]

                    # --------------- Energy equation coefficients ---------------
                    aux_up = 1
                    if self.XZ[ii] < 0:
                        aux_up = 0

                    a_e = -(self.kt * h_p_loc * self.dZ) / (
                        self.rho * self.Cp * self.speed * ((self.betha_p * self.Rp) ** 2) * self.dX
                    )

                    a_w = (
                        ((h_p_loc**3) * self.dPdX[ii, jj] * self.dZ) / (12 * mi_t * (self.betha_p**2))
                        - ((self.Rs * h_p_loc * self.dZ) / (2 * self.Rp * self.betha_p))
                        - (self.kt * h_p_loc * self.dZ)
                        / (self.rho * self.Cp * self.speed * ((self.betha_p * self.Rp) ** 2) * self.dX)
                    )

                    a_n_1 = (aux_up - 1) * (
                        ((self.Rp**2) * (h_p_loc**3) * self.dPdZ[ii, jj] * self.dX)
                        / (12 * (self.L**2) * mi_t)
                    )
                    a_s_1 = (aux_up) * (
                        ((self.Rp**2) * (h_p_loc**3) * self.dPdZ[ii, jj] * self.dX)
                        / (12 * (self.L**2) * mi_t)
                    )

                    a_n_2 = -(self.kt * h_p_loc * self.dX) / (self.rho * self.Cp * self.speed * (self.L**2) * self.dZ)
                    a_s_2 = -(self.kt * h_p_loc * self.dX) / (self.rho * self.Cp * self.speed * (self.L**2) * self.dZ)

                    a_n = a_n_1 + a_n_2
                    a_s = a_s_1 + a_s_2
                    a_p_coef = -(a_e + a_w + a_n + a_s)

                    aux_b_t = (self.speed * self.mi0) / (self.rho * self.Cp * self.Tcub * self.Cr_ref[n_p])

                    b_t_g = (
                        self.mi0 * self.speed * (self.Rs**2) * self.dX * self.dZ * self.P[ii, jj] * h_pt
                    ) / (self.rho * self.Cp * self.T0 * (self.Cr_ref[n_p] ** 2))

                    b_t_h = (self.speed * self.mi0 * (h_pt**2) * 4 * mi_t * self.dX * self.dZ) / (
                        self.rho * self.Cp * self.T0 * 3 * h_p_loc
                    )

                    b_t_i = aux_b_t * (mi_t * (self.Rs**2) * self.dX * self.dZ) / (h_p_loc * self.Cr_ref[n_p])

                    b_t_j = aux_b_t * (
                        (self.Rp**2) * (h_p_loc**3) * (self.dPdX[ii, jj] ** 2) * self.dX * self.dZ
                    ) / (12 * self.Cr_ref[n_p] * (self.betha_p**2) * mi_t)

                    b_t_k = aux_b_t * (
                        (self.Rp**4) * (h_p_loc**3) * (self.dPdZ[ii, jj] ** 2) * self.dX * self.dZ
                    ) / (12 * self.Cr_ref[n_p] * (self.L**2) * mi_t)

                    b_t_val = b_t_g + b_t_h + b_t_i + b_t_j + b_t_k
                    b_t[k_t_idx] = b_t_val

                    # filling mat_coef_t according to position
                    if ii == 0 and jj == 0:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s - a_w
                        mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                        mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                        b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (self.Tcub / self.T0)
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (mat_coef_t[k_t_idx, k_t_idx + 1] + mat_coef_t[k_t_idx, k_t_idx + self.nX])
                        )

                    if ii == 0 and 0 < jj < self.nX - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s
                        mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                        mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                        mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (
                                mat_coef_t[k_t_idx, k_t_idx + 1]
                                + mat_coef_t[k_t_idx, k_t_idx - 1]
                                + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                            )
                        )

                    if ii == 0 and jj == self.nX - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_s
                        mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                        mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (mat_coef_t[k_t_idx, k_t_idx - 1] + mat_coef_t[k_t_idx, k_t_idx + self.nX])
                        )

                    if jj == 0 and 0 < ii < self.nZ - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef - a_w
                        mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                        mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                        mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                        b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (self.Tcub / self.T0)
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (
                                mat_coef_t[k_t_idx, k_t_idx + 1]
                                + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                            )
                        )

                    if 0 < ii < self.nZ - 1 and 0 < jj < self.nX - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef
                        mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                        mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                        mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                        mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (
                                mat_coef_t[k_t_idx, k_t_idx + 1]
                                + mat_coef_t[k_t_idx, k_t_idx - 1]
                                + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                            )
                        )

                    if jj == self.nX - 1 and 0 < ii < self.nZ - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e
                        mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                        mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                        mat_coef_t[k_t_idx, k_t_idx + self.nX] = a_n
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (
                                mat_coef_t[k_t_idx, k_t_idx - 1]
                                + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                                + mat_coef_t[k_t_idx, k_t_idx + self.nX]
                            )
                        )

                    if jj == 0 and ii == self.nZ - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n - a_w
                        mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                        mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                        b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (self.Tcub / self.T0)
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (mat_coef_t[k_t_idx, k_t_idx + 1] + mat_coef_t[k_t_idx, k_t_idx - self.nX])
                        )

                    if ii == self.nZ - 1 and 0 < jj < self.nX - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n
                        mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                        mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                        mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (
                                mat_coef_t[k_t_idx, k_t_idx + 1]
                                + mat_coef_t[k_t_idx, k_t_idx - 1]
                                + mat_coef_t[k_t_idx, k_t_idx - self.nX]
                            )
                        )

                    if ii == self.nZ - 1 and jj == self.nX - 1:
                        mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_n
                        mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                        mat_coef_t[k_t_idx, k_t_idx - self.nX] = a_s
                        test_diag[k_t_idx] = (
                            mat_coef_t[k_t_idx, k_t_idx]
                            - (mat_coef_t[k_t_idx, k_t_idx - 1] + mat_coef_t[k_t_idx, k_t_idx - self.nX])
                        )

                    k_t_idx += 1

            # temperature field solution
            t_vec = np.linalg.solve(mat_coef_t, b_t)
            cont = 0
            for i_lin in np.arange(self.nZ):
                for j_col in np.arange(self.nX):
                    t_new[i_lin, j_col] = self.T0 * t_vec[cont]
                    cont += 1

        # save updated temperature of the pad
        self.TT_i[:, :, n_p] = t_new

        # ===================== Hydrodynamic forces =====================
        self.P_dimen = self.P * (self.mi0 * self.speed * self.Rp**2) / (self.Cr_ref[n_p] ** 2)

        aux_f1 = np.zeros((self.nZ, self.nX))
        aux_f2 = np.zeros((self.nZ, self.nX))
        for ni in np.arange(self.nZ):
            aux_f1[ni, :] = np.cos(self.XTETA)
            aux_f2[ni, :] = np.sin(self.XTETA)

        y_teta_f1 = self.P * aux_f1
        f1_teta = np.trapezoid(y_teta_f1, self.XTETA)
        self.F1[n_p] = -np.trapezoid(f1_teta, self.XZ)

        y_teta_f2 = self.P * aux_f2
        f2_teta = np.trapezoid(y_teta_f2, self.XTETA)
        self.F2[n_p] = -np.trapezoid(f2_teta, self.XZ)

        # resultant forces (inertial frame)
        self.Fx[n_p] = self.F1[n_p] * np.cos(psi_pad[n_p] + self.sigma[n_p])
        self.Fy[n_p] = self.F1[n_p] * np.sin(psi_pad[n_p] + self.sigma[n_p])
        self.Mj[n_p] = self.F2[n_p] * (self.Rp + self.tpad)

        self.Fx_dim[n_p] = self.Fx[n_p] * self.dimForca[n_p]
        self.Fy_dim[n_p] = self.Fy[n_p] * self.dimForca[n_p]
        self.Mj_dim[n_p] = self.Mj[n_p] * self.dimForca[n_p]
        self.Fj_dim[n_p] = self.F1[n_p] * self.dimForca[n_p]

        self.score_dim = self.Mj[n_p] * self.dimForca[n_p]
        return abs(self.score_dim)

    
    def xr_fun(self, n_p, eq_0, eq_1):
        
        xx = (
            eq_0
            * self.Cr
            * np.cos(eq_1)
        )
        yy = (
            eq_0
            * self.Cr
            * np.sin(eq_1)
        )
    
        xryr = np.dot(
            [
                [np.cos(self.sigma[n_p]), np.sin(self.sigma[n_p])],
                [-np.sin(self.sigma[n_p]), np.cos(self.sigma[n_p])],
            ],
            [[xx], [yy]],
        )

        xryrpt = np.dot(
            [
                [np.cos(self.sigma[n_p]), np.sin(self.sigma[n_p])],
                [-np.sin(self.sigma[n_p]), np.cos(self.sigma[n_p])],
            ],
            [[self.xpt], [self.ypt]],
        )
        
        xr = xryr[0, 0]
        yr = xryr[1, 0]

        xrpt = xryrpt[0, 0]
        yrpt = xryrpt[1, 0]
        
        return xryr, xryrpt, xr, yr, xrpt, yrpt

    def plot_results(self):

        XH, YH = np.meshgrid(self.XTETA, self.XZ)
        ax = plt.axes(projection='3d')
        ax.plot_surface(XH, YH, 1e-6*self.PPdim[:,:,self.pad_in], rstride=1, cstride=1, cmap='jet', edgecolor='none')
        plt.grid()
        ax.set_title('Pressure field')
        ax.set_xlim([np.min(self.XTETA), np.max(self.XTETA)])
        ax.set_ylim([np.min(self.XZ), np.max(self.XZ)])
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        ax.set_zlabel('Pressure [MPa]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, self.TT_i[:, :, self.pad_in], cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Temperature field [°C]')
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        plt.show()
        
    def plot_results2(self):

        d_axial = self.L/self.nZ
        axial = np.arange(0, self.L+d_axial , d_axial) 
        axial = axial[1:]-np.diff(axial)/2

        ang = []

        for k in range(self.npad):
            ang1 = (self.XTETA + self.sigma[k]) * 180 / np.pi 
            ang.append(ang1)

        fig_SP = self.plot_scatter(x_data=ang, y_data=self.PPdim, pos=15, y_title="Pressão (Pa)")
        # fig_SP.write_image("pressure_T.pdf", width=900, height=500, engine="kaleido")

        fig_ST = self.plot_scatter(x_data=ang, y_data=self.TT_i, pos=15, y_title="Temperature (ºC)")
        # fig_ST.write_image("temperature_T.pdf", width=900, height=500, engine="kaleido")

        fig_CP = self.plot_contourP(x_data=ang, y_data=axial, z_data=self.PPdim, z_title="Pressão (Pa)")
        # fig_CP.write_image("pressure_field_T.pdf", width=900, height=500, engine="kaleido")

        fig_CP = self.plot_contourT(x_data=ang, y_data=axial, z_data=self.TT_i, z_title="Temperature (ºC)")
        # fig_CP.write_image("temperature_field_T.pdf", width=900, height=500, engine="kaleido")

    def plot_scatter(self, x_data, y_data, pos, y_title):
        """This method plot a scatter(x,y) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        pos : float
            Probe position.
        y_title : str
            Name of the Y axis

        Returns
        -------
        fig : object
            Scatter figure.
        """

        fig = go.Figure()
        for i in range(self.npad):
            fig.add_trace(
                go.Scatter(
                    x=x_data[i],  # horizontal axis
                    y=y_data[pos][:, i],  # vertical axis
                    name=f"Segmento {i+1}",
                    line = dict(width = 4)
                )
            )
        fig.update_layout(xaxis_range=[np.array(x_data).min()*1.1, 360-abs(np.array(x_data).min())])
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor="LightGray")
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angulo do pivô (º)", font=dict(family="Arial", size=30, color = "black")),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(text=y_title, font=dict(family="Arial", size=30, color = "black")),
        )
        fig.update_layout(legend = dict(font = dict(family = "Arial", size = 24, color = "black")))
        fig.show()
        return fig

    def plot_contourP(self, x_data, y_data, z_data, z_title):
        """This method plot a contour(x,y,z) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        z_data : float
            Z axis data.
        z_title : str
            Name of the z axis

        Returns
        -------
        fig : object
            Contour figure.
        """

        fig = go.Figure()
        max_val = z_data.max()
        for l in range(self.npad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],  # horizontal axis
                    y=y_data,  # vertical axis
                    zmin=0,
                    zmax=max_val,
                    ncontours=15,
                    colorbar=dict(
                        title=z_title,  # title here
                        titleside="right",
                        titlefont=dict(size=30, family="Times New Roman"),
                        tickfont=dict(size=22),
                    ),
                )
            )
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(xaxis_range=[np.array(x_data).min()*1.1, 360-abs(np.array(x_data).min())])
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        fig.show()
        return fig

    def plot_contourT(self, x_data, y_data, z_data, z_title):
        """This method plot a contour(x,y,z) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        z_data : float
            Z axis data.
        z_title : str
            Name of the z axis

        Returns
        -------
        fig : object
            Contour figure.
        """

        fig = go.Figure()
        max_val = z_data.max()
        for l in range(self.npad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],  # horizontal axis
                    y=y_data,  # vertical axis
                    zmin=40,
                    zmax=max_val,
                    ncontours=25,
                    colorbar=dict(
                        title=z_title,  # title here
                        titleside="right",
                        titlefont=dict(size=30, family="Times New Roman"),
                        tickfont=dict(size=22),
                    ),
                )
            )
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(xaxis_range=[np.array(x_data).min()*1.1, 360-abs(np.array(x_data).min())])
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        fig.show()
        return fig

    def alpha(self):
        alphamax = (self.Rp-self.Rs-np.cos(self.TETA2)*(self.xryr[0, 0]+self.Rp-self.Rs-self.Cr_ref[n_p]))/(sin(self.TETA2)*(self.Rp+self.tpad)) - (self.xryr[1, 0])/(self.Rp+self.tpad)
        return abs(alphamax)
    
    def lub_selector(self):

        lubricant_dict = {
            "ISOVG32": {
                "viscosity1": 0.027968,                               # Pa.s
                "temp1": 40.0,                                    # degC
                "viscosity2": 0.004667,                               # Pa. s
                "temp2": 100.0,                                   # degC
                "lube_density": 873.99629,                            # kg/m³
                "lube_cp": 1948.7995685758851,                        # J/(kg*degC)
                "lube_conduct": 0.13126,                              # W/(m*degC)
            },
            "ISOVG46": {
                "viscosity1": 0.039693,                               # Pa.s
                "temp1": 40.0,                                    # degC
                "viscosity2": 0.006075,                               # Pa.s
                "temp2": 100.00000,                                   # degC
                "lube_density": 862.9,                                # kg/m³ 
                "lube_cp": 1950,                                      # J/(kg*degC)
                "lube_conduct": 0.15,                                 # W/(m*degC)
            },
            "ISOVG68": {
                "viscosity1": 0.060248,                               # Pa.s = N*s/m²
                "temp1": 40.00000,                                    # degC
                "viscosity2": 0.0076196,                              # Pa.s = N*s/m²
                "temp2": 100.00000,                                   # degC
                "lube_density": 886.00,                               # kg/m³ 
                "lube_cp": 1890.00,                                   # J/(kg*degC)
                "lube_conduct": 0.1316,                               # W/(m*degC)
            },
            "TEST": {
                "viscosity1": 0.02,                                   # Pa.s = N*s/m²
                "temp1": 50.00,                                       # degC
                "viscosity2": 0.01,                                   # Pa.s = N*s/m²
                "temp2": 80.00,                                       # degC
                "lube_density": 880.60,                               # kg/m³ 
                "lube_cp": 1800,                                      # J/(kg*degC)
                "lube_conduct": 0.1304                                # J/s*m*degC  --W/(m*degC)--
            },
        }
        return lubricant_dict[self.lubricant]

    def plot_results(self):
        ##### Plots
        XH, YH = np.meshgrid(self.XTETA, self.XZ)
        ax = plt.axes(projection='3d')
        ax.plot_surface(XH, YH, 1e-6*self.PPdim[:,:,self.pad_in], rstride=1, cstride=1, cmap='jet', edgecolor='none')
        plt.grid()
        ax.set_title('Pressure field')
        ax.set_xlim([np.min(self.XTETA), np.max(self.XTETA)])
        ax.set_ylim([np.min(self.XZ), np.max(self.XZ)])
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        ax.set_zlabel('Pressure [MPa]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, self.TT_i[:, :, self.pad_in], cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Temperature field [°C]')
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        plt.show()

def tilting_pad_example02():
    """Create an example of a tilting pad bearing with Thermo-Hydro-Dynamic effects.
    This function returns pressure field and dynamic coefficient. The
    purpose is to make available a simple model so that a doctest can be
    written using it.

    """

    bearing = Tilting(
        Rs = 1,
        npad = 12,
        Rp = 1.0025,
        tpad = 120e-3,
        betha_p = 20,
        rp_pad = 0.6,
        L = 350e-3,
        lubricant = "ISOVG68",
        # lubricant = "VALIDATION",
        Tcub = 40,
        nX = 30,
        nZ = 30,
        Cr = 250e-6,
        Cr_ref = [250e-6, 250e-6,250e-6,250e-6,250e-6,250e-6,250e-6,250e-6,250e-6,250e-6,250e-6,250e-6,],
                #   1       2        3      4       5       6       7       8       9       10      11      12      
        # Cr_ref = [550e-6, 700e-6, 450e-6, 250e-6, 250e-6, 300e-6, 550e-6, 500e-6, 250e-6, 250e-6, 250e-6, 350e-6],
        # Cr_ref = [721e-6, 814e-6, 443e-6, 250e-6, 250e-6, 250e-6, 386e-6, 343e-6, 250e-6, 250e-6, 250e-6, 407e-6],
        sigma = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], #graus
        speed = 90, #rpm
        choice_CAIMP = {"impos_EQ":{"pos_EQ":[0.60, 270*np.pi/180],
            "ent_angle":[0.000636496725369308,
            0.000913538939923720, 0.000690387727463155, 0.000854621895169753,
            0.000679444727719279, 0.000594279954895354, 0.000489138614623373,
            0.000377122288510832, 0.000319527936428236, 0.000340586533490991,
            0.000464320475944696, 0.000657880058055782],
            "print":["result","progress"]}}, #pos eq = [exc, ang] ent_ang = [pads angles]
        # choice_CAIMP={
        #     "calc_EQ": {
        #         "init_guess": [
        #             9.98805447808967e-10,
        #             -0.000159030915145932,
        #             0.000636496725369308,
        #             0.000913538939923720,
        #             0.000690387727463155,
        #             0.000854621895169753,
        #             0.000679444727719279,
        #             0.000594279954895354,
        #             0.000489138614623373,
        #             0.000377122288510832,
        #             0.000319527936428236,
        #             0.000340586533490991,
        #             0.000464320475944696,
        #             0.000657880058055782,
        #         ],
        #         "load": [0, -757e3],
        #         "print": ["result", "progress"]}},
            Coefs_D={"b_loc":4, "show_coef":True},

    )    


    bearing.run()

    # bearing.plot_results()
    # bearing.plot_results2()

if __name__ == "__main__":
    tilting_pad_example02()