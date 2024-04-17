from .syslib import *
import numpy as np

class TtoEleak_Planck15(residual):
    r"""
    T2E leakage template a la Planck
    te'(nu1,nu2) = te(nu1,nu2)+enu(l,nu2)*tt(nu1,nu2)
    ee'(nu1,nu2) = ee(nu1,nu2)+enu(l,nu1)*te(nu1,nu2)+enu(l,nu2)*te(nu2,nu1)
                  +enu(l,nu1)*enu(l,nu2)*tt(nu1,nu2)
    enu(l,nu1)=enu0(nu1)+enu2(nu1)*l**2+enu4(nu1)*l**4
    """

    def eval(self,enu={'100':[0.,0.,0.]},nu=None):
        self.enul=dict()
        for k1 in enu.keys():
            enu_arr=np.array(enu[k1])
            self.enul[k1]=enu_arr[0]+enu_arr[1]*self.ell**2+enu_arr[2]*self.ell**4
        self.freq=nu
        dcl=dict()
        
        for c1,f1 in enumerate(self.freq):
            for f2 in self.freq[c1:]:
                if ("TT",f1,f2) in self.cl.keys():
                    dcl["TT",f1,f2] = np.zeros_like(self.cl["TT",f1,f2])
                    dcl["TT",f2,f1] = np.zeros_like(self.cl["TT",f1,f2])
                    dcl["TE",f1,f2] = self.enul[f2]*self.cl["TT",f1,f2]

                if ("TT",f1,f2) in self.cl.keys() and ("TE",f1,f2) in self.cl.keys() \
                        and ("TE",f2,f1) in self.cl.keys():
                    dcl["EE",f1,f2] = self.enul[f2]*self.cl["TE",f2,f1] + self.enul[f1]*self.cl["TE",f1,f2] +self.enul[f1]*self.enul[f2]*self.cl["TT",f1,f2]
                #dcl["ee",f1,f2] = self.elnu2*self.cl["te",f1,f2] + self.elnu1*self.cl["te",f1,f2] +self.elnu1*self.elnu2*self.cl["tt",f1,f2]

                dcl["ET",f1,f2] = dcl["TE",f1,f2]#self.elnu1*self.cl["tt",f2,f1]
                dcl["EE",f2,f1] = dcl["EE",f1,f2]#self.elnu1*self.cl["te",f1,f2] + self.elnu2*self.cl["te",f2,f1] +self.elnu1*self.elnu2*self.cl["tt",f2,f1]

        return dcl

class Calibration_alm(residual):
    r"""
    build calibration matrix to be applied to C_ell
    from calibration factors applied to a_lm
    i.e.: a_lm^X_nu1  -> c^X_nu1 a_lm^X_nu1, with X=T,E
    cal1,cal2 are dictionaries of calibration factors c^X_nu:
    cal1[XX][0,1,2]=[c^X_nu1,c^X_nu2,c^X_nu3]
    cal1[XX]*cal2[YY]=G^XY
    NB: cal built such that calibrated TE_nu1nu2 != TE_nu2nu1
    """

    def eval(self,cal1={'T':[1.,1.,1.]},cal2={'T':[1.,1.,1.]},nu=None):
        cal=dict()
        for k1 in cal1.keys():
            c1=np.array(cal1[k1])[...,np.newaxis]
            for k2 in cal2.keys():
                c2=np.array(cal2[k2])
                cal[k1+k2] = c1*c2

        self.freq=nu
        dcl=dict()

        for i1,f1 in enumerate(self.freq):
            for i2,f2 in enumerate(self.freq):
                for spec in cal.keys():
                    if (spec,f1,f2) in self.cl.keys():
                        dcl[spec,f1,f2] = cal[spec][i1,i2]*self.cl[spec,f1,f2]

        return dcl

class Rotation_alm(residual):
    r"""
    apply rotation of C_ell due to polangle miscalibration
    each freq channel nu is rotated by its own polangle alpha_nu
    NB: alpha must be in deg
    """

    def eval(self,alpha=1.,nu=None):
        self.freq=nu
        #NB must check that len(alpha)==len(freq)
        ca=np.cos(np.deg2rad(2.*np.array(alpha)))
        sa=np.sin(np.deg2rad(2.*np.array(alpha)))
        dcl=dict()

        for i1,f1 in enumerate(self.freq):
            for i2,f2 in enumerate(self.freq):
                
                if ("TE",f1,f2) in self.cl.keys():
                    dcl['TE',f1,f2] = ca[i2]*self.cl['TE',f1,f2]
                
                if ("EE",f1,f2) in self.cl.keys():
                    dcl['EE',f1,f2] = ca[i1]*ca[i2]*self.cl['EE',f1,f2]
                
                if ("TT",f1,f2) in self.cl.keys():
                    dcl['TT',f1,f2] = self.cl['TT',f1,f2]
                
                if ("BB",f1,f2) in self.cl.keys():
                    dcl['EE',f1,f2] += sa[i1]*sa[i2]*self.cl['BB',f1,f2]
                    
                if ("BB",f1,f2) in self.cl.keys() and ("EE",f1,f2) in self.cl.keys():
                    dcl['BB',f1,f2] = (ca[i1]*ca[i2]*self.cl['BB',f1,f2] +
                                       sa[i1]*sa[i2]*self.cl['EE',f1,f2])
                
                if ("EB",f1,f2) in self.cl.keys() and ("EB",f2,f1) in self.cl.keys():
                    dcl['EE',f1,f2] += (-ca[i1]*sa[i2]*self.cl['EB',f1,f2] -
                                        sa[i1]*ca[i2]*self.cl['EB',f2,f1])
                    dcl['BB',f1,f2] += (sa[i1]*ca[i2]*self.cl['EB',f1,f2] +
                                        sa[i2]*ca[i1]*self.cl['EB',f2,f1])
                    
                if ("EB",f1,f2) in self.cl.keys() and ("EB",f2,f1) in self.cl.keys() \
                        and ("EE",f1,f2) in self.cl.keys() \
                        and ("BB",f1,f2) in self.cl.keys():
                    dcl['EB',f1,f2] = (ca[i1]*sa[i2]*self.cl['EE',f1,f2] - 
                                       sa[i1]*ca[i2]*self.cl['BB',f1,f2] +
                                       ca[i1]*ca[i2]*self.cl['EB',f1,f2] -
                                       sa[i1]*sa[i2]*self.cl['EB',f2,f1])

                if ("TB",f1,f2) in self.cl.keys():
                    dcl['TE',f1,f2] += -sa[i2]*self.cl['TB',f1,f2]
                
                if ("TB",f1,f2) in self.cl.keys() and ("TE",f1,f2) in self.cl.keys():
                    dcl['TB',f1,f2] = (sa[i2]*self.cl['TE',f1,f2] +
                                       ca[i2]*self.cl['TB',f1,f2])

        # filling also the cross spectra if required,
        for f1 in self.freq:
            for f2 in self.freq:
                if ("ET",f1,f2) in self.cl.keys():
                    dcl['ET',f1,f2] = dcl['TE',f2,f1]
                
                if ("BE",f1,f2) in self.cl.keys():
                    dcl['BE',f1,f2] = dcl['EB',f2,f1]

                if ("BT",f1,f2) in self.cl.keys():
                    dcl['BT',f1,f2] = dcl['TB',f2,f1]

        return dcl
