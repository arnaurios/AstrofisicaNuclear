import numpy as np
from physical_constants import *

# PRESSURE OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3
# INPUT x=p_F/mc
def pressure(x) :
    press=x*np.sqrt( 1.+np.power(x,2) )*(2.*np.power(x,2)-3.) + 3.*np.arcsinh(x)
    pressure=press/24./np.power(pi,2)
    return pressure

# ENERGY OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3
# INPUT x=p_F/mc
def energy(x) :
    ener=x*(1.+2.*np.power(x,2.))*np.sqrt(1.+np.power(x,2.))-np.log(x+np.sqrt(1.+np.power(x,2.)))
    energy=ener/8./np.power(pi,2)
    return energy

# DERIVATIVE OF ENERGY OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3
# INPUT x=p_F/mc
def derenergy(x) :
    derener=(np.power(x,2.)+np.power(x,4.))/np.sqrt(1.+np.power(x,2.))
    dergenergy=derener/np.power(pi,2)
    return dergenergy
