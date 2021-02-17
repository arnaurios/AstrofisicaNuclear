import numpy as np
from physical_constants import *

def energy(x) :
    ener=x*(1.+2.*np.power(x,2.))*np.sqrt(1.+np.power(x,2.))-np.log(x+np.sqrt(1.+np.power(x,2.)))
    energy=ener/8./np.power(pi,2.)
    return energy

def derenergy(x) :
    derener=(np.power(x,2.)+np.power(x,4.))/np.sqrt(1.+np.power(x,2.))
    dergenergy=derener/pi**2.
    return dergenergy
