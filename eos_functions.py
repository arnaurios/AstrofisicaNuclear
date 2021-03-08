import numpy as np
from physical_constants import *

# PRESSURE OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3/(3*pi^2)
# INPUT x=p_F/mc
def pressure(x) :
    press=x*np.sqrt( 1.+np.power(x,2) )*(2.*np.power(x,2)-3.) + 3.*np.arcsinh(x)
    pressure=press/8. #/24./np.power(pi,2)
    return pressure

# PRESSURE OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3/(3*pi^2)
# INPUT x=p_F/mc
def der_pressure(x) :
    dp=np.power(x,4)/np.sqrt( 1 + np.power(x,2) )
    der_pressure=dp #/3./np.power(pi,2)
    return der_pressure

# INVERTS EOS TO FIND VALUE OF RHO FOR A GIVEN INPUT PRESSURE, P_INPUT
def invert_eos(p_input) :
    accu=1e-10

    #print(p_input)

    # NEWTON RAPHSON PROCEDURE TO FIND THE ROOT OF P(x)-p_input
    # THIS HELPS INVERT RELATION
    if( p_input < 1 ) :
        x_guess=np.power( 5.*p_input, 1./5.)
    else :
        x_guess=np.power( 4.*p_input, 1./4.)
    x0=x_guess

    if( p_input > accu ) :
        diff=10.
        while diff > accu :
            x1=x0 - (pressure(x0)-p_input)/der_pressure(x0)
            diff= abs(x1-x0)
            #print(x1,x0,x_guess,diff,p_input)
            x0=x1

    x_out=x0

    return x_out

# ENERGY OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3/(3*pi^2)
# INPUT x=p_F/mc
def energy(x) :
    ener=x*(1.+2.*np.power(x,2.))*np.sqrt(1.+np.power(x,2.))-np.log(x+np.sqrt(1.+np.power(x,2.)))
    energy=ener*3./8.
    return energy

# DERIVATIVE OF ENERGY OF THE FREE FERMI GAS IN UNITS OF (mc^2)^4/(hbc)^3/(3*pi^2)
# INPUT x=p_F/mc
def derenergy(x) :
    derener=(np.power(x,2.)+np.power(x,4.))/np.sqrt(1.+np.power(x,2.))
    dergenergy=3.*derener
    return dergenergy
