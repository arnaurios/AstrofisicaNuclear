import numpy as np
import math
################################################################################
# DEFINE PHYSICAL CONSTANTS TO BE USED
pi=math.pi
# MASSES
mnuc_kg=(1.6726+1.6749)/2*1e-27; # Mass of nucleon as average of proton and neutron masses [kg]
me=9.1093837e-31; # Mass of electron [kg]
c=299792458; # Speed of Light [m/s]
G=6.6741e-11; # Gravitational constant [m3 kg-1 s-2]
hbar=1.0545718e-34; # Planck's h/2/pi constant [m2 kg / s]
alphaG=G*np.power(mnuc_kg,2)/hbar/c;
Msun=1.989e30; # Mass of the Sun [kg]
Msung=Msun*1e3; # Mass of the Sun [g]

# NUCLEAR PHYSICS NATURAL UNITS
hbc=197.3269804 # hbar x c in [MeV fm]
hbc3=np.power(hbc,3)
# MASSES OF PARTICLES IN MEV
mneut=939.56542052
mprot=938.27208816
mnuc=(mneut+mprot)/2.
m_el=0.5109989461
m_mu=105.6583745
