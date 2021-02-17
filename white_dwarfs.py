# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES WHITE DWARF STRUCTURE EQUATIONS USING
# A FREE-FERMI GAS APPROXIMATION FOR THE EQUATION OF STATE ELECTRONS
##########################################################################
import numpy as np
import scipy as scipy
import scipy.optimize
import matplotlib as mpl
from physical_constants import *

from eos_functions import energy,derenergy

# DEFINE THE MASS AND RADIUS SCALES OF THE WHITE-DWARF PROBLEM
# SEE SLIDES IN LECTURE 2
M0_noYe=5./6.*np.sqrt(15.*pi)/np.power(alphaG,1.5)*mnuc/Msun
R0_noYe=np.sqrt(15.*pi)/2./np.power(alphaG,0.5)*(hbar*c/me/np.power(c,2))/1e3
rho_noYe=np.power(me*np.power(c,2)/hbar/c,3)*mnuc/(3.*np.power(pi,2))

#HERE'S AN UNEXPECTED CHANGE
#M0_noYe=10.6162
#R0_noYe=17246.
#rho_noYe=9.823608e8

print('Typical Units for White Dwarfs')

txt='# M0={:.5f}*Ye^2 [M_sun]'
print(txt.format(M0_noYe))

txt='# R0={:.2f}*Ye [km]'
print(txt.format(R0_noYe))

txt='# rho0={:.4E}/Ye*xF^3 [kg m-3]'
print(txt.format(rho_noYe))

################################################################################

# ELECTRON FRACTION
Ye=0.5;

# MASS UNITS WHICH ARE USEFUL IN THIS PROGRAM
mass_units=M0_noYe*np.power(Ye,2.) # unit mass M_0 in solar masses for y_e
radial_units=R0_noYe*Ye   # unit length R_0 in Km for y_e

fmt='{:12.4f} {:12.4f} {:12.4E} {:12.4E} {:12.4E} {:12.4E}'

iteration_max=90000

# VALUE OF CENTRAL X
xfc=10
stepr=1e-4*xfc  # adapt stepr to the central density

# INITIALIZE THE PROBLEM WITH BOUNDARY CONDITIONS AT CENTER
rold=0.
xfold=xfc
mass_old=0.

#np.array()
for iter in range(0,iteration_max):
# ... compute mass density, pressure, energy density
    dens=rho_noYe/Ye*np.power(xfold,3) # In Kg/m^3
    ebar=energy(xfold)
    derebar=derenergy(xfold)
    pbar=-ebar+xfold*derebar/3.

    r_in_meters=rold*radial_units
    mass_in_kg=mass_old*mass_units

    print(fmt.format(r_in_meters,xfold,mass_in_kg,dens,pbar,ebar))

    xfnew=xfold - stepr*5.*mass_old*np.sqrt(1.+np.power(xfold,2))/(3.*np.power(rold,2)*xfold)

    if xfnew < 0. : break
    if mass_old==0. : xfnew=xfold

    mass_new=mass_old+stepr*3.*np.power(rold,2)*np.power(xfold,3)
    rnew=rold+stepr

    rold=rnew
    xfold=xfnew
    mass_old=mass_new

else:
    print "Too many iterations"
    exit()
