# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES WHITE DWARF STRUCTURE EQUATIONS USING
# A FREE-FERMI GAS APPROXIMATION FOR THE EQUATION OF STATE ELECTRONS
##########################################################################
import numpy as np
import scipy as scipy
import scipy.optimize
import matplotlib
import matplotlib
import matplotlib.pyplot as plt


# IMPORT PHYSICAL CONSTANTS FROM STORING FILE
from physical_constants import *

# IMPORT EQUATION-OF-STATE FUNCTIONS
from eos_functions import energy,derenergy
##########################################################################
# KEY INPUTS TO THIS CODE
# NUMBER OF CENTRAL DENSITIES (AND HENCE OF MASSES & RADII)
number_central_density=20
# RANGE OF CENTRAL DENSITIES
central_density_i=0.1
central_density_f=100.

# ELECTRON FRACTION
Ye=0.5;

##########################################################################
# INITIALIZE MASS AND RADIUS ARRAYS
WD_mass=np.zeros(number_central_density)
WD_radius=np.zeros(number_central_density)

# DEFINE THE MASS AND RADIUS SCALES OF THE WHITE-DWARF PROBLEM
# SEE SLIDES IN LECTURE 2
M0_noYe=5./6.*np.sqrt(15.*pi)/np.power(alphaG,1.5)*mnuc/Msun
R0_noYe=np.sqrt(15.*pi)/2./np.power(alphaG,0.5)*(hbar*c/me/np.power(c,2))/1e3
rho_noYe=np.power(me*np.power(c,2)/hbar/c,3)*mnuc/(3.*np.power(pi,2))

#M0_noYe=10.6162
#R0_noYe=17246.
#rho_noYe=823608e8

print('Typical Units for White Dwarfs')
txt='# M0={:.5f}*Ye^2 [M_sun]'
print(txt.format(M0_noYe))
txt='# R0={:.2f}*Ye [km]'
print(txt.format(R0_noYe))
txt='# rho0={:.4E}/Ye*xF^3 [kg m-3]'
print(txt.format(rho_noYe))

################################################################################
# MASS & RADIUS UNITS FOR A VALUE OF Ye
mass_units=M0_noYe*np.power(Ye,2.) # unit mass M_0 in solar masses for y_e
radial_units=R0_noYe*Ye   # unit length R_0 in Km for y_e

fmt='{:12.4f} {:12.4f} {:12.4E} {:12.4E} {:12.4E} {:12.4E}'
fmt_MR='# M={:6.3f} [M_sun] #R={:8.1f} [km]'

step_size=1e-3
iteration_max=int(10./step_size)

radial_coord=np.zeros( (number_central_density,iteration_max) )
number_coord=np.zeros( number_central_density )
mass_profile=np.zeros( (number_central_density,iteration_max) )
pres_profile=np.zeros( (number_central_density,iteration_max) )

xfc_range=np.power(10,np.linspace(np.log10(central_density_i),np.log10(central_density_f),number_central_density))

ifc=0;
# LOOP OVER CENTRAL DENSITIES
# xfc=VALUE OF CENTRAL X
for xfc in xfc_range :

    stepr=step_size/xfc  # adapt stepr to the central density

# INITIALIZE THE PROBLEM WITH BOUNDARY CONDITIONS AT CENTER
    rold=0.
    xfold=xfc
    mass_old=0.

    for iter in range(0,iteration_max):
# ... compute mass density, pressure, energy density
        dens=rho_noYe/Ye*np.power(xfold,3) # In Kg/m^3
        ebar=energy(xfold)
        derebar=derenergy(xfold)
        pbar=-ebar+xfold*derebar/3.

# CHANGE TO SI UNITS
        r_in_meters=rold*radial_units
        mass_in_kg=mass_old*mass_units

# STORE IN ARRAY
        #print([ifc,iter])
        radial_coord[ ifc,iter ]=r_in_meters
        mass_profile[ ifc,iter ]=mass_in_kg
        pres_profile[ ifc,iter ]=pbar

        if number_central_density==1 : print(fmt.format(r_in_meters,xfold,mass_in_kg,dens,pbar,ebar))

# EULER STEP FORWARD
        xfnew=xfold - stepr*5.*mass_old*np.sqrt(1.+np.power(xfold,2))/(3.*np.power(rold,2)*xfold)
        mass_new=mass_old+stepr*3.*np.power(rold,2)*np.power(xfold,3)

        if xfnew < 0. : break
        if mass_old==0. : xfnew=xfold
        rnew=rold+stepr

        rold=rnew
        xfold=xfnew
        mass_old=mass_new

    else:
        print "Too many iterations"
        exit()
    # EXITING LOOP OVER RADIAL COORDINATES

    if( iter < 100 ) :
        print(("Small number of iterations niter=",iter))


    number_coord[ ifc ] = iter

    WD_radius[ ifc ] = rnew*radial_units
    WD_mass[ ifc ] = mass_new*mass_units

    print(fmt_MR.format(WD_mass[ifc],WD_radius[ifc]))

    ifc=ifc+1
    # END LOOP OVER CENTRAL DENSITY

fig, ax = plt.subplots()
for iplot in range(0,number_central_density) :
    plt.plot(radial_coord[iplot,1:number_coord[iplot]], mass_profile[iplot,1:number_coord[iplot]],'c--',)

plt.plot( WD_radius,WD_mass,'o')
plt.xlabel('Radial coordinate [km]')
plt.xlim(0,30000)
plt.ylabel('Mass profile [M_sun]')
plt.ylim(0,1.5)
fig.savefig("WD.pdf")
plt.show()
