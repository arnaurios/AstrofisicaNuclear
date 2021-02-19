# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES WHITE DWARF STRUCTURE EQUATIONS USING
# A FREE-FERMI GAS APPROXIMATION FOR THE EQUATION OF STATE ELECTRONS
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
# DENSITIES DISTRIBUTED IN LOGARITHMIC STENCIL
xfc_range=np.power(10,np.linspace(np.log10(central_density_i),np.log10(central_density_f),number_central_density))

################################################################################
# ELECTRON FRACTION
Ye=0.5;

# MASS UNITS FOR A VALUE OF Ye
M0_noYe=5./6.*np.sqrt(15.*pi)/np.power(alphaG,1.5)*mnuc/Msun
mass_units=M0_noYe*np.power(Ye,2.) # unit mass M_0 in solar masses for y_e
# RADIUS UNITS FOR A VALUE OF Ye
R0_noYe=np.sqrt(15.*pi)/2./np.power(alphaG,0.5)*(hbar*c/me/np.power(c,2))/1e3
radial_units=R0_noYe*Ye   # unit length R_0 in Km for y_e
# DENSITY UNITS
rho_noYe=np.power(me*c/hbar,3)*mnuc/(3.*np.power(pi,2))

print('Typical Units for White Dwarfs')
txt='# M0={:.5f} [M_sun]'
print(txt.format(mass_units))
txt='# R0={:.2f} [km]'
print(txt.format(radial_units))
txt='# rho0={:.4E}/Ye*xF^3 [kg m-3]'
print(txt.format(rho_noYe))

##########################################################################
# STEP SIZE OF INTEGRATION IN RADIAL COORDINATE
step_size=1e-3
# MAXIMUM NUMBER OF STEPS IN INTEGRATION
iradial_max=int(10./step_size)

# INITIALIZE MASS AND RADIUS ARRAYS
WD_mass=np.zeros(number_central_density)
WD_radius=np.zeros(number_central_density)

# FOR EACH RHO_C, THIS STORES THE NUMBER OF POINTS IN RADIAL COORDINATE
number_coord=np.zeros( number_central_density,dtype=int )
# THIS STORES THE VALUE OF r, m_< & p FOR ALL RHO_C AND RADIAL COORDINATES
radial_coord=np.zeros( (number_central_density,iradial_max) )
mass_profile=np.zeros( (number_central_density,iradial_max) )
pres_profile=np.zeros( (number_central_density,iradial_max) )

# FORMATS FOR OUTPUT
fmt='{:12.4f} {:12.4f} {:12.4E} {:12.4E} {:12.4E} {:12.4E}'
fmt_MR='# M={:6.3f} [M_sun] # R={:8.1f} [km]'

ifc=0;
# LOOP OVER CENTRAL DENSITIES
# xfc=VALUE OF CENTRAL X
for xfc in xfc_range :

    stepr=step_size/xfc  # adapt stepr to the central density

# INITIALIZE THE PROBLEM WITH BOUNDARY CONDITIONS AT CENTER
    rold=0.
    xfold=xfc
    mass_old=0.

    for iradial in range(0,iradial_max):
# ... compute mass density, pressure, energy density
        dens=rho_noYe/Ye*np.power(xfold,3) # In Kg/m^3
        ebar=energy(xfold)
        derebar=derenergy(xfold)
        pbar=-ebar+xfold*derebar/3.

# STORE IN ARRAY
        radial_coord[ ifc,iradial ]=rold*radial_units # IN METERS
        mass_profile[ ifc,iradial ]=mass_old*mass_units # IN SOLAR MASSES
        pres_profile[ ifc,iradial ]=pbar # IN

        if number_central_density==1 : print(fmt.format(r_in_meters,xfold,mass_in_kg,dens,pbar,ebar))

# EULER STEP FORWARD

        if mass_old==0. :
            xfnew=xfold
        else :
            xfnew=xfold - stepr*5.*mass_old*np.sqrt(1.+np.power(xfold,2))/(3.*np.power(rold,2)*xfold)

        mass_new=mass_old+stepr*3.*np.power(rold,2)*np.power(xfold,3)

        if xfnew < 0. : break

        rnew=rold+stepr

        rold=rnew
        xfold=xfnew
        mass_old=mass_new

    else:
        print("Too many iterations")
        exit()
    # EXITING LOOP OVER RADIAL COORDINATES

    if( iradial < 100 ) :
        print(("Small number of iterations niter=",iradial))

    # FOR EACH CENTRAL DENSITY ifc, STORE THE NUMBER OF RADIAL COORDINATES; THE RADIUS AND MASS OF THE STAR
    number_coord[ ifc ] = int(iradial)
    WD_radius[ ifc ] = rold*radial_units
    WD_mass[ ifc ] = mass_old*mass_units

    print(fmt_MR.format(WD_mass[ifc],WD_radius[ifc]))

    ifc=ifc+1
    # END LOOP OVER CENTRAL DENSITY

# PLOT A MASS-RADIUS DIAGRAM, INCLUDING THE STAR'S PROFILE AS A FUNCTION OF r
fig, (ax1,ax2) = plt.subplots(2,sharex=True)
for iplot in range(0,number_central_density) :
    ax1.plot(radial_coord[iplot,1:number_coord[iplot]], mass_profile[iplot,1:number_coord[iplot]],'c--',)

ax1.plot( WD_radius,WD_mass,'o-')
ax1.set(ylabel='Mass, $M$ / Mass profile, $m_<(r)$ [$M_\odot$]')
ax1.set_ylim([0,1.5])

for iplot in range(0,number_central_density) :
    ax2.semilogy(radial_coord[iplot,1:number_coord[iplot]], pres_profile[iplot,1:number_coord[iplot]],'g-',)

ax2.set(xlabel='Radial coordinate [km]')
ax2.set(ylabel='Pressure profile, $P(r)$ [Pa]')

ax2.set_xlim([0,30000])
ax2.set_ylim([1e-8,1e6])

#fig.savefig("WD.pdf")
plt.show()
