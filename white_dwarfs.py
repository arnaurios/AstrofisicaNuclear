
# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES WHITE DWARF STRUCTURE EQUATIONS USING
# A FREE-FERMI GAS APPROXIMATION FOR THE EQUATION OF STATE ELECTRONS
##########################################################################
import numpy as np
import matplotlib.pyplot as plt

# IMPORT PHYSICAL CONSTANTS FROM STORING FILE
from physical_constants import *

# IMPORT EQUATION-OF-STATE FUNCTIONS
from eos_functions import *
##########################################################################
# KEY INPUTS TO THIS CODE
# NUMBER OF CENTRAL DENSITIES (AND HENCE OF MASSES & RADII)
number_central_density=1
# RANGE OF CENTRAL DENSITIES
central_density_i=1.
central_density_f=100.
# DENSITIES DISTRIBUTED IN LOGARITHMIC STENCIL
xfc_range=np.power(10,np.linspace(np.log10(central_density_i),np.log10(central_density_f),number_central_density))

################################################################################
# ELECTRON FRACTION
Ye=0.5;

# MASS UNITS FOR A VALUE OF Ye
M0_noYe=np.sqrt(3.*pi)/2./np.power(alphaG,1.5)*mnuc_kg/Msun
mass_units=M0_noYe*np.power(Ye,2.) # unit mass M_0 in solar masses for y_e
# RADIUS UNITS FOR A VALUE OF Ye
R0_noYe=np.sqrt(3.*pi)/2./np.power(alphaG,0.5)*(hbar/me/c)/1e3
radial_units=R0_noYe*Ye   # unit length R_0 in Km for y_e

# DENSITY UNITS
rho_noYe=np.power(me*c/hbar,3)*mnuc_kg/(3.*np.power(pi,2))

# PRESSURE UNITS
press_units=np.power(me*c,4)/np.power(hbar,3)*c/3./np.power(pi,2)

print('Typical Units for White Dwarfs')
txt='# M0={:.5f} [M_sun]'
print(txt.format(mass_units))
txt='# R0={:.2f} [km]'
print(txt.format(radial_units))
txt='# rho0={:.4E}/Ye*xF^3 [kg m-3]'
print(txt.format(rho_noYe))
txt='# P0={:.4E} [Pa]'
print(txt.format(press_units))

##########################################################################
# STEP SIZE OF INTEGRATION IN RADIAL COORDINATE
step_size=2e-4
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
fmt_MR='# M={:6.3f} [M_sun] # R={:8.1f} [km]'

# LOOP OVER CENTRAL DENSITY
for irhoc, xfc in enumerate( xfc_range) :

    # Numerical trick to provide a similar number of steps for all masses
    stepr=step_size/xfc**(0.75)

    # Solve differential equations with Euler method
    # Initial conditions
    mass_old=0.
    press_old=pressure(xfc)
    dens_old=np.power(xfc,3)
    r_old=stepr

    for iradial in range(0,iradial_max):

# STORE IN ARRAY
        radial_coord[ irhoc,iradial ]=r_old*radial_units # IN METERS
        mass_profile[ irhoc,iradial ]=mass_old*mass_units # IN SOLAR MASSES
        pres_profile[ irhoc,iradial ]=press_old*press_units # IN

# EULER STEP FORWARD
        dm=np.power(r_old,2)*dens_old*stepr
        dp=-mass_old*dens_old/np.power(r_old,2)*stepr

# NEW DATA IN EULER
        press_new=press_old+dp
        if press_new < 0. : break
        mass_new=mass_old+dm
        r_new=r_old+stepr
        x=invert_eos(press_new)
        dens_old=np.power(x,3)
        
        r_old=r_new
        mass_old=mass_new
        press_old=press_new

    else:
        print("Too many iterations")
        exit()
    # EXITING LOOP OVER RADIAL COORDINATES

    if( iradial < 100 ) :
        print(("Small number of iterations niter=",iradial))

    # FOR EACH CENTRAL DENSITY irhoc, STORE THE NUMBER OF RADIAL COORDINATES; THE RADIUS AND MASS OF THE STAR
    number_coord[ irhoc ] = int(iradial)
    WD_radius[ irhoc ] = r_old*radial_units
    WD_mass[ irhoc ] = mass_old*mass_units

    print(fmt_MR.format(WD_mass[irhoc],WD_radius[irhoc]))
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

ax2.set_xlim([0,20000])
ax2.set_ylim([1e19,1e32])

#fig.savefig("WD.pdf")
plt.show()
