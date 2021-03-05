
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
# RANGE OF CENTRAL DENSITIES - 0.35 to 0.75 WORKS WELL
central_density_i=0.55
central_density_f=0.55
# DENSITIES DISTRIBUTED IN LOGARITHMIC STENCIL
#xfc_range=np.power(10,np.linspace(np.log10(central_density_i),np.log10(central_density_f),number_central_density))
xfc_range=np.linspace(central_density_i,central_density_f,number_central_density)

################################################################################
# ELECTRON FRACTION

# MASS UNITS FOR A VALUE OF Ye
M0_noYe=np.sqrt(3.*pi)/2./np.power(alphaG,1.5)*mnuc_kg/Msun
mass_units=M0_noYe # unit mass M_0 in solar masses for y_e
# RADIUS UNITS FOR A VALUE OF Ye
R0_noYe=np.sqrt(3.*pi)/2./np.power(alphaG,0.5)*(hbar/mnuc_kg/c)/1e3
radial_units=R0_noYe   # unit length R_0 in Km for y_e

# DENSITY UNITS
rho_noYe=np.power(mnuc_kg*c/hbar,3)*mnuc_kg/(3.*np.power(pi,2))

# PRESSURE UNITS
press_units=np.power(mnuc_kg*c,4)/np.power(hbar,3)*c/3./np.power(pi,2)

print('Typical Units for Neutron Stars')
txt='# M0={:.5f} [M_sun]'
print(txt.format(mass_units))
txt='# R0={:.2f} [km]'
print(txt.format(radial_units))
txt='# rho0={:.4E}*xF^3 [kg m-3]'
print(txt.format(rho_noYe))
txt='# P0={:.4E} [Pa]'
print(txt.format(press_units))

##########################################################################
# ... FIXED NUMBER OF CRUST AND CORE POINTS
# ... Read files of density, energy density and presure (crust and core)
ncrust=2573    # points to consider from the crust file
file_crust="./NS_EOS/crust.dat"
# ... READ CRUST FILE
eden_crust, pres_crust,den_crust = np.loadtxt(file_crust,usecols=(0,1,2)).T

ncore=100      # points to consider from the core file
file_core="./NS_EOS/EOS_npe_HHparam.dat"
eden_core, pres_core,den_core = np.loadtxt(file_core,usecols=(0,1,2)).T


# CONCATENATE ARRAYS
nf=ncrust+ncore
eden=np.concatenate((eden_crust,eden_core))
pres=np.concatenate((pres_crust,pres_core))
numden=np.concatenate((den_crust,den_core))

# TRANSFORM TO DIMENSIONLESS UNITS
# Energy density and pressure from MeVfm-3 to dimensionless
conv1=np.power(mneut,4)/hbc3/3./np.power(pi,2)
eden=eden/conv1
pres=pres/conv1

# Number density in units of rho0
numden=numden

densmax=np.amax( eden )

##########################################################################
# STEP SIZE OF INTEGRATION IN RADIAL COORDINATE
step_size=1e-4
# MAXIMUM NUMBER OF STEPS IN INTEGRATION
iradial_max=int(1000./step_size)

# INITIALIZE MASS AND RADIUS ARRAYS
NS_mass=np.zeros(number_central_density)
NS_radius=np.zeros(number_central_density)

# FOR EACH RHO_C, THIS STORES THE NUMBER OF POINTS IN RADIAL COORDINATE
number_coord=np.zeros( number_central_density,dtype=int )
# THIS STORES THE VALUE OF r, m_< & p FOR ALL RHO_C AND RADIAL COORDINATES
radial_coord=np.zeros( (number_central_density,iradial_max) )
mass_profile=np.zeros( (number_central_density,iradial_max) )
pres_profile=np.zeros( (number_central_density,iradial_max) )
numden_profile=np.zeros( (number_central_density,iradial_max) )

# FORMATS FOR OUTPUT
fmt='{:12.4f} {:12.4f} {:12.4E} {:12.4E} {:12.4E} {:12.4E}'
fmt_MR='# M={:6.3f} [M_sun] # R={:8.2f} [km]'

# LOOP OVER CENTRAL DENSITY
for irhoc, xfc in enumerate(xfc_range) :

    # Numerical trick to provide a similar number of steps for all masses
    stepr=step_size/xfc#**(0.75)

    # Solve differential equations with Euler method
    # Initial conditions
    mass_old=0.
    dens_old=np.power(xfc,3)
    press_old=np.interp(dens_old,eden,pres)
    numden_old=np.interp(dens_old,eden,numden)
    r_old=stepr

    if( dens_old >= densmax) :
        print('Maximum density is too high - decrease')
        print('Current density')
        print(dens_old)
        exit()

    for iradial in range(0,iradial_max):

# STORE IN ARRAY
        radial_coord[ irhoc,iradial ]=r_old*radial_units # IN METERS
        mass_profile[ irhoc,iradial ]=mass_old*mass_units # IN SOLAR MASSES
        pres_profile[ irhoc,iradial ]=press_old*press_units # IN
        numden_profile[ irhoc,iradial ]=numden_old

# EULER STEP FORWARD
        dm=np.power(r_old,2)*dens_old*stepr

        if( mass_old == 0) :
            dp=0.
        else:
            dp=-mass_old*dens_old/np.power(r_old,2)*stepr
            dp=dp*(1.+press_old/dens_old)
            dp=dp*(1.+press_old*np.power(r_old,3) / mass_old)
            dp=dp/(1.-2.*mass_old/r_old)

# NEW DATA IN EULER
        press_new=press_old+dp
        if press_new < 0. : break
        mass_new=mass_old+dm
        r_new=r_old+stepr

        # USE FOR FFG
        #dens_old=invert_eos(press_new)
        dens_old=np.interp(press_new,pres,eden)
        numden_old=np.interp(press_new,pres,numden)

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
    NS_radius[ irhoc ] = r_old*radial_units
    NS_mass[ irhoc ] = mass_old*mass_units

    print(fmt_MR.format(NS_mass[irhoc],NS_radius[irhoc]))
    # END LOOP OVER CENTRAL DENSITY

# PLOT A MASS-RADIUS DIAGRAM, INCLUDING THE STAR'S PRESSURE AND NUMBER DENSITY PROFILE AS A FUNCTION OF r
fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True)
for iplot in range(0,number_central_density) :
    ax1.plot(radial_coord[iplot,1:number_coord[iplot]], mass_profile[iplot,1:number_coord[iplot]],'c--',)
    ax2.plot(radial_coord[iplot,1:number_coord[iplot]], numden_profile[iplot,1:number_coord[iplot]],'g-',)
    ax3.semilogy(radial_coord[iplot,1:number_coord[iplot]], pres_profile[iplot,1:number_coord[iplot]],'g-',)

ax1.plot( NS_radius,NS_mass,'o-')

ax1.set(ylabel='Mass, $m_<(r)$ [$M_\odot$]')
ax1.set_ylim([0,2.5])
ax2.set(ylabel='$n(r)/0.16$ []')
ax2.set_ylim([0,10])
ax3.set(ylabel='Pressure, $P(r)$ [Pa]')
ax3.set_ylim([1e26,1e36])

ax3.set(xlabel='Radial coordinate [km]')
ax3.set_xlim([0,20])

plt.show()

# WRITING OUTPUT IN MR.dat FILE
#data_to_write = np.array( [NS_mass[:],NS_radius[:]] ).T
#outputfile="MR.dat"
#with open(outputfile,"w+") as file_id :
#    np.savetxt(file_id,data_to_write,fmt=["%16.6E","%16.6E"],header="  Radius [km]   Mass [M_sun]")
