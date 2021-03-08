# coding: utf-8
#############################################################################
# THIS PYTHON CODE SOLVES FOR THE BETA-STABILITY CONDITIONS OF DENSE MATTER
#############################################################################
import numpy as np
import matplotlib.pyplot as plt

# IMPORT PHYSICAL CONSTANTS FROM STORING FILE
from physical_constants import *

# IMPORT EQUATION-OF-STATE FUNCTIONS
from eos_functions import *

##########################################################################
# KEY INPUTS TO THIS CODE
# NUMBER OF DENSITIES
num_densities=100
# CREATE DENSITY ARRAY
n0=0.16 # Density of normal nuclear matter in fm^{-3}
numden_i=0.5*n0
numden_f=10*n0
numden=np.linspace( numden_i,numden_f,num_densities )

# Parameters of EOS, Heiselberg+Hjorth-Jensen, Phys. Rep. 328 237 (2000)
# ANALYTICAL FORM: e = e0*u*(u-2-delta)/(1.+delta*u)+s0*u**gamma*(1-2.*xp)**2.
e0=15.8    # Saturation energy in MeV
delta=0.2  #
s0=32.     # Symmetry energy in MeV
gamma=0.6  # Power-law dependence of symmetry energy

# LOGICAL VALUE TO INCLUDE MUONS OR NOT IN CALCULATION
muons = True

# Asymmetry term as a function of density
sn=s0*np.power(numden/n0,gamma)

# STARTING NON-ZERO VALUE OF XP FOR ZERO FINDING ROUTINE
xp=np.zeros(num_densities)+1e-4
# Initial proton fraction
accuracy=1e-12
# START LOOP OVER densities
for iden,dens in enumerate( numden ):
    xpold=xp[iden]
    ssn=sn[iden]

# FIRST GUESS OF ELECTRON FUNCTION
    xmu_e=4.*ssn*(1.-2.*xpold)   # electron fraction

# FIND XP BY SELF-CONSISTENT XP BY
    diff=1.
    while diff > accuracy :
        xpnew=np.power(4.*ssn*(1.-2.*xpold),3)
        if (xmu_e > m_mu) & muons :
            xpnew=xpnew + np.power( np.power(4.*ssn*(1.-2.*xpold),2) - np.power(m_mu,2) ,1.5)
        xpnew=xpnew/(3.*np.power(pi,2)*dens*hbc3)

        xpold=(xpnew+xpold)/2.
        xmu_e=4.*ssn*(1.-2.*xpold)
        diff=np.abs( xpnew-xpold )

    #print(xpold,xpnew,diff)

# THIS SETS THE PROTON FRACTION FROM NOW ON
    xp[iden]=xpnew

##############################################################################
# NOW THAT WE HAVE DENSITY AND PROTON FRACTION, COMPUTE THERMODYNAMICAL
# PROPERTIES FOR NUCLEONS, ELECTRONS AND MUONS
##############################################################################
# NUCLEONS
# NUCLEAR ENERGY PER PARTICLE
u=numden/n0
enuc=e0*u*(u-2.-delta)/(1.+delta*u) \
+s0*np.power(u,gamma)*np.power( (1.-2.*xp),2 )

# proton density in fm-3
rhop=numden*xp
# neutron density in fm-3
rhon=numden*(1.-xp)
xn=1.-xp

# energy density (baryonic part)   in MeV fm-3
edens_n=enuc*numden + rhon*mneut + rhop*mprot

# Calculating the pressure from (total) derivative of energy density (baryonic part)
pres0=-e0*(n0-numden)*np.power(numden,2)*( (2.+delta)*n0 + delta*numden )
pres0=pres0/( n0*np.power( n0+delta*numden,2 ) )
pres1=s0*gamma*np.power(n0,-gamma)*np.power( (rhon - rhop),2 )*np.power(numden,gamma-1.)
p_nuc=pres0+pres1

##############################################################################
# PREPARE LEPTON CONTRIBUTION
electron_eden_units=np.power(m_el,4)/hbc3/3./np.power(pi,2)
muon_eden_units=np.power(m_mu,4)/hbc3/3./np.power(pi,2)

# ELECTRONS
# Electron chemical potential
xmue=4.*sn*(1.-2.*xp)
# Electron Fermi momentum (ultrarelativistic electron)
xkfe=np.sqrt( np.power(xmue,2) - np.power( m_el,2) )
# Electron density in fm-3
rhoe=np.power(xkfe,3)/(3*np.power(pi,2)*hbc3)
# Electron energy density in MeV*fm-3
edens_e=electron_eden_units*energy(xkfe/m_el)
x_el=rhoe/numden

# MUONS
# Muon density
rhomu=np.where( xmue>m_mu, rhop - rhoe, 0. )
# Muon chemical potential
xmumu=np.where( xmue>m_mu, xmue, 0. )
# Muon Fermi momentum
xkfmu=np.zeros(num_densities)
xkfmu=np.sqrt( np.power(xmumu,2) - np.power( m_mu,2) , where=xmue>m_mu )
# MUON ENERGY DENSITY
edens_m=muon_eden_units*energy(xkfmu/m_mu)
x_mu=rhomu/numden

# LEPTON ENERGY CONTRIBUTIONS
edens_l=edens_e+edens_m

# LEPTON PRESSURE CONTRIBUTIONS
p_el=electron_eden_units*pressure(xkfe/m_el)
p_mu=muon_eden_units*pressure(xkfmu/m_mu)
p_lep=p_el+p_mu

##############################################################################
# ADD CONTRIBUTIONS OF NUCLEONS, ELECTRONS AND MUONS
edens=edens_n + edens_l
press=p_nuc + p_lep


# EXPORTS DATA TO SCREEN
#fmt_out='{:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f} {:11.6f}'
#for iden,den in enumerate( numden ) :
#    print(fmt_out.format(edens[iden],press[iden],numden[iden]/rho0, \
#    numden[iden],xp[iden],xn[iden],x_el[iden],x_mu[iden]))

# PLOT PARTICLE FRACTIONS AS A FUNCTION OF DENSITY
fig, (ax1,ax2) = plt.subplots(2,sharex=True)
ax1.semilogy(numden,xn,label='Neutrons',lw=1.5)
ax1.semilogy(numden,xp,'--',lw=1.5,label='Protons')
ax1.semilogy(numden,x_el,'-.',lw=1.5,label='Electrons')
ax1.semilogy(numden,x_mu,':',lw=1.5,label='Muons')
ax1.legend(loc='lower right')
ax1.set_ylim([1e-3,1])
ax2.set(xlabel=r'Density, $ \rho $ fm$^{-3}$')
ax1.set(ylabel=r'Particle fractions, $x_i$')
ax2.set(ylabel=r'Pressure, $P$ [MeV fm$^{-3}$]')

# PLOT PRESSURE CONTRIBUTIONS
ax2.semilogy(numden,press,label='Total',lw=1.5)
ax2.semilogy(numden,p_nuc,'--',label='Nucleon',lw=1.5)
ax2.semilogy(numden,p_lep,'-.',label='Lepton',lw=1.5)
ax2.legend(loc='lower right')

plt.show()

# WRITING OUTPUT IN MR.dat FILE
#data_to_write = np.array( [edens[:],press[:],u[:]] ).T
#outputfile="EoS_HH.dat"
#with open(outputfile,"w+") as file_id :
#    np.savetxt(file_id,data_to_write,fmt=["%16.6E","%16.6E","%16.6E"],header="  eden [MeVfm-3]   Pres [MeVfm-3]     Den/0.16")
