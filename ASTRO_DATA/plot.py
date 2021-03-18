# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES WHITE DWARF STRUCTURE EQUATIONS USING
# A FREE-FERMI GAS APPROXIMATION FOR THE EQUATION OF STATE ELECTRONS
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

df = pd.read_csv (r'Mass_Radius_WD.csv',skiprows=4)
print (df)

#df.plot(x ='RadH', y='MassH', kind = 'scatter')
#df.errorbar(x ='RadH', y='MassH', yerr='e_MassH')
df[['RadG','MassG','e_RadG','e_MassG']].to_numpy()

units_radius=696340/100
radius=df.RadG*units_radius
error_radius=df.e_RadG*units_radius
mass=df.MassH
error_mass=df.e_MassG
fig, ax = plt.subplots()
plt.errorbar(radius,mass,xerr=error_radius,yerr=error_mass,fmt='ok')
plt.xlabel('Radius [km]')
plt.xlim(0,20000)
plt.ylabel('Mass [M_sun]')
plt.ylim(0,1.5)
##fig.savefig("WD.pdf")
#plt.show()
plt.show()
