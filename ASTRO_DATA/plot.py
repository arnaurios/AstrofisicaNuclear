# coding: utf-8
##########################################################################
# THIS PYTHON CODE SOLVES WHITE DWARF STRUCTURE EQUATIONS USING
# A FREE-FERMI GAS APPROXIMATION FOR THE EQUATION OF STATE ELECTRONS
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

df = pd.read_csv (r'Mass_Radius_WD.csv')
print (df)

#df.plot(x ='RadH', y='MassH', kind = 'scatter')
#df.errorbar(x ='RadH', y='MassH', yerr='e_MassH')
df[['RadH','MassH','e_RadH','e_MassH']].to_numpy()
plt.show()
#fig, ax = plt.subplots()
#plt.plot( WD_radius,WD_mass,'o')
#plt.xlabel('Radial coordinate [km]')
#plt.xlim(0,30000)
#plt.ylabel('Mass profile [M_sun]')
#plt.ylim(0,1.5)
##fig.savefig("WD.pdf")
#plt.show()
