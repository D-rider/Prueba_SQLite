# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import sqlite3
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import functions_00 as solar

con = sqlite3.connect('data.sqlite')
cur = con.cursor()

cur.execute('select GLOBAL_RADIATION from RADIACION where DIA=?',('355',))
radiacion = cur.fetchall()
for i in np.arange(0,len(radiacion)):
    radiacion[i] = radiacion[i][0]
    
cur.execute('select HORA from RADIACION where DIA=?',('355',))
hora = cur.fetchall()
for i in np.arange(0,len(hora)):
    hora[i] = hora[i][0]

azimuth = np.zeros(len(hora))
for i in hora:
    azimuth[i-5] = solar.solarAzimuth(1,i)

tck = interpolate.splrep(hora, radiacion, s=0)
hora_new = np.arange(5,20,0.1)
radiacion_new = interpolate.splev(hora_new,tck,der=0)

plt.plot(hora,radiacion,'o', hora_new,radiacion_new,'-')