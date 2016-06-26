# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import sqlite3
import csv
import numpy as np

con = sqlite3.connect('data.sqlite')
cursor = con.cursor()

cursor.execute('select ID from LOCALIDADES where CIUDAD=?',('Alicante',))
CIUDAD_ID = cursor.fetchone()


with open('ALICANTE.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    reader.next()
    dia = 0
    for row in reader:
        ano = int(row[0])
        dia = dia + 1
        columnas = np.arange(3,19)
        for i in columnas:
            hora = i+2
            cursor.executemany('insert into RADIACION values (?,?,?,?,?)', \
            [(CIUDAD_ID[0],ano,dia,hora,int(row[i]))])
        
    con.commit()