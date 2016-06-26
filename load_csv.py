# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import sqlite3
import csv
import numpy as np
import datetime

con = sqlite3.connect('data.sqlite')
cursor = con.cursor()

cursor.execute('select ID from LOCALIDADES where LOCALIDAD=?',('Alicante',))
CIUDAD_ID = cursor.fetchone()


with open('ALICANTE.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    reader.next()
    dia = 0
    for row in reader:
        agno = int(row[0])
        mes = int(row[1])
        dia = int(row[2])
        year_day = datetime.date(agno,mes,dia).toordinal()\
            - datetime.date(agno,1,1).toordinal() +1
        columnas = np.arange(3,19)
        for i in columnas:
            hora = i+2
            cursor.executemany('insert into RADIACION values (?,?,?,?,?)', \
            [(CIUDAD_ID[0],agno,year_day,hora,int(row[i]))])
        
    con.commit()