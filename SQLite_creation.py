# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 10:01:03 2016

@author: Jesus
"""

import sqlite3
con = sqlite3.connect('data.sqlite')
cur = con.cursor()

con.execute('''create table LOCALIDADES(
    ID          INT     PRIMARY KEY     UNIQUE  NOT NULL,
    LOCALIDAD      TEXT    UNIQUE,
    LATITUD     REAL    UNIQUE,
    LONGITUD    REAL    UNIQUE)''')

alicante =[(int(00),"Alicante",round(float(38.3453),4),round(float(-0.4831),4))]
ALICANTE = [(0,'Alicante',38.3254,-0.4831)]

con.executemany('insert into LOCALIDADES(ID,CIUDAD, LATITUD, LONGITUD) \
    values (?,?,?,?)', ALICANTE)

con.execute('''create table RADIACION(
    ID          INT     PRIMARY KEY     NOT NULL,
    CIUDAD_ID   INT     NOT NULL,
    AÑO         INT     NOT NULL,
    DIA         INT     NOT NULL,
    HORA         INT     NOT NULL,
    GLOBAL_RADIATON  REAL)''')
    
con.commit()
    
