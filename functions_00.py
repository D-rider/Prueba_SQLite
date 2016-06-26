# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 14:27 2016

Based on Eq_Time_1_2.py; Added equation for obtaining altitude and azimuth for
each hour o'clock.

@author: Jesús Rodríguez Venzal
"""
# Mandatory including
from matplotlib.path import Path
import matplotlib.patches as patches

# Include when running this script alone
import scipy.misc as sc
import numpy as np
import matplotlib.pyplot as plt
import logging

from __main__ import *

days = np.arange(1,366,1)
solstices = [172,355]

minutos = "minute"
grados = "deg"
radianes = "rad"

latitude = 37.178*np.pi/180
longitude = -3.6*np.pi/180
GMT = 1 #GMT time zone for local time

def daily_angle(_day):
    """ Daily angle in radians"""
            
    _daily_angle = 2*np.pi*(_day-1)/365.25
    
    return _daily_angle

def eqtime (_daily_angle,_rad_deg_min = "rad"):
    """Compute the equation of time in radians or minutes"""
#When calling this function it will return the value in radians unless otherwise specified
    _eqtime1 = 0.001868*np.cos(_daily_angle)-0.032077*np.sin(_daily_angle)
    _eqtime2 = -0.014615*np.cos(2*_daily_angle)-0.04089*np.sin(2*_daily_angle)
    _eqtime = 0.000075+_eqtime1+_eqtime2

# The result in radians is converted to the unit specified in the argument
    if _rad_deg_min == "rad":
        return _eqtime
    elif _rad_deg_min == "minute":
        return 229.18*_eqtime
    elif _rad_deg_min == "deg":
        return _eqtime*180/np.pi
        
def declination(_day, _rad_deg = "rad"):
    """Declination in radians"""
#When calling this function it will return the value in radians unless otherwise specified  
    _delta1 = -0.399912*np.cos(daily_angle(_day))+0.070257*np.sin(daily_angle(_day))
    _delta2 = -0.006758*np.cos(2*daily_angle(_day))+0.000907*np.sin(2*daily_angle(_day))
    _delta3 = -0.002697*np.cos(3*daily_angle(_day))+0.00148*np.sin(3*daily_angle(_day))
    _delta = 0.006918 + _delta1 + _delta2 + _delta3

# The result in radians is converted to the unit specified in the argument    
    if _rad_deg == "rad":
        return _delta
    elif _rad_deg == "deg":
        return _delta*180/np.pi
    
    

    
def sunRise(_decl,_latitude):
    """Sun rise/set angle in degrees"""
    
    return np.arccos(-np.tan(_decl)*np.tan(_latitude))*180/np.pi


hourAngle = np.linspace(-np.pi,np.pi,360)

altitude = np.empty(np.size(hourAngle))
altitude_deg = np.empty(np.size(hourAngle))

def solarAltitude(_day,_hour = None):
    """Solar altitude in radians"""
    if _hour is None:
        for i in range(np.size(hourAngle)):
            sin_alt = np.cos(latitude)*np.cos(declination(_day,radianes))*np.cos(hourAngle[i]) \
            +np.sin(latitude)*np.sin(declination(_day,radianes))
            altitude[i] = np.arcsin(sin_alt)
            altitude_deg[i] = altitude[i]*180/np.pi
        
    else:
        _hourAngle = (12-_hour)*np.pi/12
        sin_alt = np.cos(latitude)*np.cos(declination(_day,radianes))*np.cos(_hourAngle) \
        +np.sin(latitude)*np.sin(declination(_day,radianes))
        _altitude = np.arcsin(sin_alt)*180/np.pi
        return _altitude
        
    return
    
azimuth = np.empty(np.size(hourAngle))
azimuth_deg = np.empty(np.size(hourAngle))
derivada = np.empty(np.size(hourAngle))

def solarAzimuth(_day, _hour = None):
    """Solar azimuth in radians"""  
    def cos_azmth(_hourAngle):
        """Definition of the cosine of the azimuth replacing the altitude by 
        it formula"""
        return (np.sin(np.arcsin(np.cos(latitude)*np.cos(declination(_day,radianes))* \
        np.cos(_hourAngle)+np.sin(latitude)*np.sin(declination(_day,radianes))))*np.sin(latitude)- \
        np.sin(declination(_day,radianes))) \
        / (np.cos(np.arcsin(np.cos(latitude)*np.cos(declination(_day,radianes))* \
        np.cos(_hourAngle)+np.sin(latitude)*np.sin(declination(_day,radianes))))*np.cos(latitude))

    if _hour is None:
        for i in range(np.size(hourAngle)):
            derivada[i] = sc.derivative(cos_azmth,hourAngle[i],dx=1e-6)
            _signo = -np.sign(derivada[i])
            azimuth[i] =_signo * np.arccos(cos_azmth(hourAngle[i]))
            azimuth_deg[i] = azimuth[i]*180/np.pi
            
    else:
         _hourAngle = (12-int(_hour))*np.pi/12
         _derivada = sc.derivative(cos_azmth,_hourAngle,dx=1e-6)
         _signo = - np.sign(_derivada)
         if cos_azmth(_hourAngle) > 1:
             _azimuth = _signo * np.arccos(1) * 180 / np.pi
         else:
             _azimuth = _signo * np.arccos(cos_azmth(_hourAngle)) * 180 / np.pi
         return _azimuth
    
    return


def EqTime_plot():
    """ Calculate the solution of the equation of time in minutes to check the result"""
    _eqtime_minutos = np.empty(np.size(days))

    for i in days:
        _eqtime_minutos[i-1] = eqtime(daily_angle(i),minutos)
    
    plt.figure(figsize=[12,7.5])
    plt.plot(days, _eqtime_minutos,color="red", lw=1.5)
    plt.axis([0,365,-15,20])
    plt.title('Ecuacion del Tiempo')
    plt.xlabel("Dias")
    plt.ylabel("minutos")
    plt.savefig('Ecuacion_del_Tiempo')
    return


def Decl_plot():
    """Calculate the declination in degrees to check the result"""
    _declinacion = np.empty(np.size(days))
    for i in range(np.size(days)):
        _declinacion[i] = declination(days[i],grados)
        
    plt.figure(figsize=[12,7.5])
    plt.plot(days, _declinacion,color="blue", lw=1.5)
    #plt.axis([0,365,-15,20])
    plt.title("Declinacion")
    plt.xlabel("Dias")
    plt.ylabel("Grados")
    plt.savefig("Declinacion")
    return


def SunRise_print(_day):
    
    _grados = sunRise(declination(_day,radianes),latitude)
# Hour angle for sunrise is negative while it is positive for sunset
    _grados_amanecer = - _grados
    _grados_atardecer = _grados
    
    amanecer_horas = 12 + (_grados_amanecer - eqtime(daily_angle(_day),grados))*1/15 - \
    (longitude*180/np.pi)/15 + GMT
    atardecer_horas = 12 + (_grados_atardecer - eqtime(daily_angle(_day),grados))*1/15 -\
    longitude/15 + GMT
    _minutos_amanecer = int((amanecer_horas - int(amanecer_horas))*60)
    _hora_amanecer =  int(amanecer_horas)
    _minutos_atardecer = int((atardecer_horas - int(atardecer_horas))*60)
    _hora_atardecer =  int(atardecer_horas)
    
    if _minutos_amanecer < 10:
        print("Amanecerá a las %s:0%s hora solar")%(_hora_amanecer,_minutos_amanecer)
    else:
        print("Amanecerá a las %s:%s hora solar")%(_hora_amanecer,_minutos_amanecer)
    
    if _minutos_atardecer <10:
        print("Se pondrá el sol a las %s:0%s hora solar")%(_hora_atardecer,_minutos_atardecer)
    else:
        print("Se pondrá el sol a las %s:%s hora solar")%(_hora_atardecer,_minutos_atardecer)
    return
 

hourlyAltitude = np.empty((15,184))
hourlyAzimuth = np.empty((15,184))
hour_angle = 0  

def SunPaths_plot(_day = 79):
    _altitude = np.empty((3,np.size(hourAngle)))
    _azimuth = np.empty((3,np.size(hourAngle)))
    _days = np.hstack([solstices,_day])
    for i in range(3):
        solarAltitude(_days[i])
        solarAzimuth(_days[i])
        for j in range(np.size(altitude)):
            _altitude[i][j] = altitude[j]*180/np.pi
            _azimuth[i][j] = azimuth[j]*180/np.pi
    
#    _hourlyAltitude = np.empty((15,184))
#    _hourlyAzimuth = np.empty((15,184))
    for i in np.linspace(-7,7,15):
        #print("Hora: %s")%(12+i)
        for j in np.linspace(172,355,184):
            hour_angle = (12+i)*np.pi/12
            hourlyAzimuth[int(i)+7][int(j)-172] = solarAzimuth(int(j),12+i)
            hourlyAltitude[int(i)+7][int(j)-172] = solarAltitude(int(j),12+i)
#            if np.isnan(solarHour(int(j),12+i)[1]) or solarHour(int(j),12+i)[1]==0\
#            or solarHour(int(j),12+i)[0] <= 0:
#                    print('%s;%s;%s'%(round(hour_angle,3),\
#                    round(solarHour(int(j),12+i)[1],3),round(solarHour(int(j),12+i)[0],3)))
#                    logging.debug('%s;%s;%s'%(round(hour_angle,3),\
#                    round(solarHour(int(j),12+i)[1],3),round(solarHour(int(j),12+i)[0],3)))
                    
    fig = plt.figure(0,figsize=[12,7.5])
    #plt.plot(hourAngle,a_azimuth[1], color="red", lw=1.5)
    plt.plot(_azimuth[0], _altitude[0],color="blue", lw=1.5)
    plt.plot(_azimuth[1], _altitude[1],color="blue", lw=1.5)  
    plt.plot(_azimuth[2], _altitude[2],color="red",linestyle='--', lw=1.5)
    for i in np.linspace(-7,7,15):
        plt.plot(hourlyAzimuth[int(i)], hourlyAltitude[int(i)], color="black",\
        lw = 0.5)
    plt.axis([-130,130,0,80])
    _etiquetas = np.arange(-130,130,10)
    plt.xticks(_etiquetas,rotation = 45)
    plt.title("Solar Paths")
    plt.xlabel("Azimut")
    plt.ylabel("Altura solar")
    plt.savefig('SolarPaths') 
#    plt.title("Shadow Map")
#    plt.xlabel("Azimut")
#    plt.ylabel("Altura solar")
#    plt.savefig('Shadow Map')
    return

#SunPaths_plot()

def Shadow():
    """Plot the sun paths chart"""
    _altitude = np.empty((2,np.size(hourAngle)))
    _azimuth = np.empty((2,np.size(hourAngle)))
    for i in range(2):
        solarAltitude(solstices[i])
        solarAzimuth(solstices[i])
        for j in range(np.size(altitude)):
            _altitude[i][j] = altitude[j]*180/np.pi
            _azimuth[i][j] = azimuth[j]*180/np.pi
        
    fig = plt.figure(0,figsize=[12,7.5])
    graf = fig.add_subplot(111)
    #plt.plot(hourAngle,a_azimuth[1], color="red", lw=1.5)
    plt.plot(_azimuth[0], _altitude[0],color="blue", lw=1.5)
    plt.plot(_azimuth[1], _altitude[1],color="blue", lw=1.5)  
    plt.axis([-180,180,0,90])
    _etiquetas = np.arange(-180,180,10)
    plt.xticks(_etiquetas,rotation = 45)
    plt.title("Shadow Map")
    plt.xlabel("Azimut")
    plt.ylabel("Altura solar") 
    
    """Plot the shadows"""
    # Object 1
    #Calculate angles
    _x_a = -7.
    _y_a = 8.
    _x_b = -6.
    _y_b = 8.
    _h_1 = 15.
    _az_a = np.arctan(_x_a/_y_a)    # Azimuth of left point of the object
    _az_b = np.arctan(_x_b/_y_b)    # Azimuth of right point of the object
    _alt_a = np.arctan(_h_1/np.sqrt(_x_a**2+_y_a**2))   # Altitude of a
    _alt_b = np.arctan(_h_1/np.sqrt(_x_b**2+_y_b**2))   # Altitude of b
    
    print("Objeto 1:")
    print("\tAzimut izquierda:\t{}".format(_az_a*180/np.pi))
    print("\tAltura izquierda:\t{}".format(_alt_a*180/np.pi))
    print("\tAzimut derecha:\t\t{}".format(_az_b*180/np.pi))
    print("\tAltura derecha:\t\t{}".format(_alt_b*180/np.pi))
    
    verts1 = [
    (-41.19, 0.), # left, bottom
    (-41.19, 54.68), # left, top
    (-36.87, 56.31), # right, top
    (-36.87, 0.), # right, bottom
    (0., 0.), # ignored
    ]
    
    # Object 2
    verts2 = [
    (26.57, 0.), # left, bottom
    (26.57, 30.81), # left, top
    (45, 25.24), # right, top
    (45, 0.), # right, bottom
    (0., 0.), # ignored
    ]

    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]

    path1 = Path(verts1, codes)
    path2 = Path(verts2, codes)

    patch1 = patches.PathPatch(path1, facecolor='orange') #lw=2
    patch2 = patches.PathPatch(path2, facecolor='green') #lw=2
    plt.figure(0)
    graf.add_patch(patch1)
    graf.add_patch(patch2)    
    #plt.show()
    plt.savefig('ShadowMap')
    
    """Calculate losses due to shadows"""
    #Object 1
    
#    for i in range(np.size(_azimuth[1])):
#        if _azimuth[1][i] > -41.19:
#            _max_az = i
#        if _azimuth[1][i] > -36.87:
#            _min_az = i            
#       
#    dx1 = (41.19-36.87)/(_max_az-_min_az)
#    x1 = np.linspace(-41.19,-36.87,_max_az-_min_az)
#    
#    def parte_superior1(_x):
#        _m = (56.31-54.68)/(-36.87-(-41.19))
#        return 54.68 + _m * (_x-(-41.19))
#        
#    _losses_area = 0.0
#    for j in np.arange(_min_az,_max_az,1):
#        _losses_area = _losses_area + (parte_superior1(x1[j-_min_az])-_altitude[1][j])*dx1
#        
#    #Object 2
#    
#    for i in range(np.size(_azimuth[1])):
#        if _azimuth[1][i] > 26.57:
#            _min_az = i
#        elif _azimuth[1][i] > 45:
#            _max_az = i
#    
#    dx2 = (45-26.57)/(_max_az-_min_az)
#    x2 = np.linspace(26.57,45,_max_az-_min_az)
#    def parte_superior2(_x):
#        _m = (25.24-30.81)/(45-26.57)
#        return 30.81 + _m * (_x-26.57)
#    for j in np.arange(_min_az,_max_az,1):
#        _losses_area = _losses_area + (parte_superior2(x2[j-_min_az])-_altitude[1][j])*dx2
#        
#    _sun_area = 0.0
#    dx3 = (_azimuth[0][1]-_azimuth[0][358])/np.size(_azimuth[0])
#    for i in range(np.size(_azimuth[1])):
#        if _altitude[1][i] <= 0:
#            _sun_area = _sun_area + (0+_altitude[0][i])*dx3
#        else:
#            _sun_area = _sun_area + (_altitude[0][i] - _altitude[1][i])*dx3
#        
#    _losses = np.around((_losses_area/_sun_area)*100,1)
#    
#    print("Pérdidas por sombras de un %s %%")%(_losses)

#Shadow()