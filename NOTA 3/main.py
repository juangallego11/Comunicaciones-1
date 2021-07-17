######################## LIBRERIAS ########################

from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift
from scipy.interpolate import interp1d
from scipy import signal
from scipy import integrate
from pylab import savefig

######################## FUNCIONES ########################

# Funcion encargada de traducir el archivo "P_1.wav" a datos tangibles
#
# Salida: 
#   data            - datos de la senal
#   rate            - tasa de muestreo
def abrirArchivo():
    rate,info = read("C:/Users/fbdav/Desktop/LAB-3-Redes/P_1.wav")
    dimension = info[0].size

    if dimension == 1:
    	data = info
    else:
    	data = info[:,dimension-1]

    return data,rate


# Funcion encargada de realizar la Modulacion AM con una frecuencia
# de portadora de 8000 Hz
#
# Entrada: 
#   data            - datos de la senal
#   rate            - tasa de muestreo
#
# Salida: 
#   AM              - senal modulada en AM
def modulacionAM(data,rate):
    largo = len(data)
    tiempo =  largo/float(rate)
    #frec=100 # Frecuencia en la que se modulará la senal
    frec=8000# Frecuencia en la que se modulará la senal 8KHz

    #Arreglo tiempo de la moduladora
    x = np.linspace(0, tiempo, largo)

    #Arreglo tiempo de la portadora
    t = np.linspace(0, tiempo, largo*10)    
    
    #Interpolación
    xt = np.interp(t, x, data)  

    graficar('Senal de Audio AM','Tiempo [s]','Amplitud [db]',t[:600],xt[:600],(0,0.008),(60,250),True,'Audio_AM')

	#Obtención de wct utilizando el arreglo tiempo de la portadora.
    wct = 10 * np.pi * frec * t  # Modulacion formula

	#Obtención de la senal modulada.
    AM = xt * np.cos(wct)
    
    return AM

# Funcion encargada de realizar la demodulacion AM retornando una 
# senal demodulada en AM
#
# Entrada: 
#   senal            - senal a demodular
#   tiempo           - tiempo de la senal
#   f_dem            - frecuencia de demodulacion
#
# Salida: 
#   demodulada     - senal demodulada
def demodulacionAM(senal, tiempo, f_dem):
    portadora = np.cos(10 * np.pi * f_dem * tiempo)
    demodulada = senal * portadora
    return demodulada



# Funcion encargada de aplicar la transformada de fourier sobre una senal
#
# Entrada: 
#   data            - datos de la senal
#   rate            - tasa de muestreo
#
# Salida: 
#   fourier            - datos de la senal transformada
#   ffreq              - frecuencia de la senal transformada
def fourier(data,rate):
    largo = len(data)
    tiempo = largo/rate
    fourier = np.fft.fft(data)
    k = np.arange(-len(fourier)/2,len(fourier)/2)
    #shift
    ffreq=fftshift(k/tiempo)
    return fourier,ffreq

# Funcion encargada de generar graficos
#
# Entrada: 
#   title            - titulo del grafico
#   xlabel           - titulo del eje x
#   ylabel           - titulo del eje y
#   X                - datos del eje x
#   Y                - datos del eje y
#   xl               - limites del eje x
#   yl               - limites del eje y
#   line             - bool del tipo de linea
#   fig              - titulo de la figura
def graficar(title,xlabel,ylabel,X,Y,xl,yl,line,fig):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xl[0], xl[1])
    plt.ylim(yl[0], yl[1])
    if line:
        plt.plot(X, Y, linewidth = 0.5)
    else:
        plt.plot(X, Y, '-')
    savefig(fig)
    plt.show()
    
    
# Funcion encargada de generar graficos de fourier
#
# Entrada: 
#   title            - titulo del grafico
#   xlabel           - titulo del eje x
#   ylabel           - titulo del eje y
#   X                - datos del eje x
#   Y                - datos del eje y
#   yl               - limites del eje y
#   fig              - titulo de la figura
def graficarFourier(title,xlabel,ylabel,X,Y,yl,fig):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(yl[0], yl[1])
    plt.plot(X, Y, '-')
    savefig(fig)
    plt.show()

# Funcion encargada de aplicar un filtro de pasabajo a la senal dada
#
# Entrada: 
#   data            - datos de la senal
#   rate            - tasa de muestreo
#
# Salida: 
#   filtrada        - senal filtrada
def filtro_pasabajo(data, rate):
    taps=1001
    nyq = rate/2
    corte=nyq*0.09
    coef_fir = signal.firwin(taps,corte/nyq, window = "hamming")
    filtrada = signal.lfilter(coef_fir,1.0,data)
    return filtrada


######################## BLOQUE PRINCIPAL ########################
data,rate = abrirArchivo()
largo = len(data)
tiempo = largo/rate
tiempo_resample = np.linspace(0,tiempo,largo*10)
t = np.linspace(0,tiempo,largo)
new_rate=rate*10

#f_portadora=100
f_portadora=8000

#-------> SENAL ORIGINAL <-------#

print('MOSTRANDO Y ALMACENANDO GRAFICOS DE LA SENAL ORIGINAL: ')

plt.title('Senal Original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.plot(t, data, '-')
savefig('Audio_Original')
plt.show()

interp = interp1d(t,data)
data_resample= interp(tiempo_resample)

Fdata,ffreq = fourier(data_resample,new_rate)
graficarFourier('Fourier senal original','Frecuencia [hz]','Amplitud [db]',ffreq,Fdata,(0,1.0e8),'Fourier_Original')

#-------> MODULACION AM <-------#

print('MOSTRANDO Y ALMACENANDO GRAFICOS DE LA SENAL MODULADA EN AM: ')

senalPortadora = np.cos(2*np.pi*f_portadora*tiempo_resample)
senalModulada = modulacionAM(data,rate)

graficar('Senal Portadora AM','Tiempo [s]','Amplitud [db]',tiempo_resample[:600],senalPortadora[:600],(0,0.0075),(-1,1),True,'Portadora_AM')
graficar('Senal Modulada AM','Tiempo [s]','Amplitud [db]',tiempo_resample[:600],senalModulada[:600],(0,0.0075),(-300,300),True,'Modulada_AM')

Fdata,ffreq=fourier(senalModulada,new_rate)
Pdata,Pfreq=fourier(senalPortadora,new_rate)
graficarFourier('Fourier Modulada AM','Frecuencia [hz]','Amplitud [db]',ffreq,Fdata,(0,0.6e8),'Fourier_AM')
graficarFourier('Fourier Portadora AM','Frecuencia [hz]','Amplitud [db]',Pfreq,Pdata,(0,2.8e5),'Fourier_Port_AM')

#-------> DEMODULACION AM <-------#

print('MOSTRANDO Y ALMACENANDO GRAFICOS DE LA SENAL DEMODULADA EN AM: ')

senalDemodulada = demodulacionAM(senalModulada,tiempo_resample,20000)
senalDemodulada = filtro_pasabajo(senalDemodulada,new_rate)

plt.title('Senal Demodulada AM')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [db]')
plt.plot(tiempo_resample,senalDemodulada, '-')
savefig('Demodulada_AM')
Fdata,ffreq=fourier(senalDemodulada,new_rate)
graficarFourier('Fourier senal Demodulada AM','Frecuencia [hz]','Amplitud [db]',ffreq,Fdata,(0,600),'Fourier_DemoAM')
plt.show()



