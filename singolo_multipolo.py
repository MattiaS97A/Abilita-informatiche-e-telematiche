import astropy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import statistics
import numpy
from astropy.io import fits

#Primo set di misure, multipolo 0

#Innanzitutto vengono letti i dati dal set di misure:

Nbins=200 				#lunghezza del vettore di dati
Nmisure=1000				#numero di misure considerate (fino a 10000)
test=1					#variabile che identifica il set di misure considerato

print('Caricamento multipolo:')

#Lettura multipolo 0
dati1=[]				#definisco la lista vuota dati1
for i in np.arange(Nmisure)+1:
    filename= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file=fits.open(filename)		#apro il file
    table= file[1].data.copy()		
    dati1.append(table['XI0'])		#aggiungo alla lista dati1 i dati presenti nella colonna XI0
    if i==1:
        scale = table['SCALE']
    del table
    file.close()
dati1=np.asarray(dati1).transpose()	#trasformo la lista dati1 in un array e lo traspongo
print('Multipolo 0 caricato')


# ------------------------- Covarianza Teorica -------------------------

sigma = [0.02, 0.02, 0.02]
ls = [25, 50, 75]

def covf(x1, x2, sig, l):				#funzione che calcola la covarianza teorica per l'autocorrelazione di un singolo multipolo
    return sig**2.*np.exp(-(x1 - x2)**2./(2.*l**2.))

Cth1 = np.zeros((Nbins,Nbins),dtype=float)		#definisco una matricie di zeri

for i in range(Nbins):					#riempio la matrice con i valori delle covarianze calcolate con le funzioni di cui sopra
    for j in range(Nbins):
        Cth1[i,j] = covf(scale[i],scale[j],sigma[0],ls[0])

print('Plot della matrice delle covarianze teoriche:')
plt.title('Matrice delle covarianze teoriche')
plt.imshow(Cth1)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{th}$')
plt.show()

# ------------------------- Covarianza Misurata -------------------------

media1 = np.zeros((Nbins),dtype=float)			#definisco delle matrici di zeri
Cmes1 = np.zeros((Nbins,Nbins),dtype=float)
media1 = np.sum(dati1, axis=1)/Nmisure			#calcolo i valori medi nella lista delle misure

for i in range(Nbins):					#Calcolo le covarianze misurate secondo la definizione di covarianza
    for j in range(Nbins):
        q=np.sum((dati1[i]-media1[i])*(dati1[j]-media1[j]))
        Cmes1[i,j] = q/(Nmisure-1.)

print('Plot della matrice delle covarianze misurate:')
plt.title('Matrice delle covarianze misurate')
plt.imshow(Cmes1)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{mes}$')
plt.show()


# ------------------------- Residui -------------------------

#Per calcolare i residui bisogna calcolare la matrice di correlazione

R1=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 0
for i in range(Nbins):
    for j in range(Nbins):
        R1[i,j]=Cth1[i,j]**2./math.sqrt(Cth1[i,i]*Cth1[j,j]**2.)

Res1=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 0
for i in range(Nbins):
    for j in range(Nbins):
        Res1[i,j]=(Cth1[i,j]-Cmes1[i,j])*np.sqrt((Nmisure - 1.)/((1.+R1[i,j])*Cth1[i,i]*Cth1[j,j]))

print('Plot della matrice dei residui:')
plt.title('Matrice dei residui')
plt.imshow(Res1)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ Res$')
plt.show()

print('La deviazione standard dei residui è:', np.std(Res1))
