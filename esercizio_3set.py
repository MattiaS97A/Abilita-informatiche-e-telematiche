import astropy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import statistics
import numpy
from astropy.io import fits

#L'esercizio è diviso in 3 parti simili, una per ognuno dei 3 set di misure.
#Ogni parte è divisa in 4 sezioni principali: lettura dei multipoli, calcolo della covarianza teorica, calcolo della covarianza misurata, calcolo dei residui.


####################################################################
##################### PRIMO SET DI MISURE ##########################
####################################################################


#Innanzitutto vengono letti i dati dal set di misure:

Nbins=200 				#lunghezza del vettore di dati
Nmisure=1000				#numero di misure considerate (fino a 10000)
test=1					#variabile che identifica il set di misure considerato

print('PRIMO SET DI MISURE')
print('Caricamento multipoli:')

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


#Lettura multipolo 2
dati2=[]
for i in np.arange(Nmisure)+1:
    filename2= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file2=fits.open(filename2)
    table2= file2[1].data.copy()
    dati2.append(table2['XI2'])
    if i==1:
        scale2 = table2['SCALE']
    del table2
    file.close()
dati2=np.asarray(dati2).transpose()
print('Multipolo 2 caricato')


#Lettura multipolo 4
dati3=[]
for i in np.arange(Nmisure)+1:
    filename3= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file3=fits.open(filename3)
    table3= file3[1].data.copy()
    dati3.append(table3['XI4'])
    if i==1:
        scale3 = table3['SCALE']
    del table3
    file.close()
dati3=np.asarray(dati3).transpose()
print('Multipolo 4 caricato')


#print('Controllo che i dati abbiano il giusto formato:')
#print(dati1.shape) #1000 measures for 200 bin
#print(dati2.shape)
#print(dati3.shape)



# ------------------------- Covarianza Teorica -------------------------

sigma = [0.02, 0.02, 0.02]
ls = [25, 50, 75]

def covf(x1, x2, sig, l):				#funzione che calcola la covarianza teorica per l'autocorrelazione di un singolo multipolo
    return sig**2.*np.exp(-(x1 - x2)**2./(2.*l**2.))

def covf1f2(x1, x2, sig1, l1, sig2, l2):		#funzione che calcola la covarianza teorica per la correlazione tra due multipoli diversi
    return (np.sqrt(2.*l1*l2)*np.exp(-(np.sqrt((x1 - x2)**2.)**2./(l1**2. + l2**2.)))*sig1*sig2)/np.sqrt(l1**2. + l2**2.)

Cth1 = np.zeros((Nbins,Nbins),dtype=float)		#definisco delle matrici di zeri
Cth2= np.zeros((Nbins,Nbins),dtype=float)
Cth3 = np.zeros((Nbins,Nbins),dtype=float)
Cth12 = np.zeros((Nbins,Nbins),dtype=float)
Cth13 = np.zeros((Nbins,Nbins),dtype=float)
Cth23 = np.zeros((Nbins,Nbins),dtype=float)

for i in range(Nbins):					#riempio le matrici con i valori delle covarianze calcolate con le funzioni di cui sopra
    for j in range(Nbins):
        Cth1[i,j] = covf(scale[i],scale[j],sigma[0],ls[0])
        Cth2[i,j] = covf(scale2[i],scale2[j],sigma[1],ls[1])
        Cth3[i,j] = covf(scale3[i],scale3[j],sigma[2],ls[2])
        Cth12[i,j] = covf1f2(scale[i],scale[j],sigma[0],ls[0],sigma[1],ls[1])
        Cth13[i,j] = covf1f2(scale[i],scale[j],sigma[0],ls[0],sigma[2],ls[2])
        Cth23[i,j] = covf1f2(scale[i],scale[j],sigma[1],ls[1],sigma[2],ls[2])


matriceCth=np.zeros((3*Nbins,3*Nbins), dtype=float)		#costruisco la matrice delle covarianze teoriche per i 3 multipoli

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceCth[I,J]=Cth1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceCth[I,J]=Cth12[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceCth[I,J]=Cth13[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth23[i,j]

print('Plot della matrice delle covarianze teoriche:')
plt.title('Matrice delle covarianze teoriche (primo set)')
plt.imshow(matriceCth)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{th}$')
plt.show()


# ------------------------- Covarianza Misurata -------------------------

media1 = np.zeros((Nbins),dtype=float)			#definisco delle matrici di zeri
media2 = np.zeros((Nbins),dtype=float)
media3 = np.zeros((Nbins),dtype=float)
Cmes1 = np.zeros((Nbins,Nbins),dtype=float)
Cmes2 = np.zeros((Nbins,Nbins),dtype=float)
Cmes12 = np.zeros((Nbins,Nbins),dtype=float)
Cmes13 = np.zeros((Nbins,Nbins),dtype=float)
Cmes23 = np.zeros((Nbins,Nbins),dtype=float)
Cmes3 = np.zeros((Nbins,Nbins),dtype=float)
media1 = np.sum(dati1, axis=1)/Nmisure			#calcolo i valori medi nelle liste delle misure
media2 = np.sum(dati2, axis=1)/Nmisure
media3 = np.sum(dati3, axis=1)/Nmisure

for i in range(Nbins):					#Calcolo le covarianze misurate secondo la definizione di covarianza
    for j in range(Nbins):
        q=np.sum((dati1[i]-media1[i])*(dati1[j]-media1[j]))
        Cmes1[i,j] = q/(Nmisure-1.)
        q2=np.sum((dati2[i]-media2[i])*(dati2[j]-media2[j]))
        Cmes2[i,j] = q2/(Nmisure-1.)
        q3=np.sum((dati3[i]-media3[i])*(dati3[j]-media3[j]))
        Cmes3[i,j] = q3/(Nmisure-1.)
        q12=np.sum((dati1[i]-media1[i])*(dati2[j]-media2[j]))
        Cmes12[i,j] = q12/(Nmisure-1.)
        q13=np.sum((dati1[i]-media1[i])*(dati3[j]-media3[j]))
        Cmes13[i,j] = q13/(Nmisure-1.)
        q23=np.sum((dati2[i]-media2[i])*(dati3[j]-media3[j]))
        Cmes23[i,j] = q23/(Nmisure-1.)


matriceCmes=np.zeros((3*Nbins,3*Nbins), dtype=float)		#costruisco la matrice delle covarianze misurate per i 3 multipoli

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes12[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes13[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes23[i,j]

print('Plot della matrice delle covarianze misurate:')
plt.title('Matrice delle covarianze misurate (primo set)')
plt.imshow(matriceCmes)
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

R2=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 2
for i in range(Nbins):
    for j in range(Nbins):
        R2[i,j]=Cth2[i,j]**2./math.sqrt(Cth2[i,i]*Cth2[j,j]**2.)

Res2=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 2
for i in range(Nbins):
    for j in range(Nbins):
            Res2[i,j]=(Cth2[i,j]-Cmes2[i,j])*np.sqrt((Nmisure - 1.)/((1.+R2[i,j])*Cth2[i,i]*Cth2[j,j]))

R3=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 4
for i in range(Nbins):
    for j in range(Nbins):
        R3[i,j]=Cth3[i,j]**2./math.sqrt(Cth3[i,i]*Cth3[j,j]**2.)

Res3=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 4
for i in range(Nbins):
    for j in range(Nbins):
            Res3[i,j]=(Cth3[i,j]-Cmes3[i,j])*np.sqrt((Nmisure - 1.)/((1.+R3[i,j])*Cth3[i,i]*Cth3[j,j]))

R12=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 0 e 2
for i in range(Nbins):
    for j in range(Nbins):
        R12[i,j]=Cth12[i,j]**2./math.sqrt(Cth12[i,i]*Cth12[j,j]**2.)

Res12=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 0 e 2
for i in range(Nbins):
    for j in range(Nbins):
        Res12[i,j]=(Cth12[i,j]-Cmes12[i,j])*np.sqrt((Nmisure - 1.)/((1.+R12[i,j])*Cth12[i,i]*Cth12[j,j]))

R13=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 0 e 4
for i in range(Nbins):
    for j in range(Nbins):
        R13[i,j]=Cth13[i,j]**2./math.sqrt(Cth13[i,i]*Cth13[j,j]**2.)

Res13=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 0 e 4
for i in range(Nbins):
    for j in range(Nbins):
        Res13[i,j]=(Cth13[i,j]-Cmes13[i,j])*np.sqrt((Nmisure - 1)/((1+R13[i,j])*Cth13[i,i]*Cth13[j,j]))

R23=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 2 e 4
for i in range(Nbins):
    for j in range(Nbins):
        R23[i,j]=Cth23[i,j]**2./np.sqrt(Cth23[i,i]*Cth23[j,j]**2.)

Res23=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 2 e 4
for i in range(Nbins):
    for j in range(Nbins):
        Res23[i,j]=(Cth23[i,j]-Cmes23[i,j])*np.sqrt((Nmisure - 1.)/((1.+R23[i,j])*Cth23[i,i]*Cth23[j,j]))


Res12t=np.transpose(Res12)				#traspongo le matrici di correlazione per multipoli diversi
Res13t=np.transpose(Res13)
Res23t=np.transpose(Res23)


matriceRes=np.zeros((3*Nbins,3*Nbins), dtype=float)	#costruisco la matrice dei residui per i 3 multipoli

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceRes[I,J]=Res1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceRes[I,J]=Res12t[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceRes[I,J]=Res13t[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res23t[i,j]

print('Plot della matrice dei residui:')
plt.title('Matrice dei residui (primo set)')
plt.imshow(matriceRes)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ Res$')
plt.show()

print('La deviazione standard dei residui è:', np.std(matriceRes))



####################################################################
#################### SECONDO SET DI MISURE #########################
####################################################################

test=2

print('')
print('SECONDO SET DI MISURE')
print('Caricamento multipoli:')

#Lettura multipolo 0
dati1=[]
for i in np.arange(Nmisure)+1:
    filename= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file=fits.open(filename)
    table= file[1].data.copy()
    dati1.append(table['XI0'])
    if i==1:
        scale = table['SCALE']
    del table
    file.close()
dati1=np.asarray(dati1).transpose()
print('Multipolo 0 caricato')


#Lettura multipolo 2
dati2=[]
for i in np.arange(Nmisure)+1:
    filename2= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file2=fits.open(filename2)
    table2= file2[1].data.copy()
    dati2.append(table2['XI2'])
    if i==1:
        scale2 = table2['SCALE']
    del table2
    file.close()
dati2=np.asarray(dati2).transpose()
print('Multipolo 2 caricato')


#Lettura multipolo 4
dati3=[]
for i in np.arange(Nmisure)+1:
    filename3= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file3=fits.open(filename3)
    table3= file3[1].data.copy()
    dati3.append(table3['XI4'])
    if i==1:
        scale3 = table3['SCALE']
    del table3
    file.close()
dati3=np.asarray(dati3).transpose()
print('Multipolo 4 caricato')


# ------------------------- Covarianza Teorica -------------------------

sigma = [0.02, 0.01, 0.005]
ls = [50, 50, 50]

Cth1 = np.zeros((Nbins,Nbins),dtype=float)
Cth2= np.zeros((Nbins,Nbins),dtype=float)
Cth3 = np.zeros((Nbins,Nbins),dtype=float)
Cth12 = np.zeros((Nbins,Nbins),dtype=float)
Cth13 = np.zeros((Nbins,Nbins),dtype=float)
Cth23 = np.zeros((Nbins,Nbins),dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        Cth1[i,j] = covf(scale[i],scale[j],sigma[0],ls[0])
        Cth2[i,j] = covf(scale2[i],scale2[j],sigma[1],ls[1])
        Cth3[i,j] = covf(scale3[i],scale3[j],sigma[2],ls[2])
        Cth12[i,j] = covf1f2(scale[i],scale[j],sigma[0],ls[0],sigma[1],ls[1])
        Cth13[i,j] = covf1f2(scale[i],scale[j],sigma[0],ls[0],sigma[2],ls[2])
        Cth23[i,j] = covf1f2(scale[i],scale[j],sigma[1],ls[1],sigma[2],ls[2])


matriceCth=np.zeros((3*Nbins,3*Nbins), dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceCth[I,J]=Cth1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceCth[I,J]=Cth12[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceCth[I,J]=Cth13[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth23[i,j]

print('Plot della matrice delle covarianze teoriche:')
plt.title('Matrice delle covarianze teoriche (secondo set)')
plt.imshow(matriceCth)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{th}$')
plt.show()


# ------------------------- Covarianza Misurata -------------------------

media1 = np.zeros((Nbins),dtype=float)
media2 = np.zeros((Nbins),dtype=float)
media3 = np.zeros((Nbins),dtype=float)
Cmes1 = np.zeros((Nbins,Nbins),dtype=float)
Cmes2 = np.zeros((Nbins,Nbins),dtype=float)
Cmes12 = np.zeros((Nbins,Nbins),dtype=float)
Cmes13 = np.zeros((Nbins,Nbins),dtype=float)
Cmes23 = np.zeros((Nbins,Nbins),dtype=float)
Cmes3 = np.zeros((Nbins,Nbins),dtype=float)
media1 = np.sum(dati1, axis=1)/Nmisure
media2 = np.sum(dati2, axis=1)/Nmisure
media3 = np.sum(dati3, axis=1)/Nmisure

for i in range(Nbins):
    for j in range(Nbins):
        q=np.sum((dati1[i]-media1[i])*(dati1[j]-media1[j]))
        Cmes1[i,j] = q/(Nmisure-1.)
        q2=np.sum((dati2[i]-media2[i])*(dati2[j]-media2[j]))
        Cmes2[i,j] = q2/(Nmisure-1.)
        q3=np.sum((dati3[i]-media3[i])*(dati3[j]-media3[j]))
        Cmes3[i,j] = q3/(Nmisure-1.)
        q12=np.sum((dati1[i]-media1[i])*(dati2[j]-media2[j]))
        Cmes12[i,j] = q12/(Nmisure-1.)
        q13=np.sum((dati1[i]-media1[i])*(dati3[j]-media3[j]))
        Cmes13[i,j] = q13/(Nmisure-1.)
        q23=np.sum((dati2[i]-media2[i])*(dati3[j]-media3[j]))
        Cmes23[i,j] = q23/(Nmisure-1.)


matriceCmes=np.zeros((3*Nbins,3*Nbins), dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes12[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes13[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes23[i,j]

print('Plot della matrice delle covarianze misurate:')
plt.title('Matrice delle covarianze misurate (secondo set)')
plt.imshow(matriceCmes)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{mes}$')
plt.show()


# ------------------------- Residui -------------------------

R1=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 0
for i in range(Nbins):
    for j in range(Nbins):
        R1[i,j]=Cth1[i,j]**2./math.sqrt(Cth1[i,i]*Cth1[j,j]**2.)

Res1=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 0
for i in range(Nbins):
    for j in range(Nbins):
        Res1[i,j]=(Cth1[i,j]-Cmes1[i,j])*np.sqrt((Nmisure - 1.)/((1.+R1[i,j])*Cth1[i,i]*Cth1[j,j]))

R2=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 2
for i in range(Nbins):
    for j in range(Nbins):
        R2[i,j]=Cth2[i,j]**2./math.sqrt(Cth2[i,i]*Cth2[j,j]**2.)

Res2=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 2
for i in range(Nbins):
    for j in range(Nbins):
            Res2[i,j]=(Cth2[i,j]-Cmes2[i,j])*np.sqrt((Nmisure - 1.)/((1.+R2[i,j])*Cth2[i,i]*Cth2[j,j]))

R3=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 4
for i in range(Nbins):
    for j in range(Nbins):
        R3[i,j]=Cth3[i,j]**2./math.sqrt(Cth3[i,i]*Cth3[j,j]**2.)

Res3=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 4
for i in range(Nbins):
    for j in range(Nbins):
            Res3[i,j]=(Cth3[i,j]-Cmes3[i,j])*np.sqrt((Nmisure - 1.)/((1.+R3[i,j])*Cth3[i,i]*Cth3[j,j]))

R12=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 0 e 2
for i in range(Nbins):
    for j in range(Nbins):
        R12[i,j]=Cth12[i,j]**2./math.sqrt(Cth12[i,i]*Cth12[j,j]**2.)

Res12=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 0 e 2
for i in range(Nbins):
    for j in range(Nbins):
        Res12[i,j]=(Cth12[i,j]-Cmes12[i,j])*np.sqrt((Nmisure - 1.)/((1.+R12[i,j])*Cth12[i,i]*Cth12[j,j]))

R13=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 0 e 4
for i in range(Nbins):
    for j in range(Nbins):
        R13[i,j]=Cth13[i,j]**2./math.sqrt(Cth13[i,i]*Cth13[j,j]**2.)

Res13=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 0 e 4
for i in range(Nbins):
    for j in range(Nbins):
        Res13[i,j]=(Cth13[i,j]-Cmes13[i,j])*np.sqrt((Nmisure - 1)/((1+R13[i,j])*Cth13[i,i]*Cth13[j,j]))

R23=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 2 e 4
for i in range(Nbins):
    for j in range(Nbins):
        R23[i,j]=Cth23[i,j]**2./np.sqrt(Cth23[i,i]*Cth23[j,j]**2.)

Res23=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 2 e 4
for i in range(Nbins):
    for j in range(Nbins):
        Res23[i,j]=(Cth23[i,j]-Cmes23[i,j])*np.sqrt((Nmisure - 1.)/((1.+R23[i,j])*Cth23[i,i]*Cth23[j,j]))


Res12t=np.transpose(Res12)
Res13t=np.transpose(Res13)
Res23t=np.transpose(Res23)

matriceRes=np.zeros((3*Nbins,3*Nbins), dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceRes[I,J]=Res1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceRes[I,J]=Res12t[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceRes[I,J]=Res13t[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res23t[i,j]

print('Plot della matrice dei residui:')
plt.title('Matrice dei residui (secondo set)')
plt.imshow(matriceRes)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ Res$')
plt.show()

print('La deviazione standard dei residui è:', np.std(matriceRes))


####################################################################
##################### TERZO SET DI MISURE ##########################
####################################################################

test=3

print('')
print('TERZO SET DI MISURE')
print('Caricamento multipoli:')

#Lettura multipolo 0
dati1=[]
for i in np.arange(Nmisure)+1:
    filename= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file=fits.open(filename)
    table= file[1].data.copy()
    dati1.append(table['XI0'])
    if i==1:
        scale = table['SCALE']
    del table
    file.close()
dati1=np.asarray(dati1).transpose()
print('Multipolo 0 caricato')


#Lettura multipolo 2
dati2=[]
for i in np.arange(Nmisure)+1:
    filename2= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file2=fits.open(filename2)
    table2= file2[1].data.copy()
    dati2.append(table2['XI2'])
    if i==1:
        scale2 = table2['SCALE']
    del table2
    file.close()
dati2=np.asarray(dati2).transpose()
print('Multipolo 2 caricato')


#Lettura multipolo 4
dati3=[]
for i in np.arange(Nmisure)+1:
    filename3= f'/home/mattia/Desktop/Esercizio finale/data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
    file3=fits.open(filename3)
    table3= file3[1].data.copy()
    dati3.append(table3['XI4'])
    if i==1:
        scale3 = table3['SCALE']
    del table3
    file.close()
dati3=np.asarray(dati3).transpose()
print('Multipolo 4 caricato')


# ------------------------- Covarianza Teorica -------------------------

sigma = [0.02, 0.01, 0.005]
ls = [5, 5, 5]

Cth1 = np.zeros((Nbins,Nbins),dtype=float)
Cth2= np.zeros((Nbins,Nbins),dtype=float)
Cth3 = np.zeros((Nbins,Nbins),dtype=float)
Cth12 = np.zeros((Nbins,Nbins),dtype=float)
Cth13 = np.zeros((Nbins,Nbins),dtype=float)
Cth23 = np.zeros((Nbins,Nbins),dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        Cth1[i,j] = covf(scale[i],scale[j],sigma[0],ls[0])
        Cth2[i,j] = covf(scale2[i],scale2[j],sigma[1],ls[1])
        Cth3[i,j] = covf(scale3[i],scale3[j],sigma[2],ls[2])
        Cth12[i,j] = covf1f2(scale[i],scale[j],sigma[0],ls[0],sigma[1],ls[1])
        Cth13[i,j] = covf1f2(scale[i],scale[j],sigma[0],ls[0],sigma[2],ls[2])
        Cth23[i,j] = covf1f2(scale[i],scale[j],sigma[1],ls[1],sigma[2],ls[2])


matriceCth=np.zeros((3*Nbins,3*Nbins), dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceCth[I,J]=Cth1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceCth[I,J]=Cth12[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceCth[I,J]=Cth23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceCth[I,J]=Cth13[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceCth[I,J]=Cth23[i,j]

print('Plot della matrice delle covarianze teoriche:')
plt.title('Matrice delle covarianze teoriche (terzo set)')
plt.imshow(matriceCth)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{th}$')
plt.show()


# ------------------------- Covarianza Misurata -------------------------

media1 = np.zeros((Nbins),dtype=float)
media2 = np.zeros((Nbins),dtype=float)
media3 = np.zeros((Nbins),dtype=float)
Cmes1 = np.zeros((Nbins,Nbins),dtype=float)
Cmes2 = np.zeros((Nbins,Nbins),dtype=float)
Cmes12 = np.zeros((Nbins,Nbins),dtype=float)
Cmes13 = np.zeros((Nbins,Nbins),dtype=float)
Cmes23 = np.zeros((Nbins,Nbins),dtype=float)
Cmes3 = np.zeros((Nbins,Nbins),dtype=float)
media1 = np.sum(dati1, axis=1)/Nmisure
media2 = np.sum(dati2, axis=1)/Nmisure
media3 = np.sum(dati3, axis=1)/Nmisure

for i in range(Nbins):
    for j in range(Nbins):
        q=np.sum((dati1[i]-media1[i])*(dati1[j]-media1[j]))
        Cmes1[i,j] = q/(Nmisure-1.)
        q2=np.sum((dati2[i]-media2[i])*(dati2[j]-media2[j]))
        Cmes2[i,j] = q2/(Nmisure-1.)
        q3=np.sum((dati3[i]-media3[i])*(dati3[j]-media3[j]))
        Cmes3[i,j] = q3/(Nmisure-1.)
        q12=np.sum((dati1[i]-media1[i])*(dati2[j]-media2[j]))
        Cmes12[i,j] = q12/(Nmisure-1.)
        q13=np.sum((dati1[i]-media1[i])*(dati3[j]-media3[j]))
        Cmes13[i,j] = q13/(Nmisure-1.)
        q23=np.sum((dati2[i]-media2[i])*(dati3[j]-media3[j]))
        Cmes23[i,j] = q23/(Nmisure-1.)

matriceCmes=np.zeros((3*Nbins,3*Nbins), dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes12[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceCmes[I,J]=Cmes23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceCmes[I,J]=Cmes13[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceCmes[I,J]=Cmes23[i,j]

print('Plot della matrice delle covarianze misurate:')
plt.title('Matrice delle covarianze misurate (terzo set)')
plt.imshow(matriceCmes)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ C_{mes}$')
plt.show()


# ------------------------- Residui -------------------------

R1=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 0
for i in range(Nbins):
    for j in range(Nbins):
        R1[i,j]=Cth1[i,j]**2./math.sqrt(Cth1[i,i]*Cth1[j,j]**2.)

Res1=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 0
for i in range(Nbins):
    for j in range(Nbins):
        Res1[i,j]=(Cth1[i,j]-Cmes1[i,j])*np.sqrt((Nmisure - 1.)/((1.+R1[i,j])*Cth1[i,i]*Cth1[j,j]))

R2=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 2
for i in range(Nbins):
    for j in range(Nbins):
        R2[i,j]=Cth2[i,j]**2./math.sqrt(Cth2[i,i]*Cth2[j,j]**2.)

Res2=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 2
for i in range(Nbins):
    for j in range(Nbins):
            Res2[i,j]=(Cth2[i,j]-Cmes2[i,j])*np.sqrt((Nmisure - 1.)/((1.+R2[i,j])*Cth2[i,i]*Cth2[j,j]))

R3=np.zeros((Nbins,Nbins), dtype=float)			#Matrice di correlazione per il multipolo 4
for i in range(Nbins):
    for j in range(Nbins):
        R3[i,j]=Cth3[i,j]**2./math.sqrt(Cth3[i,i]*Cth3[j,j]**2.)

Res3=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipolo 4
for i in range(Nbins):
    for j in range(Nbins):
            Res3[i,j]=(Cth3[i,j]-Cmes3[i,j])*np.sqrt((Nmisure - 1.)/((1.+R3[i,j])*Cth3[i,i]*Cth3[j,j]))

R12=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 0 e 2
for i in range(Nbins):
    for j in range(Nbins):
        R12[i,j]=Cth12[i,j]**2./math.sqrt(Cth12[i,i]*Cth12[j,j]**2.)

Res12=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 0 e 2
for i in range(Nbins):
    for j in range(Nbins):
        Res12[i,j]=(Cth12[i,j]-Cmes12[i,j])*np.sqrt((Nmisure - 1.)/((1.+R12[i,j])*Cth12[i,i]*Cth12[j,j]))

R13=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 0 e 4
for i in range(Nbins):
    for j in range(Nbins):
        R13[i,j]=Cth13[i,j]**2./math.sqrt(Cth13[i,i]*Cth13[j,j]**2.)

Res13=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 0 e 4
for i in range(Nbins):
    for j in range(Nbins):
        Res13[i,j]=(Cth13[i,j]-Cmes13[i,j])*np.sqrt((Nmisure - 1)/((1+R13[i,j])*Cth13[i,i]*Cth13[j,j]))

R23=np.zeros((Nbins,Nbins), dtype=float)		#Matrice di correlazione per i multipoli 2 e 4
for i in range(Nbins):
    for j in range(Nbins):
        R23[i,j]=Cth23[i,j]**2./np.sqrt(Cth23[i,i]*Cth23[j,j]**2.)

Res23=np.zeros((Nbins,Nbins), dtype=float)		#Residuo multipoli 2 e 4
for i in range(Nbins):
    for j in range(Nbins):
        Res23[i,j]=(Cth23[i,j]-Cmes23[i,j])*np.sqrt((Nmisure - 1.)/((1.+R23[i,j])*Cth23[i,i]*Cth23[j,j]))

Res12t=np.transpose(Res12)
Res13t=np.transpose(Res13)
Res23t=np.transpose(Res23)


matriceRes=np.zeros((3*Nbins,3*Nbins), dtype=float)

for i in range(Nbins):
    for j in range(Nbins):
        for t1 in range(3):
            for t2 in range (3):
                I=i+t1*Nbins
                J=j+t2*Nbins
                if I<Nbins and J<Nbins:
                    matriceRes[I,J]=Res1[i,j]
                if I<2*Nbins and I>Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res2[i,j]
                if I>2*Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res3[i,j]
                if I<Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res12[i,j]
                if I<Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res13[i,j]
                if I<2*Nbins and I>Nbins and J<Nbins:
                    matriceRes[I,J]=Res12t[i,j]
                if I<2*Nbins and I>Nbins and J>2*Nbins:
                    matriceRes[I,J]=Res23[i,j]
                if I>2*Nbins and J<Nbins:
                    matriceRes[I,J]=Res13t[i,j]
                if I>2*Nbins and J<2*Nbins and J>Nbins:
                    matriceRes[I,J]=Res23t[i,j]

print('Plot della matrice dei residui:')
plt.title('Matrice dei residui (terzo set)')
plt.imshow(matriceRes)
cbar = plt.colorbar(orientation="vertical", pad=0.02)
cbar.set_label(r'$ Res$')
plt.show()

print('La deviazione standard dei residui è:', np.std(matriceRes))
