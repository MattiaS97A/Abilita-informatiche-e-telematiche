# Abilità informatiche e telematiche
Si divide l'esercizio finale di Abilità informatiche e telematiche in 3 file:


* ***copia_file.sh***

Script che copia i file contenuti in una directory di origine all'interno di una directory di destinazione. I percorsi delle due directory sono specificati all'interno dello script.
Lo script si può rendere eseguibile attraverso il comando "chmod +x copia_file.sh", e poi lo si può eseguire usando "./copia_file.sh".
Ad esempio, nello script qui presente i tre set di dati contenuti nella cartella "data" presente nella cartella "Download" vengono copiati nella cartella "data" nella directory in cui si trovano i codici in python che costituiscono l'esercizio.

<br>

* ***singolo_multipolo.py***

Esercizio di prova per il singolo multipolo 0 del primo set di dati.
Questo codice è diviso nelle seguenti sezioni:
- caricamento dei dati del singolo multipolo 0 del primo set di dati
- calcolo della covarianza teorica e plot della matrice delle covarianze teoriche 
- calcolo della covarianza misurata e plot della matrice delle covarianze misurate
- calcolo della matrice di correlazione e dei residui, e plot della matrice dei residui.
- stampa su schermo del valore della deviazione standard dei residui (che è circa 1).

<br>

* ***esercizio_3set.py***

Lo stesso procedimento fatto per il singolo multipolo viene esteso ai tre multipoli 0,2,4 e ai tre set di dati.
L'esercizio è diviso in 3 parti simili, una per ognuno dei 3 set di misure. Ogni parte è divisa nelle seguenti sezioni:
- lettura dei multipoli 0,2,4
- calcolo della covarianza teorica e plot della matrice delle covarianze teoriche per i tre multipoli
- calcolo della covarianza misurata e plot della matrice delle covarianze misurate per i tre multipoli
- calcolo dei residui e plot della matrice dei residui per i tre multipoli
- stampa su schermo del valore della deviazione standard dei residui

Le deviazioni standard dei residui per i tre set di dati sono circa uguali a 1.
