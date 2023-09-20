#!/bin/bash

origine="/home/mattia/Downloads/data"				#specifico la directory in cui si trovano i file da copiare
destinazione="/home/mattia/Desktop/Esercizio finale/data"	#specifico la directory in cui voglio copiare i file

if [ ! -d "$origine" ]; then				#controllo che la directory di origine esista
  echo "La directory di origine specificata non esiste."
  exit 1
fi

mkdir -p "$destinazione" || exit 1			#creo la directory di destinazione se questa non esiste

cp -r "$origine"/* "$destinazione" && echo "Copia avvenuta con successo"	#copio i file da origine a destinazione

