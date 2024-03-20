#!/bin/bash

mkdir -p alov300pp
cd alov300pp

wget https://isis-data.science.uva.nl/alov/alov300++_frames.zip
wget https://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip

unzip alov300++_frames.zip
unzip alov300++GT_txtFiles.zip

rm alov300++_frames.zip
rm alov300++GT_txtFiles.zip

cd ..

chmod -R 777 alov300pp