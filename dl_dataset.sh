#!/bin/bash

# download dataset
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
mkdir dataset dataset/Apple dataset/Grape
mv images/Apple_* dataset/Apple
mv images/Grape_* dataset/Grape
rm -rf images leaves.zip
