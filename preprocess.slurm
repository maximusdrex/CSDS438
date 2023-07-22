#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH -p classc -A sxg125_csds438
#SBATCH -N 1 -n 2
#SBATCH --mail-user=mrs314@case.edu
#SBATCH --mail-type=all

module swap intel gcc
module load python/3.8.6

source start_python.sh

unzip Fuji.zip -d $PFSDIR
cp preprocess.py $PFSDIR
cd $PFSDIR
mkdir processed
python3 preprocess.py
tar -cvzf processed.tar.gz processed/*
cp processed.tar.gz $SLURM_SUBMIT_DIR
rm -rf "$PFSDIR"/*