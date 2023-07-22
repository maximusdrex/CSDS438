The Fuji.zip file contains text files with three lists for training and testing purposes, pointing to pairs of files in the folders long/ and short/

The preprocessing program takes each of these pairs and creates a tensor from each of the RAW files and saves them to the disk. The `preprocess.slurm` batch job then packs them into an archive.

The train.py file will eventually run training on all of the training pairs, but currently just tests the loss function with a simple model.