#!/bin/sh
#SBATCH -J StyconMultiP           # Job name
#SBATCH -o slurmout/%j.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gpu           		 # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 12:00:00              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A CS395T         # Specify allocation to charge against

echo $@
date
ipython ostagram_stycon_train.py -- --train-path ostagram_crawler/downloaded/ --model-prefix OstagramStyconSmallLossProduct  --styconNet-type product
date

