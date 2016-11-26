#!/bin/sh
#SBATCH -J ResizeconvNoTV           # Job name
#SBATCH -o slurmout/%j.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gpu           		 # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 12:00:00              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A CS395T         # Specify allocation to charge against

echo $@
date
ipython train.py --  --train-path 'contents/coco/train2014/' --style $@ --model-prefix ResizeconvNoTVStyleH768 --resize-style-height 768 --checkpoint-dir chkpts/ResizeconvNoTVResizeStyleimg
date

