#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00                                  #specify time for the job
##SBATCH --nodelist=                                     #specify specific nodes, if you want those specific nodes
##SBATCH --partition=                                    #specify your partition if required

srun bash train.sh
