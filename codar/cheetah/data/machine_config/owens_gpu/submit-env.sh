#!/bin/bash

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
###### USE FOR PYTORCH DIST ######
# export MASTER_PORT=12340
# export WORLD_SIZE=2

# ### get the first node name as master address - customized for vgg slurm
# ### e.g. master(gnodee[2-5],gnoded1) == gnodee2
# # echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
#################################

module load cuda/11.6.1

source $CONDA_HOME/bin/activate

source activate harp_env


