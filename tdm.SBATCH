#!/bin/bash
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=8:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=tdm
#SBATCH --account=pr_215_general
#SBATCH --output=tdm.out

module purge

singularity exec  \
	    --overlay /scratch/zy2043/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh python main.py  --config=arguments-TDM.yml"
