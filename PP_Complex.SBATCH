#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=10                # uses 1 compute core per task
#SBATCH --time=80:00:00
#SBATCH --mem=400GB
#SBATCH --job-name=pp-memory
#SBATCH --account=pr_215_general
#SBATCH --output=outs/pp/%A_%a.out

module purge

singularity exec \
	    --overlay /scratch/zy2043/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python main.py --config=arguments$SLURM_ARRAY_TASK_ID.yml"



