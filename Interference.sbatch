#!/bin/bash
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=python-hello-world
#SBATCH --output=python-hello-world.out

module purge
module load python/intel/3.11
python3 -m venv ./test_venv
source ./test_venv/bin/activate
pip install -r requirements.txt

python interference.py
