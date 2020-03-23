#!/usr/bin/env bash
#SBATCH --mail-user=emma.lathouwers@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=1G
#SBATCH --no-requeue
#SBATCH --account=rrg-dsivak

cd $SLURM_SUBMIT_DIR
echo "Current working directory is `pwd`"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Starting run at: `date`"
echo "Node running job: $SLURMD_NODENAME"
module load python/3.7.0 scipy-stack/2019b
python main.py > calculation.out 2> calculation.err < /dev/null
echo "Job finished with exit code $? at: `date`"
