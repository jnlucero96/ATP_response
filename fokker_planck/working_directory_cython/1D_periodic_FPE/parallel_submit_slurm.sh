#!/usr/bin/env bash

# ============================= DESCRIPTION ==================================
# Submit each job one-by-one and label it appropriately
# ============================================================================

E_array=(0.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
psi1_array=(0.0 1.0 2.0 4.0 8.0)
psi2_array=(-8.0 -4.0 -2.0 -1.0 0.0)
n_array=(3.0)

ID_NUM=0

touch reporter_file.dat

for n in ${n_array[@]}
do
    cd n_${n}_dir/

    for E in ${E_array[@]}
    do
        cd E_${E}_dir/

        for psi1 in ${psi1_array[@]}
        do
            for psi2 in ${psi2_array[@]}
            do
                cd psi1_${psi1}_psi2_${psi2}_dir/

                # in case of crashes: know which parameters have crashed
                echo "${ID_NUM} corresponds to: E = ${E}, psi1 = ${psi1}, psi2 = ${psi2}, n = ${n}" >> ../../../reporter_file.dat
                # record the ID number in the directory (but hide it)
                echo $ID_NUM > .varstore

                # submit the job
                sbatch --job-name=1D${ID_NUM} --time=01-00:00:00 production_slurm.sh
                # random sleep time (btwn 1 and 10 s) to not overwhelm the SLURM system
                sleep $[ ( $RANDOM % 10 )  + 1 ]s

                cd ..
                ((ID_NUM+=1))
            done
        done
        cd ..
    done
    cd ..
done
