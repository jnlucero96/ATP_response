#!/usr/bin/env bash

# ============================= DESCRIPTION ==================================
# Submit each job one-by-one and label it appropriately
# ============================================================================

E0_array=(0.0 2.0 4.0)
Ecouple_array=(0.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
psi1_array=(0.0 1.0 2.0 4.0 8.0)
psi2_array=(-8.0 -4.0 -2.0 -1.0 0.0)
#n1_array=(3.0)
#n2_array=(3.0)
phase_array=(0. 0.314159 0.628319 0.942478 1.25664 1.5708 1.88496 2.19911 2.51327 2.82743 3.14159)

ID_NUM=0

touch reporter_file.dat

for phase in ${phase_array[@]}
do
    cd phase_${phase}_dir/

    for E0 in ${E0_array[@]}
    do
        for Ecouple in ${Ecouple_array[@]}
        do
            cd E0_${E0}_Ecouple_${Ecouple}_E1_${E0}_dir/

            for psi1 in ${psi1_array[@]}
            do
                for psi2 in ${psi2_array[@]}
                do
                    cd psi1_${psi1}_psi2_${psi2}_dir/

                    # in case of crashes: know which parameters have crashed
                    echo "${ID_NUM} corresponds to: E0 = ${E0}, Ecouple = ${Ecouple}, E1 = ${E0}, psi1 = ${psi1}, psi2 = ${psi2}, phase = ${phase}" >> ../../../reporter_file.dat
                    # record the ID number in the directory (but hide it)
                    echo $ID_NUM > .varstore

                    # submit the job
                    sbatch --job-name=ID${ID_NUM} --time=03-00:00:00 production_slurm.sh

                    # random sleep time (btwn 1 and 10s) to not overwhelm the SLURM system
                    sleep $[ ( $RANDOM % 10 )  + 1 ]s

                    cd ..
                    ((ID_NUM+=1))
                done
            done
            cd ..
        done
    done
    cd ..
done
