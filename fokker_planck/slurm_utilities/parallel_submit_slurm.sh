#!/usr/bin/env bash

# ============================= DESCRIPTION ==================================
# Submit each job one-by-one and label it appropriately
# ============================================================================

E0_array=(6.0)
Ecouple_array=(32.0 64.0 128.0)
psi1_array=(10.0)
psi2_array=(-9.0)
phase_array=(0.0)

n_array1=(10.0)
n_array2=(3.0)

ID_NUM=0

touch reporter_file.dat

for n1 in ${n_array1[@]}
do
    for n2 in ${n_array2[@]}
    do
        cd n1_${n1}_n2_${n2}_dir/

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
                            echo "${ID_NUM} corresponds to: E0 = ${E0}, Ecouple = ${Ecouple}, E1 = ${E0}, psi1 = ${psi1}, psi2 = ${psi2}, phase = ${phase}, n1 = ${n1}, n2 = ${n2}" >> ../../../../reporter_file.dat
                            # record the ID number in the directory (but hide it)
                            echo $ID_NUM > .varstore

                            # submit the job
                            sbatch --job-name=ID${ID_NUM} --time=7-00:00:00 production_slurm.sh

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
        cd ..
    done
done
