#!/usr/bin/env bash

# ============================= DESCRIPTION ==================================
# Submit each job one-by-one and label it appropriately
# ============================================================================

E0_array=(2.0)
Ecouple_array=(2.0 4.0 8.0 16.0 32.0 64.0 128.0)
psi1_array=(4.0 4.0 2.0)
psi2_array=(-1.0 -2.0 -1.0)
psi_index=(0 1 2)
n1_array=(12.0)
#n2_array=(1.0)
#phase_array=(0. 0.349066 0.698132 1.0472 1.39626 1.74533 2.0944 2.44346 2.79253 3.14159 3.49066 3.83972 4.18879 4.53786 4.88692 5.23599 5.58505 5.93412 6.28319)
phase_array=(0. 0.08727 0.17453 0.26180 0.34633 0.52360)


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

            for i in ${psi_index[@]}
            do
                psi1=${psi1_array[$i]}
                psi2=${psi2_array[$i]}
                #for psi2 in ${psi2_array[@]}
                #do
                cd psi1_${psi1}_psi2_${psi2}_dir/
                
                for n1 in ${n1_array[@]}
                do
                    #for n2 in ${n1_array[@]}
                    #do
                    cd n1_${n1}_n2_${n1}_dir/

                    # in case of crashes: know which parameters have crashed
                    echo "${ID_NUM} corresponds to: E0 = ${E0}, Ecouple = ${Ecouple}, E1 = ${E0}, psi1 = ${psi1}, psi2 = ${psi2}, phase = ${phase}, n1 = ${n1}, n2 = ${n1}" >> ../../../../reporter_file.dat
                    # record the ID number in the directory (but hide it)
                    echo $ID_NUM > .varstore

                    # submit the job
                    sbatch --job-name=ID${ID_NUM} --time=01-00:00:00 production_slurm.sh

                    # random sleep time (btwn 1 and 10s) to not overwhelm the SLURM system
                    sleep $[ ( $RANDOM % 10 )  + 1 ]s

                    cd ..
                    ((ID_NUM+=1))
                done
                cd ..
            done
            cd ..
        done
    done
    cd ..
done
