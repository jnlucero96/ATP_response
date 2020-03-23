#!/usr/bin/env bash

ID_array=(0 1 2 3 4 5 6)

Ecouple_array=(0.0 32.0 32.0 32.0 32.0 64.0 64.0)

psi1_array=(4.0 1.0 2.0 4.0 4.0 2.0 4.0)

psi2_array=(0.0 -1.0 -2.0 -4.0 -4.0 -2.0 -4.0)

n1_array=(1.0 1.0 2.0 1.0 2.0 2.0 2.0)

for (( i=0; i<${#Ecouple_array[@]}; i++))
do
    echo "Resubmitting Job: ${ID_array[i]}"
    # cd phase_${phase_array[i]}_dir/
    cd phase_0._dir/
    cd E0_2.0_Ecouple_${Ecouple_array[i]}_E1_2.0_dir/
    cd psi1_${psi1_array[i]}_psi2_${psi2_array[i]}_dir/
    cd n1_${n1_array[i]}_n2_3.0_dir/
    sbatch --job-name=nr${ID_array[i]} --time=05-00:00:00 production_slurm.sh
    sleep $[ ( $RANDOM % 10 ) + 1 ]s
    cd ../../../../
done

#cd ..
