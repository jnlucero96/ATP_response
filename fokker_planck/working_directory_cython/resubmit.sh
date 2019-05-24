#!/usr/bin/env bash

ID_array=(1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24)
Ecouple_array=(0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	2.0	2.0	2.0	2.0	2.0	2.0)
psi1_array=(8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0	8.0)
psi2_array=(0.0	0.0	0.0	0.0	0.0	0.0	-1.0	-1.0	-1.0	-1.0	-1.0	-1.0	-2.0	-2.0	-2.0	-2.0	-2.0	-2.0	0.0	0.0	0.0	0.0	0.0	0.0)
phase_array=(0.	0.628319	1.25664	1.88496	2.51327	3.14159	0.	0.628319	1.25664	1.88496	2.51327	3.14159	0.	0.628319	1.25664	1.88496	2.51327	3.14159	0.	0.628319	1.25664	1.88496	2.51327	3.14159)

#cd n1_3.0_n2_3.0_dir/

for (( i=0; i<${#Ecouple_array[@]}; i++))
do
    echo "Resubmitting Job: ${ID_array[i]}"
    cd phase_${phase_array[i]}_dir/
    cd E0_2.0_Ecouple_${Ecouple_array[i]}_E1_2.0_dir/
    cd psi1_${psi1_array[i]}_psi2_${psi2_array[i]}_dir/
    sbatch --job-name=nr${ID_array[i]} --time=06-00:00:00 production_slurm.sh
    sleep $[ ( $RANDOM % 10 ) + 1 ]s
    cd ../../../
done

#cd ..
