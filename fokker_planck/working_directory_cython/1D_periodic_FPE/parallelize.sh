#!/usr/bin/env bash

# ========================= DESCRIPTION =====================================
# Script to generate separate jobs for each parameter configuration on the 
# cluster. Puts every job in its own directory and modifies the main program
# appropriately in terms of parameters.
# ===========================================================================

E_array=(0.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
psi1_array=(0.0 1.0 2.0 4.0 8.0)
psi2_array=(-8.0 -4.0 -2.0 -1.0 0.0)
n_array=(3.0)

mkdir -p master_output_dir

for n in ${n_array[@]}
do
    mkdir n_${n}_dir/
    cd n_${n}_dir/

    for E in ${E_array[@]}
    do
        mkdir E_${E}_dir/
        cd E_${E}_dir/

        for psi1 in ${psi1_array[@]}
        do
            for psi2 in ${psi2_array[@]}
            do
                mkdir psi1_${psi1}_psi2_${psi2}_dir/
                cd psi1_${psi1}_psi2_${psi2}_dir/

                cp ../../../main_1d.py ./
                cp ../../../*.so ./
                cp ../../../production_slurm.sh ./

                # edit the copied file in place to the correct parameters
                sed -ie "19s/3.0/${E}/" main_1d.py
                sed -ie "21s/3.0/${psi1}/" main_1d.py
                sed -ie "22s/3.0/${psi2}/" main_1d.py
                sed -ie "24s/3.0/${n}/" main_1d.py

                # remove extraneous files
                rm *.pye

                # print lines to make sure everythin is kosher
                echo "E = ${E}, psi1 = ${psi1}, psi2 = ${psi2}, n = ${n}"
                sed -n 19p main_1d.py
                sed -n 21p main_1d.py
                sed -n 22p main_1d.py
                sed -n 24p main_1d.py
                
                echo
                cd ..
            done
        done
        cd ..
        sleep 2
    done
    cd ..
done
