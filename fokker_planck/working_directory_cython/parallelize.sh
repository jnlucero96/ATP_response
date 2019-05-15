#!/usr/bin/env bash

# ========================= DESCRIPTION =====================================
# Script to generate separate jobs for each parameter configuration on the 
# cluster. Puts every job in its own directory and  modifies the main program
# appropriately in terms of parameters.
# ===========================================================================

E0_array=(0.0 2.0 4.0)
Ecouple_array=(0.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
psi1_array=(0.0 1.0 2.0 4.0 8.0)
psi2_array=(-8.0 -4.0 -2.0 -1.0 0.0)
n1_array=(3.0 8.0 10.0 15.0)
n2_array=(3.0)

mkdir -p master_output_dir

for n2 in ${n2_array[@]}
do
    for n1 in ${n1_array[@]}
    do
        mkdir n1_${n1}_n2_${n2}_dir/
        cd n1_${n1}_n2_${n2}_dir/

        for E0 in ${E0_array[@]}
        do
            for Ecouple in ${Ecouple_array[@]}
            do
                mkdir E0_${E0}_Ecouple_${Ecouple}_E1_${E0}_dir/
                cd E0_${E0}_Ecouple_${Ecouple}_E1_${E0}_dir/

                for psi1 in ${psi1_array[@]}
                do
                    for psi2 in ${psi2_array[@]}
                    do
                        mkdir psi1_${psi1}_psi2_${psi2}_dir/
                        cd psi1_${psi1}_psi2_${psi2}_dir/

                        cp ../../../main.py ./
                        cp ../../../*.so ./
                        cp ../../../production_slurm.sh ./

                        # edit the copied file in place to the correct 
                        # parameters
                        sed -ie "21s/3.0/${E0}/" main.py
                        sed -ie "22s/3.0/${Ecouple}/" main.py
                        sed -ie "23s/3.0/${E0}/" main.py
                        sed -ie "24s/3.0/${psi1}/" main.py
                        sed -ie "25s/3.0/${psi2}/" main.py
                        sed -ie "27s/8.0/${n1}/" main.py
                        sed -ie "28s/3.0/${n2}/" main.py

                        # remove extraneous file
                        rm *.pye

                        # print lines to make sure everything is kosher
                        echo "E0 = ${E0}, Ecouple = ${Ecouple}, E1 = ${E0}, psi1 = ${psi1}, psi2 = ${psi2}, n1 = ${n1}, n2 = ${n2}"
                        sed -n 21p main.py
                        sed -n 22p main.py
                        sed -n 23p main.py
                        sed -n 24p main.py
                        sed -n 25p main.py
                        sed -n 27p main.py
                        sed -n 28p main.py
                        
                        echo
                        cd ..
                    done
                done
                cd ..
                sleep 2
            done
        done
        cd ..
    done
done
