#!/usr/bin/env bash

# ========================= DESCRIPTION =====================================
# Script to generate separate jobs for each parameter configuration on the 
# cluster. Puts every job in its own directory and  modifies the main program
# appropriately in terms of parameters.
# ===========================================================================

E0_array=(6.0)
Ecouple_array=(32.0 64.0 128.0)
psi1_array=(10.0)
psi2_array=(-9.0)
phase_array=(0.0)

n_array1=(10.0)
n_array2=(3.0)

mkdir -p master_output_dir

for n1 in ${n_array1[@]}
do
    for n2 in ${n_array2[@]}
    do
        mkdir n1_${n1}_n2_${n2}_dir/
        cd n1_${n1}_n2_${n2}_dir/

        for phase in ${phase_array[@]}
        do
            mkdir phase_${phase}_dir/
            cd phase_${phase}_dir/

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

                            cp ../../../../*.py ./
                            cp ../../../../*.so ./
                            cp ../../../../production_slurm.sh ./

                            # edit the copied file in place to the correct 
                            # parameters
                            sed -ie "24s/3.0/${E0}/" main.py
                            sed -ie "25s/3.0/${Ecouple}/" main.py
                            sed -ie "26s/3.0/${E0}/" main.py
                            sed -ie "27s/3.0/${psi1}/" main.py
                            sed -ie "28s/3.0/${psi2}/" main.py
                            sed -ie "30s/3.0/${n1}/" main.py
                            sed -ie "31s/3.0/${n2}/" main.py
                            sed -ie "32s/0.0/${phase}/" main.py

                            # remove extraneous file
                            rm *.pye

                            # print lines to make sure everything is kosher
                            echo "E0 = ${E0}, Ecouple = ${Ecouple}, E1 = ${E0}, psi1 = ${psi1}, psi2 = ${psi2}, phase = ${phase}"
                            sed -n 24p main.py
                            sed -n 25p main.py
                            sed -n 26p main.py
                            sed -n 27p main.py
                            sed -n 28p main.py
                            sed -n 30p main.py
                            sed -n 31p main.py
                            sed -n 32p main.py
                        
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
        cd ..
    done
done
