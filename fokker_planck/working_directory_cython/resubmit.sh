#!/usr/bin/env bash

#ID_array=(340	348	356	364	372	380	388	413	421	428	429	436	437	438	444	445	446	452	453	454	461	462	469	477	494	502	509	510	516	517	518	519	524	525	526	527	528	533	534	535	536	541	542	543	544	550	551	558	575	582	583	590	591	592	606	607	608	614	615	616	617	622	623	624	625	631	632	639)
#E_couple_array=(16.0	16.0	16.0	16.0	16.0	16.0	16.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	32.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	64.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0	128.0)
#psi_1_array=(-4.0	-2.0	-1.0	0.0	1.0	2.0	4.0	-8.0	-4.0	-2.0	-2.0	-1.0	-1.0	-1.0	0.0	0.0	0.0	1.0	1.0	1.0	2.0	2.0	4.0	8.0	-8.0	-4.0	-2.0	-2.0	-1.0	-1.0	-1.0	-1.0	0.0	0.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0	4.0	4.0	8.0	-8.0	-4.0	-4.0	-2.0	-2.0	-2.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0	4.0	4.0	8.0)
#psi_2_array=(4.0	2.0	1.0	0.0	-1.0	-2.0	-4.0	8.0	4.0	1.0	2.0	0.0	1.0	2.0	-1.0	0.0	1.0	-2.0	-1.0	0.0	-2.0	-1.0	-4.0	-8.0	8.0	4.0	1.0	2.0	-1.0	0.0	1.0	2.0	-2.0	-1.0	0.0	1.0	2.0	-2.0	-1.0	0.0	1.0	-4.0	-2.0	-1.0	0.0	-4.0	-2.0	-8.0	8.0	2.0	4.0	1.0	2.0	4.0	-1.0	0.0	1.0	-2.0	-1.0	0.0	1.0	-4.0	-2.0	-1.0	0.0	-4.0	-2.0	-8.0)

ID_array=(25	54	66	74	89	98	99	100	225	226	238	239	378	379	380	399	400	427	428	431	432	433	434	435	436	437	439	460	470	550	551	982	983	984	985	986	987	1003	1004	1005	1006	1025	1026	1039	1041	1042	1062	1080	1094	1095	1097	1098	1187)
Ecouple_array=(2.0	4.0	4.0	4.0	8.0	8.0	8.0	16.0	2.0	2.0	2.0	2.0	128.0	128.0	128.0	128.0	0.0	2.0	2.0	2.0	2.0	2.0	2.0	2.0	2.0	2.0	2.0	4.0	4.0	64.0	64.0	128.0	128.0	128.0	128.0	128.0	128.0	0.0	0.0	0.0	0.0	2.0	2.0	2.0	2.0	2.0	4.0	8.0	8.0	8.0	8.0	8.0	128.0)
psi1_array=(0.0	0.0	4.0	8.0	2.0	8.0	8.0	0.0	0.0	0.0	2.0	2.0	0.0	0.0	1.0	8.0	0.0	0.0	0.0	1.0	1.0	1.0	1.0	2.0	2.0	2.0	2.0	2.0	8.0	0.0	0.0	1.0	1.0	1.0	2.0	2.0	2.0	0.0	0.0	1.0	1.0	0.0	0.0	2.0	4.0	4.0	2.0	1.0	4.0	8.0	8.0	8.0	2.0)
psi2_array=(-8.0	0.0	-4.0	0.0	0.0	-1.0	0.0	-8.0	-8.0	-8.0	-1.0	0.0	-1.0	0.0	-8.0	0.0	-8.0	-2.0	-1.0	-4.0	-2.0    -1.0	0.0	-8.0	-4.0	-2.0	0.0	-8.0	-8.0	-8.0	-4.0	-2.0	-1.0	0.0	-8.0	-4.0	-2.0	-1.0	0.0	-8.0	-4.0	-8.0	-4.0	0.0	-4.0	-2.0	-2.0	-8.0	0.0	-8.0	-2.0	-1.0	-2.0)
phase_array=(0.	0.	0.	0.	0.	0.	0.	0.	0.628319	0.628319	0.628319	0.628319	0.628319	0.628319	0.628319	0.628319	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	1.25664	2.51327	2.51327	2.51327	2.51327	2.51327	2.51327	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159	3.14159)

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
