mkdir out
for A in 1 2 4 8 16
do
    mkdir out/A$A
	for k in 1 2 4 8 16 32
	do
		mkdir out/A$A/k$k
		for c in 0 10 20 30 40 50 60 70 80 90 100 110 120
		do
			mkdir out/A$A/k$k/c$c
			cp A$A/k$k/c$c/*.dat ./out/A$A/k$k/c$c
			cp A$A/k$k/c$c/T* ./out/A$A/k$k/c$c
		done
	done
done
