for A in 1 2 4 8 16
do
	# mkdir A$A
	for k in 1 2 4 8 16 32
	do
		# mkdir A$A/k$k
		cp LetsGo_FPE.sh ./A$A/k$k
		for P in 250 500 1000 2000 4000 8000 16000
		do
			mkdir A$A/k$k/P$P
			cp *.py ./A$A/k$k/P$P
			cp runscript_opt.pbs ./A$A/k$k/P$P
			cp runscript_naive.pbs ./A$A/k$k/P$P
			echo "A=$A" >> A$A/k$k/P$P/parameters.py
			echo "k=$k" >> A$A/k$k/P$P/parameters.py
			echo "Period=$P" >> A$A/k$k/P$P/parameters.py
		done
	done
done
