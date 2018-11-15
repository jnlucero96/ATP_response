for c in 0 10 20 30 40 50 60 70 80 90 100 110 120
do
	cd c$c
	qsub runscript_opt.pbs
	qsub runscript_naive.pbs
	# python frictionProper.py
	cd ..
done

