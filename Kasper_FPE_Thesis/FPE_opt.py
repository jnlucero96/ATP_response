from math import *   # import everything from python math library
from protocol import *
from potential import TrapMin,force_x, potential_x #import potential and force functions from potential.py
from parameters import *  #import everything in parameters.py
import random #import the python random library 
import numpy as np #import python numpy library as np
import os.path

info=open('info_opt.dat','w')
plots=False
if plots==True:
	import matplotlib.pyplot as plt
	os.makedirs('Probs_opt')

# print 'N: '+str(N)+', A: '+str(A)+', k: '+str(k)+', Gamma: '+str(gamma)+ ', Period: '+str(Period)   #output to terminal

dx=2*pi/N # size of system bins, N is number of bins defined in parameters
time_check=dx*m*gamma/(1.5*A+0.5*k)
if dt>time_check:
	info.write('TIME UNSTABLE'+'\n')

##________Initalize Distribution_________##
position=list(np.linspace(0, 2*pi-2*pi/N,N))
PlottingPosition=position+[2*pi]
Z=0
for x in position: #calculates the partition function
	Z+=exp(-beta*potential_x(0,x))*dx
E_QS_now=-np.log(Z)/beta
E=0
Prob=[]
force_average=0
prob_dist=[]
for x in position: #calculates the equilirbiurm distribution at t=0
	Prob+=[exp(-beta*potential_x(0,x))*dx/Z]
	# prob_dist.append(exp(-beta*potential(0,x)/Z)
	E+=potential_x(0,x)*exp(-beta*potential_x(0,x))*dx/Z
	# force_average+=force(0,x)*exp(-beta*potential(0,x))*dx/Z
# np.savetxt('dist.gz',(position,Prob))
Prob_now=Prob
del Prob
E_afterRelax=E
E_0=E
del E

##____________Definitions_________##

def calc_flux(location):
	inst_flux=(force_x(c,location*dx)*Prob_last[location]/(m*gamma)-(Prob_last[location+1]-Prob_last[location-1])/(gamma*beta*2*dx))*dt/dx
	return inst_flux

def SaveData(*args): # example: SaveData(x, v, work) prints x, v, work to workfile3.dat
	for data in args:
		h.write(str(data)+'\t')
	h.write('\n')

def Rotation(P_dist,deg): # P_now is prob now (c=0) and P_240 is prob Period/3 ago (c=240)
	P_rotate=[0]*N
	# P_240=P_240*2
	index_rot=int(deg*N/(360))
	for i in range(0,len(P_dist)):
		P_rotate[i]=P_dist[i-index_rot]
	return P_rotate

##____________Make protocol_scaled_________##
cdegValues=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
Friction=BuildFriction_full(cdegValues)
angles=list(np.linspace(0,360,37))
raw_time=Protocol_times(Friction,angles)
norm_time=raw_time/raw_time[-1]
real_time=norm_time*Period
alpha=2*10/((real_time[1]-real_time[0])*(Friction[0]**-0.5+Friction[1]**-0.5))
velocities=alpha*(Friction**-0.5)

##____________Initialize Simulation_________##	
flux=[0]*N   # list to keep track of flux at every position
work=0 #Cumulative work
work_QS=0
heat=0 #Cumulative heat
time=0
step_counter=0
print_counter=0
print_frequency=50*write_out
SSS_counter=0
Prob_last_cycle=Prob_now  #for SSS check, need to save initial distribution 
SSS_cycle_counter=0 # counts the number of SSS cycles 
f=open('PROB_opt.dat', 'w')
g=open('WORKandHEAT_opt.dat', 'w')
h=open('FLUX_opt.dat', 'w')
###______Print out starting info____###
# for prob in Prob_now:
# 	f.write(str(prob)+'\t')
# f.write('\n')

info.write("E0: "+str(E_afterRelax)+'\n')

time+=dt

while (time<cycles*Period+dt and SSS_cycle_counter<3):
	#determine trap position
	trap_deg=Protocol_detailed(real_time,angles,velocities,time)
	trap_rad=trap_deg*2*pi/360.
	c=trap_rad
	#determine quasistatic work
	E_QS_last=E_QS_now
	Z_now=0
	for x in position: #calculates the partition function
		Z_now+=exp(-beta*potential_x(c,x))*dx
	E_QS_now=-np.log(Z_now)/beta
	work_QS+=(E_QS_now-E_QS_last)


	Prob_last=Prob_now
	Prob_now=[0]*N
	flux_now=np.zeros(N)
	E_last=E_afterRelax  #Energy at t-dt is E_last
	E_changePotential=0
	for i in range(0,N):
		E_changePotential+=potential_x(c,i*dx)*Prob_last[i]
	work+=(E_changePotential-E_last)
	work_inst=E_changePotential-E_last
	Prob_now[0]=(Prob_last[0]
				+dt*(-force_x(c,dx)*Prob_last[1]+force_x(c,-dx)*Prob_last[-1])/(2*dx*gamma*m)
				+dt*(Prob_last[1]+Prob_last[-1]-2*Prob_last[0])/(beta*gamma*dx**2))
	flux[0]+=calc_flux(0)
	flux_now[0]=calc_flux(0)
	for i in range(1,len(Prob_now)-1):
		Prob_now[i]=(Prob_last[i]
					+dt*(-force_x(c,i*dx+dx)*Prob_last[i+1]+force_x(c,i*dx-dx)*Prob_last[i-1])/(2*dx*gamma*m)
					+dt*(Prob_last[i+1]+Prob_last[i-1]-2*Prob_last[i])/(beta*gamma*dx**2))
		flux[i]+=calc_flux(i)
		flux_now[i]=calc_flux(i)
	Prob_now[-1]=(Prob_last[-1]
				+dt*(-force_x(c,0)*Prob_last[0]+force_x(c,(len(Prob_last)-1)*dx-dx)*Prob_last[-2])/(2*dx*gamma*m)
				+dt*(Prob_last[0]+Prob_last[-2]-2*Prob_last[-1])/(beta*gamma*dx**2))
	flux[-1]+=(force_x(c,(len(Prob_last)-1)*dx)*Prob_now[-1]/(m*gamma)-(Prob_now[0]-Prob_now[-2])/(gamma*beta*2*dx))*dt/dx
	flux_now[-1]=(force_x(c,(len(Prob_last)-1)*dx)*Prob_now[-1]/(m*gamma)-(Prob_now[0]-Prob_now[-2])/(gamma*beta*2*dx))*dt/dx
	E_afterRelax=0
	for i in range(0,len(Prob_now)):
		E_afterRelax+=potential_x(c,i*dx)*Prob_now[i]
	heat+=(E_afterRelax-E_changePotential)
	heat_inst=E_afterRelax-E_changePotential

	step_counter+=1
	print_counter+=1
	SSS_counter+=1
	# print "______"
	# print SSS_cycle_counter
	if trap_deg>=239.9 and trap_deg<240:
		print "detected :" +str(time) +str(trap_deg)
		Prob_240=Prob_now
		prob2plot_240=Prob_240+[Prob_240[0]]
	if SSS_counter==(Period)/dt:
		Relative_Entropy_SSS=0
		Sq_diff=0
		for i in range(0,len(Prob_now)):
			Relative_Entropy_SSS+=Prob_now[i]*log(Prob_now[i]/Prob_last_cycle[i])
			Sq_diff+=(Prob_now[i]-Prob_last_cycle[i])**2/N
		print Sq_diff
		Prob_last_cycle=Prob_now
		# print Relative_Entropy_SSS
		# if Relative_Entropy_SSS<1e-5:
		# if Sq_diff<0.00000001:
		if Sq_diff<0.00000001:
			print "YES"
			info.write(str(time)+'\t' + str(Relative_Entropy_SSS)+'\t'+str(Sq_diff)+'\n')
			SSS_cycle_counter+=1
			if SSS_cycle_counter==2:
				work=0
				work_QS=0
				heat=0
				flux=[0]*N 
			print SSS_cycle_counter
			# print time,SSS_cycle_counter
		SSS_counter=0
		## Rotation check
		Sq_diff=0
		Prob_240_rot=Rotation(Prob_240,120)
		for i in range(0,len(Prob_now)):
			Sq_diff+=(Prob_now[i]-Prob_240_rot[i])**2/N
		print Sq_diff
		if Sq_diff>0.0001:
			SSS_cycle_counter=8
			print "Failed 240 test"
		SSS_counter=0


	if step_counter==write_out: #controls write out frequency to file and terminal
		if SSS_cycle_counter==2:
		# for prob in Prob_now:
		# 	f.write(str(prob)+'\t')
		# f.write('\n')
		# g.write(str(work)+'\t'+str(heat)+'\t'+str(work_inst)+'\t'+str(heat_inst)+'\t'+str(E_afterRelax) +'\t'+str(E_afterRelax-E_0) +'\t'+str(heat+work)+'\n')
			g.write(str(work)+'\t'+str(heat)+'\t'+str(E_afterRelax)+'\t'+str(work_QS)+'\n')
		# for flux_values in flux:
		# 	h.write(str(flux_values)+'\t')
		# h.write('\n')
			h.write(str(np.mean(flux))+'\n')
		step_counter=0
	if plots==True:
		if print_counter==print_frequency:
			if SSS_cycle_counter==2:
				prob2plot=Prob_now+[Prob_now[0]]
				ax = plt.subplot(111, projection='polar')
				ax.plot([c]*10,np.linspace(0,0.2,10),color='k',linewidth=2)
				ax.plot(PlottingPosition, prob2plot, color='r',linewidth=3.0)
				ax.plot(PlottingPosition, prob2plot_240, color='b',linewidth=3.0)
				ax.set_rmax(0.13)
				ax.grid(True)
				ax.set_title("t="+str(int(time)), va='bottom')
				# plt.xticks([])
				ax.plot([0]*10,np.linspace(0,0.2,10),':',color='#808080')
				ax.plot([2*pi/3]*10,np.linspace(0,0.2,10),':',color='#808080')
				ax.plot([4*pi/3]*10,np.linspace(0,0.2,10),':',color='#808080')
				plt.yticks([])
				plt.xticks([0,2*pi/3,4*pi/3])
				print time
				plt.savefig('Probs_opt/t='+str(time)+'.png')
				plt.close()
			print_counter=0
			# s='time: '+repr(time)+", Prob: "+repr(sum(Prob_now))
			# print s
			# print np.mean(flux)
	time+=dt
f.close()
g.close()
h.close()
info.write('P_final=' + str(sum(Prob_now))+'\n')
info.write('dt='+str(dt)+'\n')
info.close()
# print 'Total Probability: ' + str(sum(Prob_now))
# print 'final energy: ' +str(E_afterRelax)
# print position

