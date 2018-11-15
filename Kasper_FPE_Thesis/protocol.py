from math import *   # import everything from python math library
import numpy as np #import python numpy library as np
from parameters import *

def BuildFriction_full(cdegValues):
	Fric=list(np.zeros(len(cdegValues)-1))
	for i in range(0,len(cdegValues)-1):
		c=cdegValues[i]
		# Fvalue=np.loadtxt('out/A'+str(A)+'/k'+str(k)+'/c'+str(c)+'/FINAL.dat',unpack=True)
		Fvalue=np.loadtxt('/Users/Kasper/OwnCloud/Documents/F1ATPSynthase/May/out/A'+str(A)+'/k'+str(k)+'/c'+str(c)+'/FINAL.dat',unpack=True)
		Fric[i]=Fvalue
	Fric_full=Fric*3+[Fric[0]]
	Fric_full=np.asarray(Fric_full)
	return Fric_full;

def Protocol_times(Friction,cdegValues):
	velocities=Friction**-0.5
	P_times=np.zeros(len(Friction))
	for i in range(1,len(Friction)):
		P_times[i]=P_times[i-1]+2*(cdegValues[i]-cdegValues[i-1])/(velocities[i-1]+velocities[i])
	return P_times

def Protocol_detailed(real_time,angles,velocities,time):
	while time >= Period:
		time-=Period
	for i in range(0,len(real_time)-1):
		if time<real_time[i+1]:
			if time>=real_time[i]:
				tau=time-real_time[i]
				pos=angles[i]+velocities[i]*tau+0.5*(tau**2)*(velocities[i+1]-velocities[i])/(real_time[i+1]-real_time[i])
	return pos


