from math import *
from parameters import *

def TrapMin(time):
	mini=2*pi*time/Period
	while mini > 2*pi:
		mini-=2*pi
	return mini

def force(time,position):
	x0=TrapMin(time)
	f=-1.5*A*cos(3*(position+pi/2))-0.5*k*sin(position-x0)
	# f=-k*(position-pi)
	return f;

def potential(time, position):
	x0=TrapMin(time)
	E=A*(1+sin(3*(position+pi/2)))/2+k*(1+sin(position-(pi/2)-x0))/2
	# E=0.5*k*(position-pi)**2
	return E;

def force_x(c, position):# input contol parameter position in radians instead of time 
	f=-1.5*A*cos(3*(position+pi/2))-0.5*k*sin(position-c)
	# f=-k*(position-pi)
	return f;

def force_x_trap(c, position):# input contol parameter position in radians instead of time 
	f=-0.5*k*sin(position-c) #just the trap
	# f=-k*(position-pi)
	return f

def potential_x(c, position):
	E=A*(1+sin(3*(position+pi/2)))/2+k*(1+sin(position-(pi/2)-c))/2
	# E=0.5*k*(position-pi)**2
	return E;

def potentialDer2(time, position):   #SECOND derivative of the potential
	x0=TrapMin(time)
	E=0.5*k*cos(x0-position)-4.5*A*sin(3*(pi/2 + position))
	return E;
	
def Deg2Rad(inDeg):
	inRads=inDeg*2*pi/360
	return inRads;

##Used for visualiztions, use shift to have state at Pi/3
def potentialShifted(time, position):
	x0=TrapMin(time)
	E=A*(1+sin(3*(position-pi/2)))/2+k*(1+sin(position-(pi/2)-pi/3-x0))/2
	return E;

def trap(time,position):
	x0=TrapMin(time)
	E=k*(1+sin(position-(pi/2)-pi/3-x0))/2
	return E;

def unperturbed(time,position):
	x0=TrapMin(time)
	E=A*(1+sin(3*(position-pi/2)))/2
	return E;