from math import exp
cycles=100 #Max number of cycles
gamma=1000.
beta=1.
m=1.
dt=0.1   #reset in code to obey stability
a=exp(-gamma*dt)
N=100
write_out=1./dt  # to have write out every second use 1/dt
# write_out=1 # to write out every calculation
A=6
k=2
Period=1000
