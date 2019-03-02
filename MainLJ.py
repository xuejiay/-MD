#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:18:38 2018

@author: xuejiaoyang
"""
import numpy as np
import MultDef as MD



############# Choose the system we want to simulate ###########################

# Choose the Method you would like to use 'MC' for Monte Carlo or 'MD' for 
# Molecular Dynamics
Meth = 'MD'



# Choose the your system to work for NVE or NVT
# If thermo=0, the system works as NVE and if thermo=1, we use Berendsen 
# thermostat to modify the system to make it working on constant temperature
thermo=1 # 0 NVE 1 NVT

############# For MD choose integrators, time steps and step size #############

# Choose integrators: 1: Semi-implicit Euler method 2: Velocity Verlet
integrator=2

# Total Time steps or MC steps
Time=3000

# Time step size (dimensionless units)
dt = 0.001 
###############################################################################


# Parameters setting###########################################################

# Start from choosing your box size. We choose a length first.
# Then choose L, the lattice size (the number of cells is L*L*L)
# and length of each cell b is computed by b = length/L

# Choose the length of the box and the volume will be length**3

length = 8

# Choose the size of L and it will determine the total molecule you would like 
# to simulate. 

L = 4 

# We put 4 molecule in each cell. Thus the total number of the molecule is
N = 4 *L**3 # Number of particles



# Size of unit cell (units of sigma)

b = length/L 


# Note that the length of the total system is L*b and the volume is (L*b)**3


# Intial maximum velocity. The initial v is normally distributed in [-0.5,0.5]

v0=0.5


###############################################################################


################ Then we can start generate initial configurations ###########

# Comment the following function if you have the csv files OR you can directly
# use random generated velocity and output from the following code.

[r,v]=MD.initiali(L,b,N,v0) 

#r = np.genfromtxt('LJr.csv', delimiter=',')
#v = np.genfromtxt('LJv.csv', delimiter=',')
###############################################################################





kb=1
# Average temperature compute from NVE is
Tc=0.59616378089612165
#Tc=0.7
###############################################################################

param = [L,b,N,Time,dt,thermo,Tc,kb]
import time


if Meth == 'MD':
    
    # Start MD Method----------------------------------------------------------
  
    
    
   
    
    # Start computing
    if integrator==1:
        start_time1 = time.time()
        [Ekt, Ept,Tt] = MD.Integ1(param,r,v)
        end_time1 = time.time()- start_time1
        
    if integrator==2:
        
        start_time2 = time.time()
        [Ekt, Ept,Tt] = MD.Integ2(param,r,v)
        end_time2 = time.time()-start_time2
        
    Et = np.array(Ekt)+ np.array(Ept)
    
    # Plot energy and temperature  
    # Comment them if you do not want to plot
#    import matplotlib.pyplot as plt    
#    plt.figure(0)
#    plt.plot(range(Time+1), Et)
#    plt.xlabel('time steps')
#    plt.ylabel('Energy')
#    plt.figure(1)
#    plt.plot(range(Time+1), Tt)
#    plt.xlabel('time steps')
#    plt.ylabel('Temperature')
    # Save energy and temperature
    saveEkt = np.column_stack((np.array(range(Time+1)), Ekt))
    saveEpt = np.column_stack((np.array(range(Time+1)), Ept))
    saveEt = np.column_stack((np.array(range(Time+1)), Et))
    saveTt = np.column_stack((np.array(range(Time+1)), Tt))
    np.savetxt('Total_energyMD.csv', saveEt, delimiter=',')
    np.savetxt('temperatureMD.csv', saveTt, delimiter=',')
    np.savetxt('Kinetic_energyMD.csv', saveEkt, delimiter=',')
    np.savetxt('Potential_energyMD.csv', saveEpt, delimiter=',')    
    
    # End MD-------------------------------------------------------------------
 

if Meth == 'MC':
    pos = r
    
    # Start MC method----------------------------------------------------------
    
    
    
    import random
    import math
    
    # Choose MC steps
    Ttotal = Time
    # Compute beta
    beta = 1/kb/Tc
    
    # maxd = 0.3 has 30% of acceptance and 0.2 has more than 50%
    maxd = 0.3
    Ep = MD.PotEnerg(pos,param)
    E=[Ep]
    test1=0
    test2=0
    for tc in range(Ttotal):
        # Choose random steps to move
        RN = random.randint(0,N-1)
        rx = random.uniform(-maxd,maxd)
        ry = random.uniform(-maxd,maxd)
        rz = random.uniform(-maxd,maxd)
        pos[RN] = pos[RN]+np.array([rx,ry,rz])
        for k in range(3):
            if pos[RN,k]<0:
                pos[RN,k]+L*b
            if pos[RN,k]>L*b:
                pos[RN,k]-L*b
        # Compute energy
        Epf = MD.PotEnerg(pos,param)
        DE = Epf-Ep
        # Check if the energy is acceptable
        if DE<0: #Accept           
            Ep = Epf
            E.append(Ep)  
            test1=test1+1 
        else:
            
            Check = math.exp(-beta*DE)
            if random.uniform(0,1)<Check: #Accept
                Ep = Epf
                E.append(Ep)  
                test1=test1+1
            else: #Reject
                pos[RN] = pos[RN]-np.array([rx,ry,rz])
                E.append(Ep)
                test2=test2+1
                
                
    LL = len(E)            
    # Plot energy and temperature
    
    # Comment them if you do not want to plot      
      
#    import matplotlib.pyplot as plt
#    plt.figure(0)
#    plt.plot(range(LL), E)
#    plt.xlabel('MC steps')
#    plt.ylabel('Energy')     
    
    # Save energy 
    saveE = np.column_stack((np.array(range(LL)), E))
    np.savetxt('energyMC.csv', saveE, delimiter=',')

    
    
    # End MC------------------------------------------------------------------- 













 
        
    

