#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:28:27 2018

@author: xuejiaoyang
"""




###############################################################################
# Function Name: initiali
# Description: This code generate initial velocities and positions for 
# the certain number of molecules.
# Inputs: L-- the size of the box in terms of cells,
#         b -- the length of the cells
#         N -- the number of the molecules
# Outputs: r -- positions
#          v -- velocities
###############################################################################   
def initiali(L,b,N,v0):
 
    import numpy as np
    import random 

    #v0=0.5 # v is in [-0.5,0.5]
    num = -1
    rMat = np.matrix([[0.0, 0.0, 0.0],[b/2, b/2, 0.],[b/2, 0., b/2],[0., b/2, b/2]])
    Vx=[]
    Vy=[]
    Vz=[]
    
    r = np.reshape([0.]*N*3,(N,-1))
    for x in range(L):
        for y in range(L):
            for z in range(L):            
                rx = b*x
                ry = b*y
                rz = b*z               
                for i in range(4):
                    ri = np.array([rx,ry,rz])+rMat[i]
                    num = num+1
                    r[num] = ri # Uniformly position
                   
                    
                    
    for j in range(N):
        vx = random.gauss(-v0, v0)
        Vx.append(vx)
        
        vy = random.gauss(-v0, v0)
        Vy.append(vy)
        vz = random.gauss(-v0, v0)
        Vz.append(vz)
        
        
    vxm=np.mean(Vx)
    vym=np.mean(Vy) 
    vzm=np.mean(Vz)

    vm = np.array([vxm,vym,vzm]*N).reshape(N,-1)
    v = np.array([Vx,Vy,Vz]).reshape(3,-1).transpose()
    v = v-vm    
    #np.savetxt('LJr.csv', r, delimiter=',')
    #np.savetxt('LJv.csv', v, delimiter=',')
    
    return r, v
    
#-----------------------------------end---------------------------------------#







###############################################################################
# Function Name: force
# Description: compute the force between two given particles 
# Inputs: ri-- the first particle
#         rj -- the second particle
#         Length -- the length of the box
# Outputs: F -- the force or acceleration of two given particles
#          Ep -- the potential energy of the two particles
###############################################################################   
def force(ri,rj,Length):
    # epsilon = 1
    import numpy as np
    dis=ri-rj   # i > j
    rcut = Length/2
    Dx = dis[0]
    
    
    # Long range correction for rcut
    Prcut = 4*((rcut**2)**(-6)-(rcut**2)**(-3))
    
    # Boundary conditions
    if Dx > 0.5*Length:
        Dx = Dx-Length
    if Dx < -0.5*Length:
        Dx = Dx+Length
    Dy = dis[1]
    if Dy > 0.5*Length:
        Dy = Dy-Length
    if Dy < -0.5*Length:
        Dy = Dy+Length
    Dz = dis[2]
    if Dz > 0.5*Length:
        Dz = Dz-Length
    if Dz < -0.5*Length:
        Dz = Dz+Length
    
    D2 = Dx**2+Dy**2+Dz**2


    # Using Lennar Johns to compute force and energy
    if D2 < rcut**2:
        Fx=48*(0.5*D2**(-4) - D2**(-7))*Dx
        Fy=48*(0.5*D2**(-4) - D2**(-7))*Dy
        Fz=48*(0.5*D2**(-4) - D2**(-7))*Dz
        F = np.array([Fx,Fy,Fz])
        
        
        Ep = 4*(D2**(-6)-D2**(-3))-Prcut
        
    else:
        F = np.array([0.,0.,0.])
        Ep = 0.
    return F, Ep
    
#-----------------------------------end---------------------------------------#









###############################################################################
# Function Name: Integ1
# Description: This code conmpute the integration of MD simulation using  
# Semi-implicit Euler method.
# Inputs: param-- the input parameters
#         pos -- the positions of all the particles
#         v -- the velocities of all the particles
# Outputs: Ekt -- kinetics energy
#          Ept -- potential energy
#           Tt -- temperatures
###############################################################################   
def Integ1(param,pos,v):
    import numpy as np
    
    # release parameters
    L = param[0]
    b = param[1]
    N = param[2]
    Time = param[3]
    dt = param[4]
    thermo = param[5]
    T0 = param[6]
    kb = param[7]
    # Initialized sets and set value for some parameters
    Ept = []
    Ekt = []
    Tt = []
    
    v2=v*v
    Ek=0.5*sum(sum(v2))
    T=2*kb*Ek/3/N    
    Ekt.append(Ek)
    Tt.append(T)
    
    # Start time loop---------------------------------------------------------
    for t in range(Time):
        ac = np.array(3*N*[0.]).reshape(N,-1)
        Ep = 0.0
        
        # Compute force and accelarate
        for i in range(N-1):
            for j in range(i+1,N):
                [f,e]=force(pos[i],pos[j],L*b)
                ac[i] = ac[i]-f
                ac[j] = ac[j]+f
                Ep=Ep+e
        
        
        # Compute velocity and position
        if thermo == 0:
            v=v+dt*ac
            
        else:
            v = ThermStat(v,T0,T,dt,ac)
            
            
        pos=pos+dt*v
        
        # Note that the position cannot exit the boundary
        for i in range(N):
            for k in range(3):
                if pos[i,k]<0:
                    pos[i,k]+L*b
                if pos[i,k]>L*b:
                    pos[i,k]-L*b
                    
                
        # Compute temperature and energy
        v2=v*v
        Ek=0.5*sum(sum(v2))
        T=2*kb*Ek/3/N        
        Tt.append(T)
        Ekt.append(Ek)
        Ept.append(Ep)
    # End time loop-----------------------------------------------------------
    
    # Compute the potential energy for the last state
    Ep=0.0
    for i in range(N-1):
        for j in range(i+1,N):
            [f,e]=force(pos[i],pos[j],L*b)
            Ep=Ep+e
    
    Ept.append(Ep)
    return Ekt, Ept, Tt


#-----------------------------------end---------------------------------------#









        
###############################################################################
# Function Name: Integ2
# Description: This code conmpute the integration of MD simulation using  
# Velocity Verlet method.
# Inputs: param-- the input parameters
#         pos -- the positions of all the particles
#         v -- the velocities of all the particles
# Outputs: Ekt -- kinetics energy
#          Ept -- potential energy
#           Tt -- temperatures
###############################################################################   
def Integ2(param,pos,v):
    import numpy as np
    # release parameters
    L = param[0]
    b = param[1]
    N = param[2]
    Time = param[3]
    dt = param[4]
    thermo = param[5]
    T0 = param[6]
    kb = param[7]
    # Initialized sets and set value for some parameters
    Ept = []
    Ekt = []
    Tt = []
    
    
    v2=v*v
    Ek=0.5*sum(sum(v2))
    T=2*kb*Ek/3/N    
    Ekt.append(Ek)
    Tt.append(T)
    
    # Compute for the a at initial time
    acs = np.array(3*N*[0.]).reshape(N,-1)
    Ep = 0.0
    for i in range(N-1):
        for j in range(i+1,N):
            [f,e]=force(pos[i],pos[j],L*b)
            acs[i] = acs[i]-f
            acs[j] = acs[j]+f
            Ep=Ep+e

    
    Ept.append(Ep)
    
    # Start time loop from time 1----------------------------------------------
    for t in range(Time):
        # Compute position first by pos = pos+dtv+0.5dt^2a
        pos=pos+dt*v+0.5*acs*dt**2
        # Note that the position cannot exit the boundary
        for i in range(N):
            for k in range(3):
                if pos[i,k]<0:
                    pos[i,k]+L*b
                if pos[i,k]>L*b:
                    pos[i,k]-L*b
        
        
        # Then acceleration
        ac = np.array(3*N*[0.]).reshape(N,-1)
        #ac = np.zeros(N,3)
        Ep = 0.0      
        # Compute force and accelarate
        for i in range(N-1):
            for j in range(i+1,N):
                [f,e]=force(pos[i],pos[j],L*b)
                ac[i] = ac[i]-f
                ac[j] = ac[j]+f
                Ep=Ep+e
        
        
        
        # Compute velocity 
        if thermo == 0:
            v=v+dt*(ac+acs)/2
        else:
            v = ThermStat(v,T0,T,dt,(ac+acs)/2)
        
        acs = ac

                    
                
        # Compute temperature and energy
        v2=v*v
        Ek=0.5*sum(sum(v2))
        T=2*kb*Ek/3/N

        Tt.append(T)
        Ekt.append(Ek)
        Ept.append(Ep)
        
    # End time loop-----------------------------------------------------------
    return Ekt, Ept, Tt

#-----------------------------------end---------------------------------------#






###############################################################################
# Function Name: ThermStat
# Description: This code use Berendsen thermostat to update velocity
# Inputs: v -- velocity; T0 -- desired temperature; T -- current temperature;
#         dt -- step size; acc -- acceleration
# Outputs: v -- updated velocity
###############################################################################  
def ThermStat(v,T0,T,dt,acc):
    tau=0.1
    lamda = (1+dt/tau*(T0/T-1))**0.5
    v=(v+dt*acc)*lamda
    return v
#-----------------------------------end---------------------------------------#









###############################################################################
# Function Name: PotEnerg
# Description: This code conmpute the total potential energy of a given state
# Inputs: param-- the input parameters
#         pos -- the positions of all the particles
# Outputs: Ept -- potential energy
###############################################################################         
def PotEnerg(pos,param):
    # epsilon = 1
    
    L = param[0]
    b = param[1]
    N = param[2]
    Length = L*b
    rcut = Length/2
    # Long range correction for rcut
    Prcut = 4*((rcut**2)**(-6)-(rcut**2)**(-3))
    
    
    Ep = 0.0    
    # Compute force and accelarate
    for i in range(N-1):
        for j in range(i+1,N):
            ri = pos[i]
            rj = pos[j]            
            dis=ri-rj   # i > j
            
            Dx = dis[0]
            if Dx > 0.5*Length:
                Dx = Dx-Length
            if Dx < -0.5*Length:
                Dx = Dx+Length
            Dy = dis[1]
            if Dy > 0.5*Length:
                Dy = Dy-Length
            if Dy < -0.5*Length:
                Dy = Dy+Length
            Dz = dis[2]
            if Dz > 0.5*Length:
                Dz = Dz-Length
            if Dz < -0.5*Length:
                Dz = Dz+Length
            
            D2 = Dx**2+Dy**2+Dz**2
        
            if D2 < rcut**2:
                Ep = Ep + 4*(D2**(-6)-D2**(-3))-Prcut
                
            else:
                Ep = Ep
    
    return Ep    
    
#-----------------------------------end---------------------------------------#






 
    