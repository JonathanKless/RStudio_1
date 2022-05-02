
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate

#plt.close('all')
"""
## ######################## Description###########################
Implementation of an implicit numerical scheme to solve Radial Flow of Water into a single root using Richards equation.
 A built-in function in Python is used for solving ODEs (ordinary differential equation).
Here a Midpoint Center Finite Difference method expands the derivatives in space.
Written by Dr. Mohsen Zare for the course of Soil " Modeling of Flow and Transport in Soil and Plant, Chair of Soil Physics, University of Bayreuth
 """

#%%  ########### Geometry information and discretization########
Ltot= 2000  # total root length [cm]
FractionActive= 0.5  # Fraction of total root length active in water uptake [>0 and <=1]
Lact=FractionActive*Ltot  # Total root length active in water uptake
rPot= 5   # radius of plant pot [cm]
hPot=10   # hieght of soil in the plant pot [cm]
rRoot=0.08   # average root radius [cm]
vPot=np.pi*rPot**2*hPot   # volume of soil [cm^3]
rb= np.sqrt(vPot/(np.pi*Ltot))   # half distance (from root center) between roots assuming a uniform root legth distribution


nNode=100               # number of nodes
r=np.linspace(rRoot,rb, nNode)  # to discretize the soil domain
nR=len(r)            # total number of grids

dr=np.diff(r)      # step size between numerical nodes
rMinus=np.append(r[0], r[1:]-dr/2)    # position of r_i-0.5 , to keep the size consistent a ghost node is add at the biggining
rPlus=np.append(r[:-1]+dr/2, r[-1])   # position of r_i+0.5, to keep the size consistent a ghost node is add at the end
dr=(rPlus+rMinus)/2    # to update step size between numerical nodes (dr), node the size of first and last node will be half of the others as we postion firs and last node at start and end of soil domain

#%% ########## Time information and discretization###############
tFinal=20*24*60*60        # desired time for the simulation [s]

#%% !!!!!!!!!!!!!! To specify the hydraulic paramters of soil based on van Genuchten-Mualem equation
VGP=dict()
VGP['Theta_s']=0.43
VGP['Theta_r']=0.078
VGP['alpha']=0.016  #[cm^-1]
VGP['n']=1.56
VGP['Ksat']=0.04 #[cm s-1]
VGP['lambda_k']=0.5
#%% ############## To define initial condition####################
""" Here I assumed that always equilibrium in matric potentil is considered as the initial condition
"""
hBot_t0=-100                 # matric potential at bottom
hTop_t0= hBot_t0            # matric potential at soil surface
h0=np.linspace(hTop_t0,hBot_t0, nR)  # to initialize the profile of

#%% ########### To impose Boundary Conditions #####################
#!!!!!!!!! To impose a fixed pressure at the root surface
#hSurf=-1
#h0[0]=hSurf
#!!!!!!!!! To impose a fixed flux at the root surface
#Plant information
E=2/3600    # plant transpiration rate [g/s  or cm^3/s]
qSurf=-E/(2*np.pi*rRoot*Lact)# -7.7e-6  # water flux into the root at root surface [cm/s], assuming a uniform root water uptake, negative for root water uptake


#!!!!!!!!! To impose a fixed flux at the root surface
#hBulk=-3
#h0[-1]=hBulk
#!!!!!!!!! To impose a fixed flux at the outermost part of soil
qBulk=0

#%% To define VGE function that takes h and VGP as input and returns Theta, K, and C as outputs
def VGE(h,VGP):
    import numpy as np

    # to extract VG parameters from dictionary
    Theta_r=VGP['Theta_r']
    Theta_s=VGP['Theta_s']
    alpha=VGP['alpha']
    n= VGP['n']
    Ksat= VGP['Ksat']
    lambda_k= VGP['lambda_k']
    m=1-1/n

    # Compute the volumetric moisture content
    Theta =(Theta_s - Theta_r)/(1 + (alpha*abs(h))**n)**m +Theta_r
    if np.any(h>=0):
        Theta[h>=0]=Theta_s
    ## Compute the effective saturation
    Se=(Theta - Theta_r)/(Theta_s - Theta_r)

    ## Compute the hydraulic conductivity
    K= Ksat*Se**(lambda_k)*(1 - (1 - Se**(1/m))**m)**2
    if np.any(h>=0):
        K[h>=0]=Ksat
    ### Compute the C (capacity of soil retention curve =dThta/dh)
    C=-(-Theta_r + Theta_s)*((alpha*abs(h - 1/2))**n + 1)**(-1 + 1/n) + (-Theta_r + Theta_s)*((alpha*abs(h + 1/2))**n + 1)**(-1 + 1/n)
    return Theta, K, C

#%% Function to average K at the locatoin of grides (between the nodes KPlus: i+0.5 or KMinus:i-0.5)
def ConducAveragingArithmaticMean(K):
    ##############          averaging based on Arithmatic mean
    KPlus=0.5*(K[1:]+K[:nR-1])
    KPlus=np.append(KPlus,[K[-1]], axis=0)          # to add a ghost node to keep the size similar.
    KMinus=0.5*(K[:-1]+K[1:])
    KMinus=np.append([K[0]],KMinus, axis=0)          # to add a ghost node to keep the size similar
    return KPlus, KMinus


#%% ####### To implement the implicit numerical scheme for solving Richards equation for the case of radially symmetric water flow into a single root
def RichardsImplicit(t, h0):

#%% To define h from the time step (explicit methods) or iterative level (implicit methods)
    hT=h0
    DthataDt=np.zeros(nR)   # prelocate an empty array which will be need latter

#%% Get C,K,Theta and average it at the locatino of grides (between the nodes KPlus: i+0.5 or KMinus:i-0.5)
    Theta,K,C = VGE(hT,VGP)
        ##############          averaging based on Arithmatic mean
    KPlus, KMinus = ConducAveragingArithmaticMean(K)


    #%%%%%%%########## 	  Force BC First Node,
    node=0
    ### ###########!!!!!!The case of Flux Boundary!!!!!!!!!!!!!!###
    DthataDt[node]= (
        (1/(r[node]*(rPlus[node]-rMinus[node])))*(
            rPlus[node]*KPlus[node]*(hT[node+1]-hT[node])/dr[node] #flux into the node
            -rMinus[node]*(-qSurf)#  flux out of the node
    )
    )

    #%%%%%%%%%%%%%%%%%Intermediate ndoes
    node=np.arange(1,nR-1,1)
    DthataDt[node]= (
        (1/(r[node]*(rPlus[node]-rMinus[node])))*(
            rPlus[node]*KPlus[node]*(hT[node+1]-hT[node])/dr[node] #flux into the node
            -rMinus[node]*KMinus[node]*(hT[node]-hT[node-1])/dr[node] #  flux out of the node
    )
    )

    #%%%%%%%%%%%%%%%    ##   Force BC Last Node,
    node=-1
    ##############!!!!!! case 'Constant or Variable Pressure'!!!!!!!!!!!!!!###


    ##############!!!!!! case 'Constant or Variable Flux at bottom'!!!!!!!!!!!!!!###
    DthataDt[node]= (
        (1/(r[node]*(rPlus[node]-rMinus[node])))*(
            rPlus[node]*(-qBulk) #flux into the node
            -rMinus[node]*KMinus[node]*(hT[node]-hT[node-1])/dr[node] #  flux out of the node
    )
    )


#%% compute DhDt based on the slop of soil retention curve
    DhDt=DthataDt*0
    DhDt[hT<0]=DthataDt[hT<0]/C[hT<0]# to avoide divsion by zero
    if np.any(hT>=0):    # for the case that soil is saturated C=D_Theta/Dh=0 ==> we should force the DhDt to zero otherwise if will be unknown
        DhDt[hT>=0]=0

    return DhDt


   #%%%%%%%%%%%%%%To set up an ODE solver in Python and get the simulation

#%% Due to the radial nature of water flow, water flux increases as radius decreases, and we get closer to the root surface. This may result in a big gradient in matric potential at the root surface. Solving the water flow equation in such a case will be challenging and may not converge. Therefore here, we add a new function that will be evaluated at each time step, and it stops the solution when matric potential at the root surface reaches a critical value.
def StoppingCriteria(t, h0):
    h_cr= -15000   # minimum expected matric potential at root surface
    if abs(h0[0])<=abs(h_cr):
        Flag=1
    else:
        Flag=0
    return Flag

StoppingCriteria.terminal = True
StoppingCriteria.direction = 0

#%%%%%%%%%%%%%%To set up an ODE solver in Python and get the simulation
t=[0, tFinal]

# to specifiy the time at solver should return the solutions
#t_sel=np.linspace(0, t[-1], 100)

Sol=scipy.integrate.solve_ivp(RichardsImplicit, t, h0,  method='BDF', dense_output=True, rtol=1e-7,atol=1e-6, max_step=int(t[-1]/200), events=StoppingCriteria)

print(Sol.message)

t=Sol.t   # the time steps that solver evaluated the solution of ODE
h=Sol.y   # solution of ODE at a specific time
# To compute the profile of water content based on VG equation
Theta,K,C = VGE(h,VGP)

#%% to plot the simulated results
#profiles as a function of depth
Xloc=1.01
Yloc=1

### to select time at where profile should be ploted. Since profiles at dry conditions are more interesting here
#the lines below help to select more frequently at end.
tPlot1=np.logspace(0, np.log10(len(t)-1), 2, dtype=int)
tPlot1=1+abs(tPlot1-tPlot1[-1])[::-1]  # I think this is just used to kick negatives out
tPlot2=np.linspace(0, len(t)-1, 2, dtype=int)
tPlot=np.unique(np.append(tPlot1, tPlot2))

# with plt.style.context('bmh'):
#     plt.figure(1,figsize=(17,5))
#     plt.subplot(1,2,1)
#     legend_tem=[]
#     for i in tPlot:
#         plt.plot(r-r[0], h[:,i])
#         legend_tem=np.append(legend_tem,'t='+str(int(t[i]*100/3600)/100) + 'h')
#     plt.xlabel("Distance from root surface [cm]")
#     plt.ylabel("Soil Matric Potential [cm]")
#     plt.legend(legend_tem, frameon=False, bbox_to_anchor=(Xloc, Yloc))

DT = (480*3600)/t[-1]

for i in tPlot:
    print(str(int(round((t[i]*100/3600)*DT/100))))
    

    # plt.subplot(1,2,2)
    # legend_tem=[]
    # for i in tPlot:
    #     plt.plot(r-r[0], Theta[:,i])
    #     legend_tem=np.append(legend_tem,'t='+str(int(t[i]/3600)) + 'h')
    # plt.xlabel("Distance from root surface [cm]")
    # plt.ylabel("Soil Water Content [$cm^{3} cm^{-3}]$")

    # plt.subplot(1,3,3)
    # plt.plot(h[-1,:], h[0,:])
    # plt.plot(h[0,:], h[0,:])
    # plt.xlabel("Soil Matric Potential at Bulk Soil [cm]")
    # plt.ylabel("Soil Matric Potential at Root-Soil Interface [cm]")
    # plt.legend(["t=480h", 't=0h'], frameon=False, bbox_to_anchor=(Xloc, Yloc))
    # plt.tight_layout()
    # plt.savefig('Fig1.png', dpi=300)   # to save the figure


    # plt.figure(2,figsize=(9,5))
    # plt.subplot(1,2,1)
    # plt.imshow(np.abs(h), cmap='jet', extent=[t[0],t[-1]/60/60,r[-1]-r[0],0], aspect='auto')
    # plt.colorbar(label='Soil Matric Potential [cm]')
    # plt.xlabel("Time [h]")
    # plt.ylabel("Distance from root surface [cm]")

    # #plt.axis('auto')
    # plt.subplot(1,2,2)
    # plt.imshow(Theta, cmap='jet', extent=[t[0],t[-1]/60/60,r[-1]-r[0],0], aspect='auto')
    # plt.colorbar(label='Soil Water Content[$cm^{3} cm^{-3}]$')
    # plt.xlabel("Time [h]")
    # plt.ylabel("Distance from root surface [cm]")
    # #plt.axis('auto')
    # plt.show()
    # plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.35, wspace=0.4)
