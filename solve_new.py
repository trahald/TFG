#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Code to solve the differential equation in order to find the optimum
### trajectory minimizing a cost functional defines as
### $J=\int \dot{u}^2 dt + \frac{1}{\gamma^2}G(x(T))$

from __future__ import division # so that 1/2 does not give the loathed 0
from IPython.display import display, Math, Latex # beautiful output

from scipy.integrate import odeint
import scipy.integrate as inte
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import pylab
from sympy import *

import matplotlib as mpl

import numpy as np
import cmath

#init_printing()  # For pretty printing.

####### Control constants ####################################################
INTEGRATION_METHOD  = 'bdf'
NSTEPS              = 5000000
INTEGRATOR          = 'vode'
T0, T1              = 0.0, 1.0
DT                  = (T1-T0)/1000.0 

NR_ITERATIONS       = 3
ITERATION_DELTA     = 1.0
GAMMA_QUANT         = (1e-6)**(0.5) ## NEVER 0, we divide by gamma
GAMMA_CLASS         = (1e-3)**(0.5)    ## NEVER 0, we divide by gamma
COUPLING_CONSTANT   = 1.0

MASS     = 1.0
K_SPRING = 0.001

GRAD_NEWTON_DELTA = 1e-6

# BFGS parameters
BFGS_ITERATION = 100
BFGS_ALPHA = 0.001
BFGS_TOLERANCE = 2

## Gradient descent parameters.
GRAD_TOLERANCE = 1
GRAD_STEP      = 0.1
GRAD_ITER      = 1
RANDOM_Y0_CONTRIBUTION = 1

# Initial conditions for our variables..
def setup_coord (q0_fin, p0_fin, phi1, theta1, Hx0, Hy0, Hz0, cx):
    phi1, theta1   = (phi1)*np.pi/180, (theta1)*np.pi/180
    rx0_fin = 1 * np.cos(phi1) * np.sin(theta1)
    ry0_fin = 1 * np.sin(phi1) * np.sin(theta1)
    rz0_fin = 1 * np.cos(theta1)
    y_blanco1    = np.array([q0_fin, p0_fin,
                        rx0_fin, ry0_fin, rz0_fin,
                        Hx0, Hy0, Hz0,
                        cx[0], cx[1], cx[2],
                        cx[3], cx[4], cx[5],
                        cx[6], cx[7]], dtype=np.float64)
    return y_blanco1

y0_ini = setup_coord(   1, 1.5,  90,  90, 0, 0, 0, [0,0,0,0,0,0,0,0])


NR_COORD = y0_ini.shape[0]

def first_half(v):
    dim = v.shape[0]
    return np.array([v[i] for i in range(0, int(dim/2))], dtype=np.float64)
def second_half(v):
    dim = v.shape[0]
    return np.array([v[int(dim/2)+i] for i in range(0, int(dim/2))],
                                                          dtype=np.float64)

y0_ini = setup_coord(1.1, 0.2,  -21, 124, 0.1, 0.2, 0.3, 
                                 [3.2,1,-2,-3.3,1.,6,.6,0.3])
y0_ini_x   = first_half(y0_ini)
y0_ini_cx  = second_half(y0_ini)
GLOB_y0_ini_x = y0_ini_x


#random_addition  = np.array([ random.random()
#                     for j in range(0,8) ],dtype=np.float64 )
#y0_ini_cx  = y0_ini_cx + RANDOM_Y0_CONTRIBUTION * random_addition
#y0_ini_cx = np.array([-0.00097155, -0.01765558,  0.02165133, -0.01016424,
#                      0.00900474,  0.01375255,  0.02176548,  0.00433647])


# These serve to impose the 'final' conditions on c_x. Note these aren't the
# the boundary conditions, just the objective states which we will need in
# order co compute c_x (T).
y_blanco = [first_half(setup_coord(1.5,  0.5,  36., 110., 0., 0., 0.,
                                               [0.,0.,0.,0.,0.,0.,0.,0.])),
            first_half(setup_coord(   2.,   2., -26.,  33., 0., 0., 0.,
                                               [0.,0.,0.,0.,0.,0.,0.,0.]))]
#            first_half(setup_coord(   3.,   -1., -65., -295., 0., 0., 0.,
#                                               [0.,0.,0.,0.,0.,0.,0.,0.]))]
'''
y_blanco = [first_half(setup_coord(1.5, 0.5,  36, 110, 0, 0, 0,
                                               [0,0,0,0,0,0,0,0]))]
'''
NR_BLANCOS = len(y_blanco)


####### Defining most of the variables #######################################
t = symbols('t', real=True);

q,p,rx,ry,rz,Hx,Hy,Hz         = symbols('q p r_x r_y r_z H_x H_y H_z');
cq,cp,crx,cry,crz,cHx,cHy,cHz = symbols(
                    'c_q c_p c_rx c_ry c_rz c_Hx c_Hy c_Hz');

LeviCivita3 = np.array([[[int((i - j) * (j - k) * (k - i) / 2)
                    for k in range(3)] for j in range(3)] 
                    for i in range(3)], dtype=np.float64)
        
x_tot     = Matrix([q,p,rx,ry,rz,Hx,Hy,Hz,cq,cp,crx,cry,crz,cHx,cHy,cHz])
size_xtot = int(len(x_tot))
x         = Matrix([q,p,rx,ry,rz,Hx,Hy,Hz])
cx        = Matrix([cq,cp,crx,cry,crz,cHx,cHy,cHz])
r         = Matrix([rx,ry,rz])
H         = Matrix([Hx,Hy,Hz])
cr        = Matrix([crx,cry,crz])
cH        = Matrix([cHx,cHy,cHz])

m, k = symbols('m k', real=True); 
    
epsilon = symbols('epsilon', real=True); 

####### THE FUNCTION f(q)

#f = 1/sqrt(1+q**2)
#f=1+COUPLING_CONSTANT*(q**2 + p**2)
#f=1+COUPLING_CONSTANT*q*p
f=q
#f=1
#######

def sum1():
    sum1 = 0
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                sum1 = sum1 + LeviCivita3[i,j,k] * f * H[i]  * r[j] * cr[k]
    return sum1


# Here we use Pontryagin's hamiltonian from Pontryagins theorem.
h = ( cq*p/m + cp*(-k*q -2*(rx*Hx+ry*Hy+rz*Hz)*(diff(f,q))) + sum1() + 
     1/2*(cHx*cHx + cHy*cHy + cHz*cHz) )
h = simplify(h)

# These are then the differential equations:
xtotdot = []
for i in range(0,int(len(x_tot)/2)):
    xtotdot.append( diff(h,cx[i]))    
for i in range(0,int(len(x_tot)/2)):
    xtotdot.append(-diff(h,x[i]))

velocity = Matrix(xtotdot)
longitud = velocity.shape[0]
velocity = velocity.subs(m,MASS).subs(k,K_SPRING)

## I will calculate jac. scipy ode for documentation.
def jacobian(X):
    var = x_tot
    long = int(len(x_tot))
    Dfun=zeros(long,long)
    for i in range(0,long):
        for j in range(0,long):
            Dfun[i,j]=diff(X[i],var[j])
    return Dfun

xtot_list = [q,p,rx,ry,rz,Hx,Hy,Hz,cq,cp,crx,cry,crz,cHx,cHy,cHz]

Dfun_sym = jacobian(velocity)

# Now, evaluating jacobian at a number is slow, thus we will
# lambdify the function to make it quicker.
def jac_lambdi(Dfun_sym):
    Dfun_l = [None]*size_xtot
    for index in range(0,size_xtot):
        Dfun_l[index] = [None]*size_xtot
    for i in range(0,size_xtot):
        for j in range(0,size_xtot):
            Dfun_l[i][j] =  lambdify( [xtot_list,t],
                                      simplify(Dfun_sym[i,j]) )
    return Dfun_l

Dfun_l = jac_lambdi(Dfun_sym)
        
def gradient(t,var):
    res = [None]*size_xtot
    for index in range(0,size_xtot):
        res[index] = [None]*size_xtot
    for i in range(0,size_xtot):
        for j in range(0,size_xtot):
            res[i][j] = (Dfun_l[i][j])(var,t)
    return res;

fx = []
for i in range(0,size_xtot):
    fx.append( lambdify( [xtot_list,t], velocity[i]) )
def func(t,var):
    res = [None]*size_xtot  # Initialized with 6 components 
    for i in range(0,size_xtot):
        res[i] = fx[i](var,t)
    return res;    

def f(t, y, arg1):
    #return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
    return func(t,y)
def jac(t, y, arg1):
    #return [[1j*arg1, 1], [0, -arg1*2*y[1]]]
    return gradient(t,y)
ode = inte.ode(f, jac).set_integrator(INTEGRATOR, nsteps=NSTEPS, #order=15,
                                  method=INTEGRATION_METHOD)

y0 = np.array(y0_ini, dtype=np.float64)
t0, dt, t1 = T0, DT, T1

### This is our time evolution core.
def L(cy0):
    y0 = np.append(GLOB_y0_ini_x, cy0)  # y0_ini_cx is a global variable
    ode = inte.ode(f, jac).set_integrator(INTEGRATOR, nsteps=NSTEPS, order=15,
                                  method=INTEGRATION_METHOD)
    ode.set_initial_value(y0,T0).set_f_params(1.0).set_jac_params(1.0)
    while ode.successful() and ode.t < T1:
        ode.integrate(ode.t + DT)
    return ode.y

# This calculates the gradient of G, somewhat self explanatory.
def dGdx (y, y_b):
    return -1.0* np.array( [ (y[0]-y_b[0])/(GAMMA_CLASS**2),
                             (y[1]-y_b[1])/(GAMMA_CLASS**2),
                             0,#(y[2]-y_b[2])/(GAMMA_QUANT**2),
                             0,#(y[3]-y_b[3])/(GAMMA_QUANT**2),
                             0,#(y[4]-y_b[4])/(GAMMA_QUANT**2),
                              0, 0, 0 ],  dtype=np.float64 )
    
# I use the following two functions later on to print debugging
# messages.
def coste_qp (y,y_b):
    return ( 0.5 * (y[0]-y_b[0])**2 /(GAMMA_CLASS**2) +
             0.5 * (y[1]-y_b[1])**2 /(GAMMA_CLASS**2) )

def coste_quant (y,y_b):
    return ( 0.5 * (y[2]-y_b[2])**2 /(GAMMA_QUANT**2) +
             0.5 * (y[3]-y_b[3])**2 /(GAMMA_QUANT**2) + 
             0.5 * (y[4]-y_b[4])**2 /(GAMMA_QUANT**2))
    
    
def Newton_vect(cx_ini, y_b):
    y_iter = L(cx_ini)
    length = y_iter.shape[0]
    loop_max = int(length/2)
    
    cx_objective = dGdx(y_iter, y_b)
    
    cx_actual = []
    for i in range(0, loop_max):
        cx_actual.append(y_iter[loop_max+i])
    cx_actual = np.array(cx_actual, dtype=np.float64)

    return [cx_actual, cx_objective]

###################
# The following is the CORE function which measures how far we are from
# the optimal solution. We use a log to add numerical stability,
# it does not affect the position of extremum points.
# From here on we write some functions for the gradient of Newton_zero
# and other things. Not all functions will be used.
####################
def Newton_zero(y0_cx):
    var = Newton_vect(y0_cx, GLOB_y_blanco)
    return np.log((np.dot(var[0]-var[1],var[0]-var[1])))

cond_ini_cx = [1,2,3,4,5,6,7,5]
#basis:
identi = eye(len(cond_ini_cx))
basis_c = []
for e in range(0,len(cond_ini_cx)):
    basis_c.append(np.array(list(identi[e,:]), dtype=np.float64))


def partial_Newton(func, punto, basis_vector, epsi):
    return ( (func(punto + epsi*basis_vector) -
              func(punto - epsi*basis_vector) ) / (2*epsi) )
    

def grad_Newton(punto):
    #  We define basis vectores along which we will take the 
    #  partial derivative.
    cond_ini_cx = [1,2,3,4,5,6,7,5]
    longi = len(cond_ini_cx)
    identi = eye(longi)
    basis_c = []
    for e in range(0,longi):
        basis_c.append(np.array(list(identi[e,:]), dtype=np.float64))

    #  Now we calculate all the different partial derivatives.
    punto = np.array(punto, dtype=np.float64)
    res = []  # We will store the result in 'res' and we need it to be a 
                 # an array type.
    for e in range(0,punto.shape[0]):
        res.append(partial_Newton(Newton_zero, punto, basis_c[e],
                                                      GRAD_NEWTON_DELTA))
        #res.append( (Newton_zero(punto + GRAD_NEWTON_DELTA*basis_c[e]) -
        #              Newton_zero(punto - GRAD_NEWTON_DELTA*basis_c[e]) ) /
        #                                (2*GRAD_NEWTON_DELTA) )
    res = np.array(res, dtype=np.float64)
    return res

def hess_Newton(x,p):
    #  We define basis vectores along which we will take the 
    #  partial derivative.
    cond_ini_cx = [1,2,3,4,5,6,7,5]
    longi = len(cond_ini_cx)
    identi = eye(longi)
    basis_c = []
    for e in range(0,longi):
        basis_c.append(np.array(list(identi[e,:]), dtype=np.float64))
       
    #  Now we proceed to calculate the Hessian.
    x = (np.matrix(x)).T
    p = (np.matrix(p)).T
    dim = x.shape[0]
    B = np.matrix(np.eye(dim))
    res = []
    x = sanitize_x(x)
    for i in range(0,dim):
        aux =( ( grad_Newton(y0_ini_cx+GRAD_NEWTON_DELTA*basis_c[i])-
                 grad_Newton(y0_ini_cx-GRAD_NEWTON_DELTA*basis_c[i])   )
                  /2/GRAD_NEWTON_DELTA )
        B[:,i] = np.matrix(aux).T
    res = B*p
    #return res ## This returns H*p, a vector.
    return res  

### NOTE: As is, it does not work very well.
def gradient_descent(y0_cx, tolerance, step_size, max_steps):
    y0 = np.array(y0_cx, dtype=np.float64)
    counter = 0
    dist = 1e88
    while dist>tolerance and counter<max_steps:
        dist = Newton_zero(y0)
        vect = grad_Newton(y0)
        vect = vect/((np.inner(vect,vect))**0.5)
        y0 = y0 - step_size*vect
        counter = counter + 1
        print('loop: ', counter, ' dist = ', dist)
    return y0

# Some functions are picky as to whether the input is an array or list.
def sanitize_x (x):
    dim = x.shape[0]
    res = [None]*dim
    res = np.array(res, dtype=np.float64)
    for i in range(0,dim):
        res[i] = x[i,0] 
    return res

# This is a manual implementation of the BFGS optimization algorithm,
# but SciPy's implementation is better, and even so it does not converge.
def BFGS(y0_cx):
    # We are assuming y0_cx is an array.
    dim = y0_cx.shape[0]
    x = (np.matrix(y0_cx)).T  # now a vector
    B = np.matrix(np.eye(dim))
    inv_B = np.matrix(np.eye(dim))
    alpha = BFGS_ALPHA
    control_par = Newton_zero(sanitize_x(x))
    for i in range(0, BFGS_ITERATION):
        if (control_par > BFGS_TOLERANCE):
            print('BFGS iter = ', i)
            grad = np.matrix(grad_Newton(sanitize_x(x))).T
            p = -inv_B*grad
            s = alpha*p
            x = x + s
            grad_new = np.matrix(grad_Newton(sanitize_x(x))).T
            y = grad_new - grad
            yyT = y*y.T
            yTs = (y.T*s)[0,0]
            ssT = s*s.T
            sTy = (s.T*y)[0,0]
            syT = s*y.T
            ysT = y*s.T
            Bs = B*s
            B = B + ( (yyT)/(yTs) - ( (Bs*(s.T*B)) /((s.T*(Bs))) ) )
            inv_B = (inv_B + (ssT*(sTy + (y.T*(inv_B*y))[0,0]) / (sTy**2)) +
                                   + ( inv_B*ysT + syT*inv_B)/sTy )
            control_par = Newton_zero(sanitize_x(x))
            print('|df/ds| = ', np.sqrt(np.linalg.norm(grad_new)))
            print('dist = ',control_par)
        else:
            return sanitize_x(x)
    
    return sanitize_x(x)

### We will divide the [T0,T1] interval into various parts.
NR_POINTS, BIG_T0, BIG_T1 = len(y_blanco), 0.0, 1.0
#NR_POINTS = 1
T_INTERVAL_ITER = (BIG_T1-BIG_T0)/NR_POINTS

### We reasign values to the global variables just in case they changed
### values along the way.
y0_ini_x   = first_half(y0_ini)
y0_ini_cx  = second_half(y0_ini)
GLOB_y0_ini_x  = y0_ini_x
GLOB_y0_ini_cx = y0_ini_cx

### Plot lists.
ts  = [[] for i in range(0,NR_POINTS)]  # Here we will store the time.
res = [[] for i in range(0,NR_POINTS)]  # Here we will store the variables.
print('GLOB_y0_ini_x= ', GLOB_y0_ini_x)
COST_QUANT = 0
COST_CLASS = 0
endpoints = []  # Here we will store the end points from each
                # interval $[t_{j-1},t_j]$.
LIST_cx_ini_iterations = []
##################################################################
# THIS IS THE CORE LOOP WHERE WE OPTIMIZE EACH INTERVAL SEPARATELY
##################################################################
for iteration in range(0,NR_POINTS):
    print('Beginning loop')
    T0 = T_INTERVAL_ITER * iteration
    T1 = T_INTERVAL_ITER * (iteration + 1)
    DT = (T1-T0)/1000.0
    GLOB_y_blanco = y_blanco[iteration]
    #GLOB_y_blanco = y_blanco[1]
    print('GLOB_y_blanco = ', GLOB_y_blanco)
    print('Iteration nr. '+ str(iteration)+'. '
                  'Started optimization. Unknown wait time.')
    
    GLOB_y0_ini_cx = np.array([0,0,0,0,0,0,0,0], dtype=np.float64)    
    OUTPUT = minimize(Newton_zero, GLOB_y0_ini_cx, 
                   method='Nelder-Mead', # jac=grad_Newton, hess=hess_Newton,
                    options={'xtol': 1e-6, 'ftol':1e-6, #'adaptive': True,
                             'maxiter': 3000, 'disp': True})

    
    print('Final result. cx_0 = ', OUTPUT.x)
    y0_ini_cx_final = OUTPUT.x
    LIST_cx_ini_iterations.append(y0_ini_cx_final)  # All vectors saved in
                                                    # this list will be
                                                    # printed.
        
    print('Beginning to set initial values.')
    ode.set_initial_value(np.append(GLOB_y0_ini_x, y0_ini_cx_final),
                          T0).set_f_params(1.0).set_jac_params(1.0)
    print('Beginning to integrate.')
    while ode.successful() and ode.t < T1:
        ode.integrate(ode.t + DT)
        ts[iteration].append(ode.t)
        res[iteration].append(ode.y)
    print('Finished integration.')
    COST_CLASS = COST_CLASS + coste_qp(first_half(res[iteration][-1]),
                                       GLOB_y_blanco)
    COST_QUANT = COST_QUANT + coste_quant(first_half(res[iteration][-1]),
                                          GLOB_y_blanco)
    print(iteration, 'AUX cost class = ', COST_CLASS)
    print(iteration, 'AUX cost quant = ', COST_QUANT)
    GLOB_y0_ini_x = first_half(res[iteration][-1])
    endpoints.append(GLOB_y0_ini_x)

    print('GLOB_y0_ini_x= ', GLOB_y0_ini_x)
    print('Finished loop')
    
    
#############################################
### Here we write to file the calculates costates.

###################
CONTROL_RUN = 20   # This is useful to label different system runs.
###################

nombre_inicial = 'RUN='
file = open(nombre_inicial+str(CONTROL_RUN)+'cx_ini'+'.txt','w')
for el in range(0,len(LIST_cx_ini_iterations)):
    string = ''
    for e in range(0,len(LIST_cx_ini_iterations[el])):
        string = string + str(LIST_cx_ini_iterations[el][e]) + ' '
    file.write(string+'\n')
file.close()

nombre_inicial = 'RUN='
file = open(nombre_inicial+str(CONTROL_RUN)+'.yblanco'+'.txt','w')
for el in range(0,len(y_blanco)):
    string = ''
    for e in range(0,len(y_blanco[el])):
        string = string + str(y_blanco[el][e]) + ' '
    file.write(string+'\n')
file.close()

####################################################################
####################################################################
### From here on, we are mainly concerned with plotting. The core
### optimization and system evolution is done.
####################################################################

aux_ts = []
for iteration in range(0,NR_POINTS):
    for el in range(0,len(res[0])):
        aux_ts.append(iteration+ts[iteration][el])
ts = aux_ts     


plot_normarho = [] 
plot_usquared  = []
plotobject = [[] for e in range(0,16)]   # Beware! Change 16 to your 
                                         # number of variables.

### Now we fuse all the differnet iterations so that we can plot all the 
### intervals together.
for nr_iter in range(0,NR_POINTS):
    for el in range(0,len(res[0])):
        for coord in range(0,16):
            plotobject[coord].append(res[nr_iter][el][coord])
        plot_normarho.append(  (res[nr_iter][el][2]**2 + 
                                res[nr_iter][el][3]**2 +
                                res[nr_iter][el][4]**2)**(0.5) )
        plot_usquared.append( 0.5* (res[nr_iter][el][13]**2 + 
                                    res[nr_iter][el][14]**2 +
                                    res[nr_iter][el][15]**2)   ) 
    
COSTE_ENERGETICO = 0
for i in range(0,len(ts)):
    COSTE_ENERGETICO = COSTE_ENERGETICO + plot_usquared[i]*DT
print('COSTE \int 0.5u^2 = ', COSTE_ENERGETICO)
print('COSTE QUANT = ', COST_QUANT)
print('COSTE CLASS = ', COST_CLASS)


#### PLOT
    
# Sources: http://www.randalolson.com/2014/06/28/how-to-make
#            -beautiful-data-visualizations-in-python-with-matplotlib/
# These are the "Tableau 20" colors as RGB.   
tableau20=[(31, 119, 180),  (174, 199, 232), (255, 127, 14),  (255, 187, 120),    
           (44, 160, 44),   (152, 223, 138), (214, 39, 40),   (255, 152, 150),    
           (148, 103, 189), (197, 176, 213), (140, 86, 75),   (196, 156, 148),    
           (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
           (188, 189, 34),  (219, 219, 141), (23, 190, 207),  (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib
# accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

def plot_qp ():
    plt.figure(figsize=(10, 8)) 
    
    plt.style.use('ggplot')
    plt.style.use('seaborn-talk')
    plt.xlabel(r'Posición ($q$)')
    plt.ylabel(r'Momento ($p$)')
    plt.plot(plotobject[0], plotobject[1],
             color=tableau20[0], label=r'$q$ y $p$')
    for i in range(0,len(y_blanco)):
        plt.plot(y_blanco[i][0],y_blanco[i][1],
             marker = 'D', label= ('Objetivo '+str(i)), color=tableau20[0])
        
    for i in range(0,len(endpoints)):
        plt.plot(endpoints[i][0],endpoints[i][1], marker= 'o',
                 color = tableau20[0])
        
    plt.plot(y0_ini[0],y0_ini[1], marker= 'o',
             label=  'Pto. de la trayectoria', color=tableau20[0])
    plt.title(r'Evolución de un sistema híbrido controlado')
    plt.legend(loc='upper left')
    plt.show()
    #pylab.savefig('pq.png', bbox_inches='tight')
    
def plot_purity():
    plt.figure(figsize=(4, 4)) 
    plt.style.use('ggplot')
    plt.style.use('seaborn-talk')
    plt.xlabel(r'Tiempo ($t$)')
    plt.ylim(0,1.2)
    plt.plot(ts, plot_normarho,  color=tableau20[2], 
            label = r'Pureza $\sim\ \sqrt{\operatorname{tr}\ \rho^2}$)')
    plt.title(r'Evolución de la pureza')
    plt.legend(loc='upper left')
    plt.show()
    
def plot_usquare():
    plt.figure(figsize=(8, 8)) 
    plt.style.use('ggplot')
    plt.style.use('seaborn-talk')
    plt.xlabel(r'Tiempo ($t$)')
    #plt.ylim(0,1.2)
    plt.plot(ts, plot_usquared,  color=tableau20[2], 
            label = r'$\frac{1}{2}u^2$')
    plt.title(r'Coste energético')
    plt.legend(loc='upper left')
    plt.show()

#### Hamiltonian plot
##### Bloch sphere plot.
#Taken from: https://matplotlib.org/gallery/mplot3d/lines3d.html
def plot_Hamilt():
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure(figsize=(9, 9))
    plt.style.use('ggplot')
    plt.style.use('seaborn-talk')
    
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    
    #x = np.array([ res[el][5] for el in range(0,len(ts))], dtype=np.float64)
    #y = np.array([ res[el][6] for el in range(0,len(ts))], dtype=np.float64)
    #z = np.array([ res[el][7] for el in range(0,len(ts))], dtype=np.float64)
    x = np.array(plotobject[5])
    y = np.array(plotobject[6])
    z = np.array(plotobject[7])
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x1 = 1 * np.outer(np.cos(u), np.sin(v))
    y1 = 1 * np.outer(np.sin(u), np.sin(v))
    z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x1, y1, z1, color=tableau20[0], rstride=4, cstride=4,
                      alpha=0.25, label = 'Esfera de Bloch')
    
    ax.plot(x, y, z, color=tableau20[2], 
            label = r'Hamiltoniano $\mathcal{H}$ en el tiempo')
    #ax.scatter(rx0,ry0,rz0, label = 'Puntos fijos')
    ax.scatter(y0_ini[5],y0_ini[6],y0_ini[7], marker = 'D',
                               label = 'Punto de inicio')
    
    ax.legend()
    
    plt.show()
    #pylab.savefig('bloch.png', bbox_inches='tight')

def plot_rho ():
    ##### Bloch sphere plot.
    #Taken from: https://matplotlib.org/gallery/mplot3d/lines3d.html
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure(figsize=(9, 9))
    plt.style.use('ggplot')
    plt.style.use('seaborn-talk')
    
    
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    
    #2,3,4
    #x = np.array([ res[el][2] for el in range(0,len(ts))], dtype=np.float64)
    #y = np.array([ res[el][3] for el in range(0,len(ts))], dtype=np.float64)
    #z = np.array([ res[el][4] for el in range(0,len(ts))], dtype=np.float64)
    x = np.array(plotobject[2])
    y = np.array(plotobject[3])
    z = np.array(plotobject[4])
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x1 = 1 * np.outer(np.cos(u), np.sin(v))
    y1 = 1 * np.outer(np.sin(u), np.sin(v))
    z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x1, y1, z1, color=tableau20[0], rstride=4, cstride=4,
                      alpha=0.25, label = 'Esfera de radio 1')
    
    ax.plot(x, y, z, color=tableau20[2], label = r'Trayectoria de $\rho$')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.text(0, 0, +1.3, r'$\vert 1 \rangle$', size = 'xx-large')
    ax.text(0, 0, -1.3, r'$\vert 0 \rangle$', size = 'xx-large')
    
    for i in range(0,len(y_blanco)):    
        ax.scatter(y_blanco[i][2], y_blanco[i][3], y_blanco[i][4],
                   label = 'Objetivo '+str(i), marker = 'D')
        
    for i in range(0,len(endpoints)):    
        ax.scatter(endpoints[i][2], endpoints[i][3], endpoints[i][4],
                   c=tableau20[2], marker = 'o')
        
    ax.scatter(y0_ini[2],y0_ini[3],y0_ini[4], marker = 'o', c=tableau20[2],
                               label = 'Pto. trayectoria')
    
    ax.legend()
    plt.show()
    #pylab.savefig('bloch.png', bbox_inches='tight')

def plot_costates():
    labels = [r'$c_{q}$', r'$c_{p}$', r'$c_{\rho x}$', r'$c_{\rho y}$',
               r'$c_{\rho z}$',  r'$c_{\mathcal{H} x}$',
                r'$c_{\mathcal{H} y}$', r'$c_{\mathcal{H} z}$']
    plt.figure(figsize=(8, 8)) 
    plt.xlim(0,1.3)
    plt.style.use('ggplot')
    plt.style.use('seaborn-talk')
    plt.xlabel(r'Tiempo ($t$)')
    #plt.ylabel(r'Coestados')
    #plt.ylim(0,1.2)
    for el in range(0,8):    
        plt.plot(ts, plotobject[8+el],  color=tableau20[el], 
                 label = labels[el])
    plt.title(r'Los coestados')
    plt.legend(loc='center right')
    plt.show()
    
#plot_costates()
#plot_usquare()
plot_rho()
plot_qp()


#plot_Hamilt()
#plot_purity()