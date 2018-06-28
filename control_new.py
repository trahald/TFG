#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### This is code which given some vector field $X_1,X_2,\ldots$, generates
###  the corresponding space $\mathrm{Lie}(X_1,X_2,\ldots)$ and checks whether it 
### spans the tangent space or not.

from __future__ import division # so that 1/2 does not give the loathed 0
from IPython.display import display, Math, Latex # beautiful output

from sympy import *
import numpy as np
import cmath

init_printing()  # For pretty printing.

####### GLOBAL CONTROLS ######################################################

COUPLING_CONSTANT   = 1.0

MASS     = 1.0
K_SPRING = 1.0
GAMMA    = 1.0

####### Defining most of the variables #######################################

t = symbols('t', real=True);
q,p,rx,ry,rz= symbols('q p r_x r_y r_z');
epsilon = symbols('epsilon', real=True); 

sig0   = Matrix([[1, 0],[0, 1]])#/sqrt(2)
sig1   = Matrix([[0, 1],[1, 0]])#/sqrt(2)
sig2   = Matrix([[0,-I],[I, 0]])#/sqrt(2)
sig3   = Matrix([[1, 0],[0,-1]])#/sqrt(2)
u1base = Matrix(4,1,([sig0,sig1,sig2,sig3]))

################################### New hamiltonian.

#######
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

VARIABLES = Matrix([p,q,rx,ry,rz,Hx,Hy,Hz])

m, k = symbols('m k', real=True); 

gamma = GAMMA
        
epsilon = symbols('epsilon', real=True); 

####### THE FUNCTION $f(q)$ which gives the coupling strength.

#f = 1/sqrt(1+q**2)
#f = 1 + COUPLING_CONSTANT*(q**2 + p**2)
#f=1 + COUPLING_CONSTANT*q*p
f=q

def sum1 ():
    sum1 = 0
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                sum1 = sum1 + -I*2*I* LeviCivita3[i,j,k] * r[j] * H[i] * cr[k]
    return sum1;


# This is Pontryagin's hamiltonian, see Pontryagin's theorem.
h = ( cq*p/m + cp*(-k*q -2* (rx*Hx+ry*Hy+rz*Hz)*(diff(f,q))) + sum1() + 
     1/2*(cHx*cHx + cHy*cHy + cHz*cHz) )
h = simplify(h)

# The dynamics are given by $\frac{\delta h}{\delta c_x}$
xtotdot = []
for i in range(0,int(len(x_tot)/2)):
    xtotdot.append( diff(h,cx[i]))    
#for i in range(0,int(len(x_tot)/2)):
#    xtotdot.append(-diff(h,x[i]))

# 'velocity' is the vector field which gives the dynamics, $\dot{x}=X$
velocity = Matrix(xtotdot)


# Now we decompose it in the form

X_0 = simplify(   velocity.subs(cHx,0).subs(cHy,0).subs(cHz,0))
X_1 = simplify(  (velocity-X_0).subs(cHy,0).subs(cHz,0)/cHx  )
X_2 = simplify(  (velocity-X_0).subs(cHx,0).subs(cHz,0)/cHy  )
X_3 = simplify(  (velocity-X_0).subs(cHx,0).subs(cHy,0)/cHz  )
X   = X_0 + cHx*X_1 + cHy*X_2 + cHz*X_3


##############################################################################
####### COMMUTATORS ##########################################################
##############################################################################

# We assume next that x_a,x_b are components of vectors $X_a$, $X_b$ in the basis 
# v[0] = $\partial_{q}$, v[1] = $\partial_{p}$,
# v[2,3,4] = $\partial_{\rho}$,
# v[5,6,7] = $\partial_{\mathcal{H}}$,
# Now, by analyzing $[X_a,X_b]$ by hand where (implicit sum over repeated
# indices) we have that $x_a=x_a^i \partial_i$, $x_b = x_b^j \partial_j$,
# and from applying the operator to $f$, $[x_a,x_b]f$ we get that 
# $[x_a,x_b]f = (x_a^i(\partial_i x_b^j) - x_b^i\partial_i x_a^j)\partial_jf$
# So now we just apply this formula.
def commutator_lie(x_a,x_b):
    #var     = [I_c,a_c,I_1,a_1,I_2,a_2]
    dim = VARIABLES.shape[0]
    new_res = []
    if x_a == x_b:
        for i in range(0,dim):
            new_res.append(0)
        return new_res
    
    for j in range(0,dim):
        summ = 0
        for i in range(0,dim):
            summ = summ + ( x_a[i]*diff(x_b[j],VARIABLES[i]) - 
                            x_b[i]*diff(x_a[j],VARIABLES[i])   )
        new_res.append(summ) 
        
    return simplify(new_res)

def vector_in_span(family,candidate):
    nr_fam = len(family)
    dim    = len(family[0])
    M      = Matrix( dim, 1, family[0])
    for e in range(1,nr_fam):
        M = M.col_insert(e,Matrix(family[e]))
    rank_M    = len( (M.rref())[1] )
    augM      = M.col_insert(nr_fam,Matrix(candidate))
    rank_augM = len( augM.rref()[1] )
    if rank_M == rank_augM:
        return True
    else:
        return False
    
# I am assuming that 'family' is linearly independent already.
def is_controllable(family):
    dim    = len(family[0])
    nr_fam = len(family)
    if nr_fam == 1:
        print('It cannot be controllable if it only has one dimension' + 
              'unless the space is one dimensional. We need at least two' +
              'in order to take the lie brackets, on which this method is' +
              'based.')
        return False
    M = Matrix( dim, 1, family[0])
    for e in range(1,nr_fam):
        M = M.col_insert(e,Matrix(family[e]))
    rank_M = len( (M.rref())[1] )
    if rank_M != nr_fam:
        print('Warning: family not linearly independent.')
    
    fam = family  # We will use this auxiliary vector to modify it.
    nr_fam = len(fam)
    #print('nr_fam = ',nr_fam)
    # We do this so we won't have repeated pairs like [1,2] and [2,1] but just
    # [1,2]. This is because the commutator is symmetric [1,2] = -[2,1] so 
    # it's just redundant information.
    index_pairs = []
    i_list_aux = [i for i in range(0,nr_fam)] 
    j_list_aux = [j for j in range(0,nr_fam)]
    while i_list_aux:  # while list not empty
        i = i_list_aux.pop()
        for j in j_list_aux:
                index_pairs.append([i,j])    
        j_list_aux.pop()
    #print(index_pairs)

    new_len_fam = len(fam)
    while ( new_len_fam<dim and index_pairs ):  # While index_pairs not empty.
        [i, j] = index_pairs.pop()
        candidate = simplify(commutator_lie(fam[i],fam[j]))
        print(candidate)
        if not vector_in_span(fam,candidate):
            fam.append(candidate)
            new_len_fam = len(fam)
            for e in range(0,new_len_fam):
                index_pairs.append([new_len_fam-1,e])
        new_len_fam = len(fam)
    #display(fam)
    if (len(fam)==dim):
        M = Matrix( dim, 1, fam[0])
        for e in range(1,new_len_fam):
            M = M.col_insert(e,Matrix(fam[e]))
        rank_M = len( (M.rref())[1] )
        display(M.rref())
        if (len(fam) == rank_M):
            return True
        else:
            print('Warning: good length but rank unequal.')
            return False
    else:
        return False
    
print('Is this system controllable?', is_controllable([X_0,X_1,X_2]))