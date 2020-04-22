#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:51:40 2020

@author: YXZhang, Ajesh Kumar

Simulation codes for simulating floquet Hamiltion and dynamics on 1-D chain:
              sum_i (h_i X_i) (for t = 0 to T/3)
    H = 3/T * sum_i (Jz_i Z_i Z_i+1) (for t = T/3 to 2T/3)
              sum_i (Jx_i X_i X_i+1) (for t = 2T/3 to T)
where X's and Z's are pauli matrices; depending on the case we might want 
open/closed chain; all parameters could be disordered

"""
import random
import numpy as np
import scipy
from scipy import linalg

'''
By using the symmetry of the system, we know that a state with parity even should
never mix with odd and vice versa. Therefore we can work in even and odd basis seperately
and reduce the dimension of Hilbert space. The following function returns odd parity basis only
'''
def get_basis(l):
    basis=np.zeros((pow(2,l-1),l))
    for i in range(pow(2,l-1)):
        count = 0
        for j in range (l-1):
            basis[i,j] = int ((i%(2)**(j+1))/((2)**(j)))
            count = count + int ((i%(2)**(j+1))/((2)**(j)))
        basis[i,l-1] = (count+1)%2
    return basis

#generates disordered parameters
def parameters(h_max, Jz_max, Jx_max, h_dis, Jz_dis, Jx_dis):
    h = np.zeros(l)
    Jz = np.zeros(l)
    Jx = np.zeros(l)
    for i in range (l):
        if h_dis == True:
            h[i] = random.uniform(-1*h_max, h_max)
        else:
            h[i] = h_max
        if Jz_dis == True:
            Jz[i] = random.uniform(-1*Jz_max, Jz_max)
        else:
            Jz[i] = Jz_max        
        if Jx_dis == True:
            Jx[i] = random.uniform(-1*Jx_max, Jx_max)
        else:
            Jx[i] = Jx_max
    return h, Jz, Jx

#Hamiltionian matrices; equivlently one could simply try doing the tensor products. 
    
def H_x(h, l):
    H = np.zeros((hamdim,hamdim))
    basis = get_basis(l)
    for i in range(hamdim):
        for j in range(l):
            H[i][i] += 2 * (basis[i][j] - 1/2) * h[j]
    return H

def H_zz(Jz, l, closed_chain):
    H = np.zeros((hamdim,hamdim))
    basis = get_basis(l)
    for i in range(hamdim):
        if closed_chain == True:
            for j in range(0, l):
                f = int((i + 2 ** (j) * (-2) * (basis[i][j] - 1/2) + 2 ** ((j + 1)% (l) )* (-2) * (basis[i][(j + 1) % (l)] - 1/2) )% (2**(l-1)))#j mod l due to the periodic boun
                H[i][f] += (-1)* (-2) * (basis[i][j] - 1/2) * (-2) * (basis[i][(j + 1)% (l)] - 1/2)* Jz[j]
        else:
            for j in range(1, l): # there's no interaction between 0 and l-1 th qubit
                f = int((i + 2 ** (j-1) * (-2) * (basis[i][j-1] - 1/2) + 2 ** (j) * (-2) * (basis[i][j] - 1/2)) % (2**(l-1)))
                H[i][f] += (-1) * (-2) * (basis[i][j] - 1/2) * (-2) * (basis[i][j-1] - 1/2) * Jz[j]
    return H    

def H_xx(Jx, l, closed_chain):
    H = np.zeros((hamdim,hamdim))
    basis = get_basis(l)
    for i in range(hamdim):
        if closed_chain == True:
            for j in range(1, l+1):
                H[i][i] += 2*(basis[i][j-2] - 1/2) * 2*(basis[i][j-1] - 1/2) * Jx[j-1]               
        else:
            for j in range(1, l): # there's no interaction between 0 and l-1 th qubit
                H[i][i] += 2*(basis[i][j] - 1/2) * 2*(basis[i][j-1] - 1/2) * Jx[j-1]
    return H   

# floquet operator
def floquet(H_x, H_xx, H_zz):
    u_x = scipy.linalg.expm(-1j*H_x)
    u_xx = scipy.linalg.expm(-1j*H_xx)
    u_zz = scipy.linalg.expm(-1j*H_zz)
    floquet = np.linalg.multi_dot([u_xx,u_zz,u_x])
    return floquet

def r_test(U_floquet):
    eigs,vecs = np.linalg.eig(U_floquet)
    thetas = np.angle(eigs)
    nens = sorted(thetas, key=float)
    vals = len(nens)
    delts = [(nens[i+1] - nens[i]) for i in range (vals - 1)]
    rlist = [min(delts[j], delts[j+1])/max(delts[j], delts[j+1]) for j in range (vals - 2)]
    return np.average(rlist)
'''
main sequence for r test; sample_size is # of disorders; h_max, Jz_para, Jx_para are 
values, h_dis, Jz_dis, Jx_dis are bool values for whether one wants to add disorders
one could also decide whether or not using OBC
'''         
def main_r_test(sample_size, h_max, Jz_para, Jx_para, h_dis, Jz_dis, Jx_dis, closed_chain):      #lb for lower bound, up for upper bound; a grid serch over the parameter space              ;
    result = []
    J = np.ones(l) * Jz_para
    u_zz = scipy.linalg.expm(-1j* H_zz(J, l, closed_chain))
    V = np.ones(l) * Jx_para
    u_xx = scipy.linalg.expm(-1j* H_xx(V, l, closed_chain))
    r_result = np.zeros(sample_size)
    for k in range (sample_size): # numb of trials to be averaged over disorder
        h = parameters(h_max, Jz_para, Jx_para, h_dis, Jz_dis, Jx_dis)[0]
        u_x = scipy.linalg.expm(-1j*H_x(h, l))
        r_result[k] = r_test(np.linalg.multi_dot([u_xx,u_zz,u_x]))
        result.append([l, Jz_para, Jx_para, np.average(r_result), np.std(r_result)/(np.sqrt(sample_size - 1))])
    return np.asarray(result)


#time correlation for a site in the chain; no Jx and Jz disorders are considered for now. 
def time_correlation(disorder_size, time, target_site, h_max, Jz_para, Jx_para):
    correx = np.zeros((disorder_size, time))
    correz = np.zeros((disorder_size, time))
    J = np.ones(l) * Jz_para #parameters of each site0
    V = np.ones(l) * Jx_para
    h = np.zeros(l) 
    paulix = np.array([[1, 0],[0, -1]]) #in x basis 
    pauliz = np.array([[0, -1j],[1j, 0]])
    x0 = np.kron(np.identity(2 ** int(target_site)), np.kron(paulix, np.identity(2 ** (l - target_site - 2)))) #x(0)
    z0 = np.kron(np.identity(2 ** int(target_site)), np.kron(pauliz, np.identity(2 ** (l - target_site - 2))))
    u_zz = scipy.linalg.expm(-1j * H_zz(J, l, False))
    u_xx = scipy.linalg.expm(-1j * H_xx(V, l, False))
    inv_u_zz = scipy.linalg.expm(1j * H_zz(J, l, False))
    inv_u_xx = scipy.linalg.expm(1j * H_xx(V, l, False))
    for i in range (disorder_size):
        for j in range (l):
            h[j] = random.uniform(-1 * h_max, h_max)
        init_state = np.ones(hamdim) /np.sqrt(hamdim)
        u_x = scipy.linalg.expm(-1j * H_x(h, l))
        u_floquet = np.linalg.multi_dot([u_xx, u_zz, u_x])
        inv_u_x = scipy.linalg.expm(1j * H_x(h, l))
        inv_u_floquet = np.linalg.multi_dot([inv_u_x, inv_u_zz, inv_u_xx])
        xt = x0
        zt = z0
        for j in range (time):
            correx[i][j] = np.trace((np.dot(xt, x0)))/hamdim
            correz[i][j] = np.trace((np.dot(zt, z0)))/hamdim
            xt = np.linalg.multi_dot([u_floquet, xt, inv_u_floquet])
            zt = np.linalg.multi_dot([u_floquet, zt, inv_u_floquet])
    return correx, correz

# An example:
l = 6
hamdim = 2**(5) #reduced H_dim by factor of 2
Jz = 1/10 * np.pi/2
Jx = 1/10 * np.pi/2
result=main_r_test(100, np.pi, Jz, Jx, True, False, False, True)
print(result)


        

