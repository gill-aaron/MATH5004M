# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from numpy import exp, sin, cos, pi
from numpy.fft import fftshift
import seaborn as sns

plt.rcParams.update({"font.size": 18})
plt.rcParams.update({"lines.linewidth": 2})

# Random vector
randomvec = np.random.rand(500)

#----------------------------------------------------------


# 2L-periodic function in [-L, L]
# Fourier extension in [-T, T]
L = 1
T = 2

# Number of collocation points before oversampling
# Odd includes right boundary, even excludes right boundary
# Even includes left boundary and midpoint, excludes right boundary
n_init = 5

# Gamma coefficient for oversampling
# (n = n_init * oversample + odd)
oversample = 2

# PDEs
t = 0.5
h = t/180

# Small value tolerance (for testing stability, -1 disables it)
tau = -1

# Plotting
sample_rate = 10
smp = np.round(n_init * sample_rate * T - 1).astype(int)
xview = 1


#----------------------------------------------------------

# Gets the correct number of points/coefficients when oversampling
# Note that the original input n becomes the number of coefficients
def oversampleFunc(N, gamma):
    # Uses (n = 2 * gamma * N + 1).
    if (N % 2 == 1):
        n_final = (N - 1) * gamma + 1
    else:
        n_final = N * gamma

    # Odd/even checker
    # Includes upper endpoint iff n_init was odd
    odd = (n_final % 2 == 1)
    
    return n_final, N, odd



#----------------------------------------------------------

# This is for interpolation
def bFunc(x, func_choice):
    if func_choice == 0:
        return exp(x)
    elif func_choice == 1:
        return x
    elif func_choice == 2:
        return cos(x * 1 * pi) 
    elif func_choice == 3:
        return 1 / (1 + 100 * x**2)
    
    
# This is for BVPS
def bvpFunc(x, func_choice):
    if func_choice == 0:
        r = (2 + x**2) * cos(x)
        r[0] = sin(1)
        r[-1] = sin(1)
        p = x
        q = np.zeros(x.size)
    
    elif func_choice == 1:
        r = (30 * x**3 * np.abs(x)) + (2 * x**5 * np.abs(x)) - 6 * x**6
        r[0] = -1
        r[-1] = 1
        p = -np.abs(x)
        q = np.ones(x.size) * 2
    
    elif func_choice == 2:
        r = 200 * (300 * x**2 - 1) / (100 * x**2 + 1)**3 + 1 / (100 * x**2 + 1)
        r[0] = 1/101
        r[-1] = 1/101
        p = np.zeros(x.size)
        q = np.ones(x.size)   
        
    elif func_choice == 3:
        r = x * exp(x)
        r[0] = -exp(-1)
        r[-1] = exp(1)
        p = np.ones(x.size) * -2
        q = np.ones(x.size) * 2
    
    elif func_choice == 4:
        r = 6 * x
        r[0] = -1
        r[-1] = 1
        p = np.zeros(x.size)
        q = np.zeros(x.size)
    
    return r, p, q

# This is for BVP solutions
def solutionFunc(x, func_choice):
    if func_choice == 0:
        return x * sin(x)
    elif func_choice == 1:
        return x**5 * np.abs(x)
    elif func_choice == 2:
        return 1/(1+100*x**2)
    elif func_choice == 3:
        return exp(x) * x
    elif func_choice == 4:
        return x**3


# This is for PDEs
def pdeSolutionFunc(x, t, func_choice, eqtype):
    if (func_choice == 0) and (eqtype == "heat"):
        return exp(-pi**2 * t / 4) * cos(pi * x / 2)
    
    elif (func_choice == 1) and (eqtype == "heat"):
        S = np.zeros(len(x), dtype="complex128")
        for n in np.arange(1, 253, 2): 
            b = 1/n**2
            S += b * cos(n * pi * x / 2) * exp(-pi**2 * n**2 * t / 4)
            #S += b * sin(n * pi * x) #* exp(-pi**2 * n**2 * t / 4)

    elif (func_choice == 2) and (eqtype == "heat"):
        S = np.zeros(len(x), dtype="complex128")
        for n in np.arange(1, 2 * len(randomvec), 2): 
            b = randomvec[n // 2] / n**2
            S += b * cos(n * pi * x / 2) * exp(-pi**2 * n**2 * t / 4)
    
    elif (func_choice == 0) and (eqtype == "wave"):
        return sin(pi * x) * cos(pi * t)

    elif (func_choice == 1) and (eqtype == "wave"):
        S = np.zeros(len(x), dtype="complex128")
        for n in np.arange(1, 101, 2): 
            b = integrate.quad(lambda x: ((exp(-20*x**2)-exp(-20)) * cos(n * pi * x / 2)), -1, 1, epsabs=1e-14)[0]
            S += b * cos(n * pi * x / 2) * cos(n * pi * t / 2)
    
    elif (func_choice == 2) and (eqtype == "wave"):
        S = np.zeros(len(x), dtype="complex128")
        for n in np.arange(1, 11, 2): 
            b = 1/n
            S += b * cos(n * pi * x / 2) * cos(n * pi * t / 2)
            
    return S

# Directs to one of the above functions depending on the problem
def pointFunc(y, problem, func_choice, time=False):
    if problem == 0:
        return bFunc(y, func_choice)
    elif problem == 1:
        return solutionFunc(y, func_choice)
    elif problem <=3:
        return pdeSolutionFunc(y, time, func_choice, "heat") 
    else:
        return pdeSolutionFunc(y, time, func_choice, "wave") 



#----------------------------------------------------------

# Used for some of the plots
def plottingFunc(x, y, S, n_final, fig, problem, func_choice, colour, time, T, gamma, add_dots=False,):
    if colour:
        col = colour
    elif problem == 1:
        col = "blue"
    elif problem >= 2:
        col = "purple"
    else:
        col = "green"
    
    # Main plot
    if fig == 1:  
        plt.plot(y, S, color=col)
        plt.plot(y, pointFunc(y, problem, func_choice, time), color="black", linestyle="dotted")
        
        if add_dots:
            plt.plot(x, pointFunc(x, problem, func_choice, time), linestyle="none", color="black", marker=".")
            
        if problem == 0:
            if T == 1:
                plt.legend(["f(x)", "$f_{}(x)$".format({n_final//2})]) 
            else:
                plt.legend(["f(x)", "$F_{}(x)$".format({n_final//2})]) 
        
        plt.tight_layout()
        
    # Error plotting
    # 2 is error plot, 2.5 returns error without plotting, 2.6 returns end error without plotting, 2.7 for midpoint error
    elif (fig >= 2) and (fig < 3):
        errors = S - pointFunc(y, problem, func_choice, time)
        
        max_err = 0
        
        for i in range(len(y)-1):
            if (abs(y[i])<= L) and (abs(y[i+1])<= L): 
                if fig == 2:
                    #plt.plot((y[i], y[i+1]), (abs(errors[i]), abs(errors[i+1])), color=col)
                    plt.plot((y[i], y[i+1]), (errors[i], errors[i+1]), color=col)
                
                if fig <= 2.5:
                    if abs(errors[i]) > max_err:
                        max_err = abs(errors[i])
                    if abs(errors[i+1]) > max_err:
                        max_err = abs(errors[i+1])
                  
                # Get endpoint errors
                elif fig == 2.6:
                    if (i > 0) and (i < len(y)-2):
                        if abs(y[i-1]) > L:
                            if abs(errors[i]) > max_err:
                                max_err = abs(errors[i])
                    
                    elif abs(y[i+2]) > L:
                        if abs(errors[i+1]) > max_err:
                            max_err = abs(errors[i+1])
                
                # Get midpoint error
                elif (fig == 2.7) and (i > 0): 
                    if (abs(y[i-1]) >= abs(y[i])) and (abs(y[i]) <= abs(y[i+1])):
                        if abs(errors[i]) > max_err:
                            max_err = abs(errors[i])

        if fig == 2:
            print(max_err)
            if add_dots:
                plt.plot(x, np.zeros(len(x)), linestyle="none", color="black", marker=".")
            plt.tight_layout()
        else:
            return max_err
    return S


#----------------------------------------------------------

# Gets the matrix named B on the project report
def getMultipliedMat(n_final, coeffs, end, x, klist, p, q):
    B = np.zeros((n_final, coeffs), dtype="complex128")
    multiplier = np.ones((n_final, coeffs), dtype="complex128")
    
    for k in range(1, n_final-1):
        multiplier[k] = -(pi * klist / end)**2 + (1j * pi * klist * p[k] / end) + q[k]
        B[k] = exp(klist * 1j * pi * x[k] / end) * multiplier[k]      

    B[0] = exp(-1j * pi * klist / end)
    B[-1] = exp(1j * pi * klist / end)
    
    return B
    

#----------------------------------------------------------

# Problem:
# 0: Interpolation
# 1: BVPs
# 2: Heat equation matrix exponential
# 3: Heat equation time-stepping
# 4: Wave equation

# Method:
# t: trapezium
# m: midpoint
# l: leapfrog (problem 4 only)

# plotvar:
# 0: Error
# 1: Eigenvalues
# 2: Amplification factor
# 3: Condition number


def pdeSolver(A, B, c, x, func_choice, problem, plotvar, time, method, coeff_eq):
    
    # Solves c for PDE matrix exponential method
    if problem == 2:
        # Demonstration purposes only, do not use generally
        if method == "p": 
            U = np.linalg.pinv(A, rcond=tau) @ B

        # The desired method
        else:                    
            U = np.linalg.lstsq(A, B, rcond=tau)[0]

        step = sp.linalg.expm(U * time)         
        c = step @ c


        # Problem 2 outputs
        if plotvar == 1:
            try:
                #eigenvals = exp(np.linalg.eigvals(U * t))
                eigenvals = np.linalg.eigvals(U * t)
                return max(np.abs(eigenvals.real))
            except:
                return  
        elif plotvar == 3:
            try:
                return np.max(np.linalg.cond(step))
            except:
                return 
            
    # Solves c for time-stepping method
    else:
        if time:
            loop = int(time / h)
        else:
            loop = 0
        
        try:
            diff = B @ np.linalg.pinv(A, rcond=tau)
        except:
            return
    
        idmat = np.identity(len(diff))
    
    
        if problem == 3: # Heat equation time-stepping
            if method == "t": # Trapezoidal
                if coeff_eq == 0: # Coefficient operator (preferred)
                    try:
                        diff = np.linalg.lstsq(A, B, rcond=tau)[0]
                    except:
                        return

                    idmat = np.identity(len(diff))
                    forward = (idmat + 0.5 * h * diff)
                    backward = (idmat - 0.5 * h * diff)

                    step = np.linalg.lstsq(backward, forward, rcond=tau)[0] 
                
                    try:
                        c = np.linalg.matrix_power(step, loop) @ np.linalg.lstsq(A, pdeSolutionFunc(x, 0, func_choice, "heat"), rcond=tau)[0]
                    except:
                        return
            
                else:
                    forward = idmat + 0.5 * h * diff
                    backward = idmat - 0.5 * h * diff
            
                    step = np.linalg.solve(backward, forward)           
                    u_current = np.linalg.matrix_power(step, loop) @ pdeSolutionFunc(x, 0, func_choice, "heat")
                
                    try:
                        c = np.linalg.lstsq(A, u_current, rcond=tau)[0]  
                    except:
                        return

            # Problem 3 outputs
            if plotvar == 1:       
                try:
                    eigenvals = np.linalg.eigvals(step)
                    return max(np.abs(eigenvals.real))
                except:
                    return
            elif plotvar == 2:
                try:
                    if coeff_eq == 1:
                        amp = max(np.abs(c / np.linalg.lstsq(A, pdeSolutionFunc(x, 0, func_choice, "heat"), rcond=tau)[0]))
                    else:
                        amp = max(np.abs(u_current[1:-1] / pdeSolutionFunc(x, 0, func_choice, "heat")[1:-1]))
                except:
                    return
            elif plotvar == 3:
                try:
                    return np.max(np.linalg.cond(step))
                except:
                    return
            

        elif problem == 4: # Wave equation (WIP)
            if method == "t": # Trapezoidal
                forward = idmat + 0.25 * h**2 * diff
                backward = idmat - 0.25 * h**2 * diff
    
                u_current = pdeSolutionFunc(x, 0, func_choice, "wave")
                v_current = np.zeros(len(u_current))

                if plotvar == 2:
                    amp = np.zeros(len(u_current))[1:-1]

                for i in range(loop):
                    u_next = forward @ u_current + h * v_current
                    u_next = np.linalg.solve(backward, u_next)
        
                    v_current = v_current + 0.5 * h * (diff @ (u_current + u_next))

                    if plotvar == 2:    
                        amp = np.maximum(amp, abs(u_next / u_current)[1:-1])

                    u_current = u_next
            
                try:
                    c = np.linalg.lstsq(A, u_current, rcond=tau)[0]
                except:
                    return
            
            elif method == "l": # Leapfrog
                u_current = pdeSolutionFunc(x, 0, func_choice, "wave")
                v_current = np.zeros(len(u_current))
            
                if loop > 0:
                    u_current = u_current + 0.5 * h * v_current
                    for i in range(loop-1):
                        v_current = (v_current + h * (diff @ u_current))
                        u_current = (u_current + h * v_current)
                
                
                    v_current = (v_current + h * (diff @ u_current))
                    u_current = (u_current + 0.5 * h * v_current)
            
                try:
                    c = np.linalg.lstsq(A, u_current, rcond=tau)[0]
                except:
                    return
    return c


def fftExt(n, end, L, gamma=oversample, samples=smp, xview=xview, fig=1, problem=0, coeff_eq=0, func_choice=0, plotvar=0, time=t, h=h, 
           method="t", colour=False, tau=tau):
    
    # Initialise variables
    n_final, coeffs, odd = oversampleFunc(n, gamma)
    x = np.linspace(-L, L, n_final, endpoint=odd)
    klist = np.linspace(0-n//2, (n-1)//2, coeffs, endpoint=True)
    c = np.zeros(coeffs)

    if problem == 1: # BVPs
        b, p, q = bvpFunc(x, func_choice)
        A = getMultipliedMat(n_final, coeffs, end, x, klist, p, q) 
        
    else:
        b = pointFunc(x, problem, func_choice, time=0)
        A = np.zeros((n_final, coeffs), dtype="complex128")
        for k in range(n_final):
            A[k] = exp(klist * 1j * pi * x[k] / end)


    # Gets coefficients Ac=b
    # (try / except used for functions that might crash the program when solution is unstable)
    if problem <= 2:
        try:
            c = np.linalg.lstsq(A, b, rcond=tau)[0]
        except:
            return
      
    # PDEs
    if problem >= 2:
        B = getMultipliedMat(n_final, coeffs, end, x, klist, np.zeros(n_final), np.zeros(n_final)) 
        B[0] = np.ones(coeffs) * 0
        B[-1] = np.ones(coeffs) * 0
         
        c = pdeSolver(A, B, c, x, func_choice, problem, plotvar, time, method, coeff_eq)

        # Various other tracked values get returned here
        if plotvar in [1,2,3]:
            return c

    # Interpolation using Fourier coefficients
    c = np.roll(fftshift(c), odd)    
    S = np.zeros(samples, dtype="complex128")
    y = np.linspace(-end * xview, end * xview, samples, endpoint=True)

    for k in range(np.ceil(-coeffs/2).astype(int), np.ceil(coeffs/2).astype(int)):
        S += c[k] * exp(k * 1j * pi * y / end)

    S = S.real

    return plottingFunc(x, y, S, n_final, fig, problem, func_choice, colour, time, T=end, gamma=oversample)



#----------------------------------------------------------

def compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, samples=smp, xview=xview, problem=0, func_choice=0, plotvar=0, coeff_eq=True,
                      time=t, h=h, method="t", tau=tau, Tseparate=True,  newplot=True, cols=False):
    
    # plotvar:
    # 0: Error plot
    # 0.1: End error plot
    # 0.2: Midpoint error plot
    # 0.3: Testing
    # 1: Eigenvalues
    # 2: Amplification factor
    
    points = 1 + np.ceil((n_upper - n_lower) / interval).astype(int)
    Tlen = len(Tvals)
    glen = len(gvals)
    
    if not(cols):
        cols = ["b", "r", "y", "m", 
                "c", "g", "brown", "black"]
        if glen == 1:
            if gvals[0] == 2:
                cols = ["r", "y", "m", 
                    "c", "g", "brown", "black"]
            elif gvals[0] == 8:
                cols = ["m", 
                    "c", "g", "brown", "black"]
   
    if plotvar < 1:
        figparam = 2.5 + plotvar
    else:
        figparam = 3
    
    for v in range(Tlen):
        if Tseparate or (v == 0) or (plotvar == 0.3):
            if plotvar == 0:
                if newplot:
                    plt.figure(figsize=(8, 6))
                plt.yscale("log")
                plt.xlabel("n")
                plt.ylabel("Maximum Error")
                plt.xlim(0, n_upper)
                plt.ylim(10**-16, 10**8)
            
            elif plotvar == 0.3:
                pass
                #plt.xlabel("n")
                #plt.ylabel("Position of Max Error")
                #plt.xlim(0, n_upper)
                #plt.ylim(-0.1, 1.1)
                
            else:
                if newplot:
                    plt.figure(figsize=(8, 6))
                #plt.yscale("log")
                plt.xlabel("n")
                if plotvar == 1:
                    plt.ylabel("Largest Eigenvalue")
                elif plotvar == 2:
                    plt.ylabel("Maximum Amplification Factor")
                else:
                    plt.ylabel("Condition number")
                plt.xlim(0, n_upper)
                #plt.ylim(bottom=1)
            
            
        endval = L * Tvals[v]
        for g in range(len(gvals)):
            i = 0
            n_plot = n_lower
            
            if plotvar == 0.3:
                errs = np.zeros([points, samples])
            else:
                 errs = np.zeros(points)

            nvals = np.linspace(n_lower, n_upper, points)
            while n_plot <= n_upper:
                errs[i] = fftExt(n_plot, endval, L, gamma=gvals[g], samples=samples, fig=figparam, problem=problem, func_choice=func_choice, plotvar=plotvar,
                                 time=time, h=h, coeff_eq=coeff_eq, method=method, tau=tau)
                
                if plotvar == 0:
                    errs[i, (errs[i] == 0) or (errs[i] == np.nan) or (errs[i] == np.inf)] = np.nan_to_num(np.inf)
                    
                n_plot += interval
                i += 1
            
            if plotvar == 0.3:
                plt.figure(figsize=(8, 6))
                sns.heatmap(errs.T, robust=True, cmap="inferno_r", norm="log")
                plt.tight_layout()
                
            elif Tseparate:
                plt.plot(nvals, errs, color=cols[g%8], label=["$γ = {}$".format({gvals[g]})])
                plt.legend(loc="upper right")
                
            else:
                plt.plot(nvals, errs, color=cols[v%8], label=["$T = {}$".format({Tvals[v]})])
                plt.legend(loc="upper right")
                
            print(str(Tvals[v]) + "-" + str(gvals[g]))

        if (len(Tvals) > 1) and Tseparate:
            plt.title("$T = {}$".format({Tvals[v]}))
        plt.tight_layout()
        
        #plt.draw()
        #plt.pause(1)

#----------------------------------------------------------

# PLOTS
# Uncommenting the below should generate the plots found in the write-up.
# compareParameters is the main function for comparing across different n values.
# fftExt can be used for plots at a specific n value.


# Gibbs phenomenon

#sample_rate = 100
#smp = np.round(n_init * sample_rate - 1).astype(int)

#L = 1
#T = 1
#xview = 2

#xplot = T * xview
#yplot = np.array([-0.4,4])

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#fftExt(4, T, L, gamma=1, problem=0, samples=smp, xview=xview, colour="red")

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#fftExt(10, T, L, gamma=1, problem=0, samples=smp, xview=xview, colour="red")

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#fftExt(20, T, L, gamma=1, problem=0, samples=smp, xview=xview, colour="red")

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#fftExt(100, T, L, gamma=1, problem=0, samples=smp, xview=xview, colour="red")


#----------------------------------------------------------

# Fourier extension

#sample_rate = 100
#smp = np.round(n_init * sample_rate * 2 - 1).astype(int)

#xview = 2
#xplot = xview
#yplot = np.array([-1, 4.5])
#n_init = 100
#L = 1


#T = 1

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#plt.title("T = 1")
#fftExt(n_init, T, L, gamma=1, problem=0, samples=smp, xview=xview, colour="red")

#T = 1.1

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#plt.title("T = 1.1")
#fftExt(n_init, T, L, gamma=1, problem=0, samples=smp, xview=xview)


#T = 2
#yplot = np.array([-6.5, 10])

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#plt.title("T = 2")
#fftExt(n_init, T, L, gamma=1, problem=0, samples=smp, xview=xview)


#----------------------------------------------------------

# Oversampling

#n_init = 51
#T = 1.5

#sample_rate = 100
#samples = np.round(n_init * sample_rate * T - 1).astype(int)
#xview = 1 

#xplot = 1
#yplot = np.array([5 * 10**-17,5 * 10**-8])

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#plt.yscale("log")
#plt.ylabel("Absolute error")
#plt.title("γ = 1")
#fftExt(n_init, T, L, gamma=1, problem=0, xview=xview, fig=2)

#plt.figure(figsize=(8, 6))
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#plt.yscale("log")
#plt.ylabel("Absolute error")
#plt.title("γ = 2")
#fftExt(n_init, T, L, gamma=2, problem=0, xview=xview, fig=2)

#----------------------------------------------------------

# Numerical testing


#sample_rate = 100
#smp = np.round(n_init * sample_rate * 8 - 1).astype(int)

#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [1, 2, 3, 4, 8]
#Tvals = [1.1, 2]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=0, samples=smp, func_choice=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=0, samples=smp, func_choice=3)


#sample_rate = 100
#smp = np.round(n_init * sample_rate * 4 - 1).astype(int)

#n_lower = 2
#n_upper = 300
#interval = 2
#gvals = [2]

#Tvals = [1.1, 2, 4, 8, 16, 32, 64]
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=0, samples=smp, func_choice=0, Tseparate=False)
#Tvals = [1.01, 1.025, 1.05, 1.075, 1.0875, 1.1, 2]
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=0, samples=smp, func_choice=3, Tseparate=False)

#----------------------------------------------------------

# BVPs

#sample_rate = 100
#smp = np.round(n_init * sample_rate * 8 - 1).astype(int)

#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1, 2]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=1, samples=smp, func_choice=3)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=1, samples=smp, func_choice=4)


#----------------------------------------------------------

# Heat equation ME

#sample_rate = 100
#smp = np.round(n_init * sample_rate * 8 - 1).astype(int)

#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1, 1.5, 2, 2.5]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=0, time=0.5, tau=-1)

#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=0, time=0.5, tau=-1, method="p")


#----------------------------------------------------------

# Heat equation TS

#sample_rate = 100
#smp = np.round(n_init * sample_rate * 8 - 1).astype(int)

#n_lower = 3
#n_upper = 501
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/180, tau=-1, coeff_eq=0)

#n_lower = 3
#n_upper = 251
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [2]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/180, tau=-1, coeff_eq=0)

#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/180, tau=-1, coeff_eq=1)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/180, tau=-1, coeff_eq=0)

#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [2]
#Tvals = [1.1]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/100, tau=-1, coeff_eq=0, newplot=False, cols="b")
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=-1, coeff_eq=0, newplot=False, cols="r")
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/10000, tau=-1, coeff_eq=0, newplot=False, cols="m")


#n_lower = 3
#n_upper = 501
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1, 2]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=2, time=0.5, h=1/1000, tau=-1, coeff_eq=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=10**-14, coeff_eq=0)

#n_lower = 3
#n_upper = 201
#interval = 2
#gvals = [1, 2]
#Tvals = [1.1]


#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=0, time=0.5, tau=-1, plotvar=1)

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=0, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=10**-14, coeff_eq=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=1, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=10**-14, coeff_eq=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=10**-14, coeff_eq=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=10**-14, coeff_eq=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=10**-14, coeff_eq=0)


#----------------------------------------------------------

# Cutoff


#sample_rate = 100
#smp = np.round(n_init * sample_rate * 8 - 1).astype(int)

#n_lower = 3
#n_upper = 501
#interval = 2
#gvals = [1, 2, 3, 4]
#Tvals = [1.1, 2]


#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=2, time=0.5, h=1/1000, tau=10**-9)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=2, time=0.5, tau=10**-9)


#n_lower = 3
#n_upper = 501
#interval = 2
#gvals = [2]

#cols = ["b", "r", "y", "m", 
#         "c", "g", "brown", "black"]

#compareParameters(n_lower, n_upper, interval, gvals, [1.1], L, problem=2, samples=smp, func_choice=2, time=0.5, tau=10**-9, newplot=False, cols=cols[0])
#compareParameters(n_lower, n_upper, interval, gvals, [2], L, problem=2, samples=smp, func_choice=2, time=0.5, tau=10**-9, newplot=False, cols=cols[1])
#compareParameters(n_lower, n_upper, interval, gvals, [4], L, problem=2, samples=smp, func_choice=2, time=0.5, tau=10**-9, newplot=False, cols=cols[2])
#compareParameters(n_lower, n_upper, interval, gvals, [8], L, problem=2, samples=smp, func_choice=2, time=0.5, tau=10**-9, newplot=False, cols=cols[3])


#n_lower = 3
#n_upper = 5003
#interval = 100
#gvals = [2]
#Tvals = [1.1]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, func_choice=0, plotvar=0, tau=10**-10)


#n_lower = 3
#n_upper = 501
#interval = 2
#gvals = [1,2,3,4]
#Tvals = [2]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=2, time=0.5, tau=10**-9, plotvar=1)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=2, time=0.5, h=1/1000, tau=10**-9, plotvar=1)


#n_lower = 3
#n_upper = 101
#interval = 2
#gvals = [1,2,3,4]
#Tvals = [1.1, 2]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, samples=smp, func_choice=0, time=0.5, tau=-1, plotvar=1)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, samples=smp, func_choice=0, time=0.5, h=1/1000, tau=-1, plotvar=1)




#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------



#n_lower = 3
#n_upper = 301
#interval = 2
#gvals = [2]
#Tvals = [1.1]

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, func_choice=0, plotvar=0)





#T = 2
#n_init = 21


#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, colour="red")
#plt.title("n = 21")
#plt.tight_layout()

#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, tau=10**-10, colour="green")
#plt.title("n = 21")
#plt.tight_layout()

#n_init = 31

#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, colour="red")
#plt.title("n = 31")
#plt.tight_layout()

#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, tau=10**-10, colour="green")
#plt.title("n = 31")
#plt.tight_layout()

#n_init = 51

#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, tau=10**-10, colour="green")
#plt.title("n = 51")
#plt.tight_layout()

#n_init = 101

#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, tau=10**-10, colour="green")
#.title("n = 101")
#plt.tight_layout()

#n_init = 301

#xplot = T
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, tau=10**-10, colour="green")
#plt.title("n = 301")
#plt.tight_layout()

#n_init = 1001

#xplot = T
#.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, fig=1, tau=10**-10, colour="green")
#plt.title("n = 1001")
#plt.tight_layout()




#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="l", plotvar=0.1)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="l", plotvar=0.2)

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=0, method="t", plotvar=0)

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=2, func_choice=0, method="t", plotvar=0)



#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="l", plotvar=0, tau=10**-8, h=t/1000)

#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="l", plotvar=0, tau=10**-10, h=t/1000)




#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-1)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-2)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-3)
###compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-4)
###compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-5)
###compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-6)
###compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-7)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-8)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-9)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-10)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-11)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-12)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-13)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=10**-14)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="t", plotvar=0, tau=-1)




#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, func_choice=0, method="t", plotvar=1, coeff_eq=True)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, func_choice=0, method="t", plotvar=2, coeff_eq=True)

##compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=3, func_choice=0, method="t", plotvar=0)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=0, method="t", plotvar=1, coeff_eq=False)
#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=0, method="t", plotvar=2, coeff_eq=False)


#compareParameters(n_lower, n_upper, interval, gvals, Tvals, L, problem=4, func_choice=2, method="l", h=t/1000, plotvar=0)


#xplot = L #* T
#yplot = np.array([-2, 2])
#plt.figure()
#plt.xlim(-xplot, xplot)
#plt.ylim(yplot[0], yplot[1])
#fftExt(n_init, T, L, oversample, problem=2, func_choice=0, method="t", fig=1, plotvar=0)


#xplot = L
#plt.figure()
#plt.xlim(-xplot, xplot)
#fftExt(n_init, T, L, oversample, problem=4, func_choice=2, method="m", fig=2)


