"""
SIR model implementation for assignment 2 (part 2) of
the Tokyo Insitute of Technology - Fall 2015 Complex Network
- Instructor: Assoc. Prof. Tsuyoshi Murata
- Date: Jan 26, 2016
- Deadline: Feb 01, 2015
- Student: NGUYEN T. Hoang - M1
- StudentID: 15M54097
- Python version: 2.7.10
- Origin: jiansenlu.blogspot.com.co/2010/06/solve-sir-model-c.html
"""

import numpy as np
import scipy.integrate as ig
import pylab as pl

# Parameter
beta = 0.8
gamma = 0.8
time_resolution = 1
plot_length = 100.0
S0 = 0.99
I0 = 0.01
R0 = 0.0
INIT = (S0, I0, R0)

def diff_eqs(PARAM, t):
    dY = np.zeros(3)
    V = PARAM
    dY[0] = - beta * V[0] * V[1]
    dY[1] = beta * V[0] * V[1] - gamma * V[1]
    dY[2] = gamma * V[1]
    return dY

t_start = 0.0
t_end = plot_length
t_res = time_resolution
t_range = np.arange(t_start, t_end + t_res, t_res)
RES = ig.odeint(diff_eqs, INIT, t_range)

print RES

pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,2], '-g^', label='Recovereds')
pl.plot(RES[:,1], '-ro', label='Infectionous')
pl.legend(loc=1)
pl.title('SIR epidemic model without births or deaths')
pl.xlabel('Time')
pl.ylabel('Portion of Susceptibles, Recovereds, and Infectionous')
pl.savefig('SIR.png', dpi=300)
pl.show()
