
"""

CHEM 272_Project 2

Particle Movement Simulation 
in Lennard-Jones Potential  
using Metropolis Monte-Carlo Algorithm
-Python Code

-Group 2-
David Houshangi
Yash Maheshwaran
Yejin Yang
Christian Fernandez
Seungho Yoo


"""

import numpy as np
import matplotlib.pyplot as plt

#1. Initialize current locations of N particles within a square box of side length 2R.
#   Particles are placed with uniform random distribution in both x and y directions.
#   Positions are sampled in [-R, R] x [-R, R].
def PlotLocation(R = 100, N = 200):
    Xinit = np.random.uniform(-R, R, (N,))
    Yinit = np.random.uniform(-R, R, (N,))
    return Xinit, Yinit


#2. Define the pairwise Lennard–Jones potential.
#   a: repulsive coefficient (∝ r^-12)
#   b: attractive coefficient (∝ r^-6)
#   - a, b absorb usual LJ constants (e.g., 4εσ^12, 4εσ^6).
#   - Self distances r_ii are set to ∞ to avoid division by zero.
def Potential(Dx, Dy, a = 4, b = 4):
    r = np.sqrt(Dx**2 + Dy**2)                  # Pairwise distances
    np.fill_diagonal(r, np.inf)                 # Avoid self-interaction
    Phi = a/r**12 - b/r**6                      # Lennard-Jones potential
    np.fill_diagonal(Phi,0)                     # Zero out self-interaction
    return Phi


#3. Compute total potential energy for each particle.
#   Utot[i]: total energy of particle i from interactions with all other particles
def DisToPotential(Xinit, Yinit, a, b, Ones):
    N = len(Xinit)
    # Tile to produce pairwise coordinate matrices (N x N)
    Xi_tile = np.tile(Xinit, (N,1))
    Yi_tile  = np.tile(Yinit, (N,1))
    
    Dx = Xi_tile - Xi_tile.T                    # pairwise Δx_ij = x_i - x_j, shape (N, N)
    Dy = Yi_tile - Yi_tile.T                    # pairwise Δy_ij = y_i - y_j, shape (N, N)
    
    Phi = Potential(Dx, Dy, a, b)               # Phi[i, j] = Lennard–Jones potential between particle i and particle j
    Utot = np.dot(Phi, Ones)                    # Per-particle total potential
    return Utot


# 4. Main Metropolis Monte Carlo loop:
#    - PROPOSAL: move ALL particles simultaneously
#    - EVALUATION: compute per-particle energy changes dU[i] using the old vs trial configurations.
#    - ACCEPT/REJECT: applied per particle using dU[i] (elementwise).
#      Note: This "joint proposal (all particle movements) + elementwise acceptance" can violate detailed balance.
def MoveParticle(Xinit, Yinit, Niter = 10_000, a = 4, b = 4, T = 1, sigma = 0.01, R=15):
    N = len(Xinit)
    Ones = np.ones((N,))

    for n in range(Niter):
        # 4a) Propose Gaussian displacements for all particles
        #    - dx, dy ~ Gaussian(0, sigma^2) for each particle
        #    - sigma controls the typical displacement size per move
        dx = np.random.normal(0, sigma, (N,))                   # sigma sets typical step scale (standard deviation)
        dy = np.random.normal(0, sigma, (N,))
        
        Xtri = Xinit + dx
        Ytri = Yinit + dy
        
        # 4b) Compute Per-particle energy between old and trial configurations
        #    - Uold[i] = total potential energy of particle i with all others (original positions)
        #    - Utri[i] = total potential energy of particle i with all others (trial positions)
        Uold = DisToPotential(Xinit, Yinit, a, b, Ones)
        Utri = DisToPotential(Xtri, Ytri, a, b, Ones)
        dU = Utri - Uold                                        # Per-particle energy difference due to moving all particles
        
        # 4c) Metropolis acceptance test (elementwise )
        #    - If dU < 0 → accept (energy decrease)
        #    - If dU > 0 → accept with probability exp(-dU / T)
        #    - criteria is clipped to avoid overflow in exp()
        rho = np.random.uniform(0,1, size = dU.shape)
        criteria = np.clip(-dU / T, -700, 700 )                 # Avoid overflow
                                                                # 700 is chosen because np.exp(709) ≈ 8.2e307 is near the maximum representable float64 value
        accept = (dU < 0) | (rho < np.exp(criteria))
        move_indices = np.flatnonzero(accept)                   # Indices of accepted moves
        
        # 4d) Update accepted particles (elementwise)
        for i in move_indices:
            Xinit[i] = Xtri[i]
            Yinit[i] = Ytri[i]
            
        # 4e) Visualization: plot every 100 iterations
        if not n%1000:
            plt.figure(figsize=(5, 4)) 
            plt.scatter(Xinit, Yinit, c='gray', alpha=0.5, s=30)
            plt.xlabel('x', fontsize=15)
            plt.ylabel('y', fontsize=15)
            plt.tick_params(axis='both', labelsize=15)
            plt.title(f'after {n} iterations\na = {a}, b = {b}, T = {T}, N = {N}', fontsize=20)
            plt.show()


# 5) Simple wrapper class to bundle parameters and run the simulation.
class SimulateParticles:
    def __init__(self, N=200, R=100, a=4, b=4, T=1, sigma=0.01):
        # Simulation parameters:
        # N     = number of particles
        # R     = half-width of initial square placement range [-R, R]
        # a, b  = Lennard-Jones parameters (repulsion, attraction)
        # T     = temperature for Metropolis acceptance
        # sigma = standard deviation of Gaussian displacement per move
        self.N = N
        self.R = R
        self.a = a
        self.b = b
        self.T = T
        self.sigma = sigma
        # Initialize particle positions randomly in the square box [-R, R] × [-R, R]
        self.Xinit, self.Yinit = PlotLocation(R, N)

    def run(self, Niter=10_001):
        # Execute the Metropolis Monte Carlo simulation
        MoveParticle(self.Xinit, self.Yinit, Niter,
                     self.a, self.b, self.T, self.sigma, self.R)

"""
# Example run with default setting:
sim = SimulateParticles(N=100,a = 4, b = 4, T = 1, sigma = 0.05)
sim.run(Niter=10_001)

# Example run with higher a:
sim = SimulateParticles(N=100,a = 20, b = 4, T = 1, sigma = 0.05)
sim.run(Niter=10_001)

# Example run with higher b:
sim = SimulateParticles(N=100,a = 4, b = 20, T = 1, sigma = 0.05)
sim.run(Niter=10_001)
"""
# Example run with higher T:
sim = SimulateParticles(N=100,a = 4, b = 4, T = 100, sigma = 0.05)
sim.run(Niter=5_001)
"""
# Example run with low T:
sim = SimulateParticles(N=100,a = 4, b = 4, T = 0.01, sigma = 0.05)
sim.run(Niter=10_001)

# Example run with lower N:
sim = SimulateParticles(N=20,a = 4, b = 4, T = 1, sigma = 0.05)
sim.run(Niter=10_001)

# Example run with higher N:
sim = SimulateParticles(N=2_000,a = 4, b = 4, T = 1, sigma = 0.05)
sim.run(Niter=201)
"""