import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import torch

#physical constants
hbar = 1.0
m = 1.0

#spatial grid: 500 points in [-1, 1]
num_points = 500
x_min, x_max = -1.0, 1.0
x = np.linspace(x_min, x_max, num_points)
dx = x[1] - x[0]
N = num_points - 2  #number of interior points

#time evolution parameters
steps = 50
T = 1.0
dt = T / steps

#empty arrays that the psi0s, psi1s, and the potentials will be stored in
num_wavefunctions = 250
psi0s = np.zeros((num_wavefunctions, num_points), dtype=np.complex128)
psi1s = np.zeros((num_wavefunctions, num_points), dtype=np.complex128)
potentials = np.zeros((num_wavefunctions, num_points), dtype=np.float64)

for i in range(num_wavefunctions):
    a = 20
    b = 0.5
    V_full = a * (x**2 - b)**2
    potentials[i] = V_full  #save the potential

    V_diag = diags(V_full[1:-1], 0)

#discrete Laplacian for interior points
e = np.ones(N)
D = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N)) / dx**2

#Hamiltonian operator
H = -(hbar**2 / (2 * m)) * D + V_diag
H = H.astype(np.complex128)

#Hermiticity check of Hamiltonian
if not np.allclose(H.toarray(), H.getH().toarray(), atol=1e-10):
    print("Warning: Hamiltonian is not Hermitian.")

#identity matrix
I = diags([np.ones(N)], [0], format="csc")

#Crank–Nicolson matrices
A = (I + 1j * dt / (2 * hbar) * H).tocsc()
B = (I - 1j * dt / (2 * hbar) * H).tocsc()
A_lu = splu(A)


tolerance = 1e-6  #tolerance for checks

#main loop: generate 250 different Gaussian psi0s, evolve to psi1s
for i in range(num_wavefunctions):
    #random variance in range
    sigma2 = np.random.uniform(0.04, 1.0)
    sigma = np.sqrt(sigma2)

    # Add diversity: randomly shift center μ and optionally add phase
    mu = np.random.uniform(-0.9, 0.9)  # random center within [-0.9, 0.9]
    k = np.random.uniform(-20.0, 20.0)     # random momentum (controls phase)

    # Diverse psi0: shifted Gaussian with traveling wave component
    # envelope = 1 - x**2  # smooth decay to 0 at boundaries
    envelope = np.sin(np.pi * (x + 1) / 2)
    psi0 = envelope * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) * np.exp(1j * k * x)

    psi0 = psi0.astype(np.complex128)


    norm_0_raw = np.sqrt(np.sum(np.abs(psi0)**2) * dx)  #before normalization
    psi0 /= norm_0_raw
    norm_0 = np.sqrt(np.sum(np.abs(psi0)**2) * dx)      #after normalization, should be ~1

    #boundary check for psi0
    if not (np.abs(psi0[0]) < tolerance and np.abs(psi0[-1]) < tolerance):
        print(f"Wavefunction {i} psi0 violates boundary conditions.")
    #save psi0
    psi = psi0.copy()
    psi0s[i] = psi0

    #time evolve with Crank–Nicolson for T seconds, checking norm at each step
    for step in range(steps):
        rhs = B @ psi[1:-1]
        psi_interior_new = A_lu.solve(rhs)
        psi[1:-1] = psi_interior_new
        psi[0] = 0
        psi[-1] = 0

        #check norm conservation during evolution
        norm = np.sum(np.abs(psi)**2) * dx
        if np.abs(norm - 1.0) > tolerance:
            print(f"Wavefunction {i} norm deviation at step {step}: {norm}")

        #check boundary conditions during evolution
        if not (np.abs(psi[0]) < tolerance and np.abs(psi[-1]) < tolerance):
            print(f"Wavefunction {i} violates boundary conditions at step {step}.")

    #save psi1
    psi1s[i] = psi.copy()

    #final normalization check for psi0 and psi1
    norm_1 = np.sum(np.abs(psi1s[i])**2) * dx
    if not (np.abs(norm_0 - 1.0) < tolerance and np.abs(norm_1 - 1.0) < tolerance):
        print(f"Wavefunction {i} failed final normalization: ψ0 norm={norm_0}, ψ1 norm={norm_1}")

    #final boundary check for psi1
    if not (np.abs(psi1s[i][0]) < tolerance and np.abs(psi1s[i][-1]) < tolerance):
        print(f"Wavefunction {i} psi1 violates final boundary conditions.")


#final outputs
print("Finished. Shapes:")
print("psi0s:", psi0s.shape)
print("psi1s:", psi1s.shape)

#print the final pair of psi0 and psi1 (real parts)
print("\nFinal ψ₀ vector (real part):")
print(psi0s[-1].real)

print("\nFinal ψ₁ vector (real part):")
print(psi1s[-1].real)

np.savez_compressed(
    "wavefunction_dataset_steps50.npz",
    x=x,
    psi0s=psi0s,
    psi1s=psi1s,
    potentials=potentials
)

print("Dataset saved to 'wavefunction_dataset.npz'")
