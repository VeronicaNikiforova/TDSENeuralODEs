# Finding Potential Function in Time-Dependent Schrodinger Equations Using Structure Preserving Neural ODEs

1. Datagen file generates training data, pairs of wavefunctions evolved through time with a fixed potential function.
2. Crank_nic has a custom forward and backward method that solves the augmented system of ODEs using Crank Nicolson, a structure-preserving solver.
3. Training file trains the neural network that approximates the potential to predict an evolved wavefunction. 
