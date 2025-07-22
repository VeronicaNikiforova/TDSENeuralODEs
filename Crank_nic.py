import torch
from torch.autograd import Function
from scipy.sparse.linalg import splu
from scipy.sparse import diags
import numpy as np

class OdeintAdjointCN(Function):
    @staticmethod
    def forward(ctx, psi0, t, *theta_and_net):
        *theta, potential = theta_and_net
        psi_type = psi0.dtype
        N = psi0.shape[0] - 2
        device = psi0.device
        x = torch.linspace(-1.0, 1.0, N+2, device=device, dtype=torch.float32)[1:-1]
        dx = (x[1] - x[0]).item()
        dt = (t[1] - t[0]).item()
        steps = len(t)
        # Construct potential and Hamiltonian diagonals
        V_vec = potential(x).cpu().numpy()
        V_diag = diags(V_vec, 0)

        e = np.ones(N)
        D = -0.5*diags([e, -2*e, e], [-1, 0, 1], shape=(N, N)) / dx**2


        H = D + V_diag
        H = H.astype(np.complex128)

        I = diags([np.ones(N)], [0], format="csc")
        A = (I + 1j * dt / (2 * 1) * H).tocsc()
        B = (I - 1j * dt / (2 * 1) * H).tocsc()
        A_lu = splu(A)

        psi = psi0.clone().cpu().numpy()
        psi_list = [psi0.clone().cpu().numpy()]
        for _ in range(steps):
            rhs = B @ psi[1:-1]
            psi_interior_new = A_lu.solve(rhs)
            psi[1:-1] = psi_interior_new
            psi[0] = 0
            psi[-1] = 0
            psi_list.append(psi.copy())

        # Convert back to torch tensor
        psi_all = torch.stack([torch.from_numpy(psi) for psi in psi_list], dim=0).to(device)


        #save items for backward in context
        ctx.save_for_backward(t, psi_all)
        ctx.potential_net = potential

        ctx.A_lu = A_lu
        ctx.B = B
        ctx.x = x
        ctx.dt = dt
        ctx.device=device
        ctx.type = psi_type
        return psi_all


    @staticmethod
    def backward(ctx, grad_output):
        #unpack context
        potential = ctx.potential_net #this is the neural network
        t, psi_all = ctx.saved_tensors

        psi_type=ctx.type
        device=ctx.device
        A_lu = ctx.A_lu
        B = ctx.B
        x = ctx.x
        dt = ctx.dt
        #grad output is the automatically computed gradient wrt loss of the tensors returned in forward
        a = grad_output.clone().detach()
        #dL/dpsi1 is the last element
        #terminal adjoint condition a(t_1)
        a_n = a[-1]
        norm = a_n.norm()
        a_n = (a_n/norm).clone().cpu().numpy()

        #initialize accumulating dL/dtheta
        U_n = [torch.zeros_like(p) for p in potential.parameters()]
        with torch.enable_grad():
            #loop backward in time
            for n in range(len(t) - 1, 0, -1):
                with torch.no_grad():
                    #crank nicolson integration for adjoint equation
                    a_nm1_interior = A_lu.solve(B @ a_n[1:-1])
                    a_nm1 = np.zeros_like(a_n)
                    a_nm1[1:-1] = a_nm1_interior
                    a_nm1[0] = 0
                    a_nm1[-1] = 0
                ##loss calculation and accumulations
                #get psi at time n
                a_n = torch.tensor(a_n).to(device)
                psi_n = psi_all[n].detach()
                #evaluate potential over space
                #compute dU/dt at time n
                inner = torch.sum(torch.conj(a_n[1:-1]) * potential(x) * psi_n[1:-1])
                grad_contrib = torch.real(1j*inner)
                #autograd.grad to calculate gradient of grad_contrib wrt network params
                grads_n = torch.autograd.grad(grad_contrib, potential.parameters(), retain_graph=True, allow_unused=True)

                if n < len(t) - 1:
                    for i, (g_n, g_nm1) in enumerate(zip(grads_n, grads_nm1)):
                        if g_n is not None and g_nm1 is not None:
                            U_n[i] += 0.5 * dt * (g_n + g_nm1) #trapezoid rule update
                grads_nm1 = grads_n
                ##

                a_n = a_nm1

        U_n = tuple(g * norm for g in U_n)
        #return gradients for optimizer.step()
        return (None, None, *U_n, None)

def odeint_adjoint_cn(y0, t, potential_net):
    #pass in the network parameters along with network
    return OdeintAdjointCN.apply(y0, t, *potential_net.parameters(), potential_net)
