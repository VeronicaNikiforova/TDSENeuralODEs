import numpy as np
import torch
import torch.nn as nn
from Crank_nic import odeint_adjoint_cn
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import torch.nn.functional as F
import torchdiffeq


datafile = "wavefunction_dataset_steps50.npz"
seed = 777

num_epochs = 180
steps = 50

learning_rate = 1e-3
momentum = 0.8
top_k = 20
show_psi = 5

hidden = 128
num_freq = 1
sigma = 1

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load dataset
npz = np.load(datafile)
x = torch.tensor(npz["x"], dtype=torch.float32)
psi0s = torch.tensor(npz["psi0s"], dtype=torch.complex64)
psi1s = torch.tensor(npz["psi1s"], dtype=torch.complex64)



x = x.to(device)
# Constants
hbar = 1.0
m = 1.0
dx = (x[1] - x[0])


# fourier features
class LearnedFourierFeatures(nn.Module):
    def __init__(self, num_frequencies, sigma):
        super().__init__()
        B =torch.linspace(1.0, num_frequencies, num_frequencies).reshape(-1, 1) * sigma
        self.B = nn.Parameter(B)  # learnable projection matrix

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_proj = 2 * np.pi * x @ self.B.T
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return features / np.sqrt(self.B.shape[0])

# potential network
class PotentialNN(nn.Module):
    def __init__(self, hidden=hidden, num_freq=num_freq, sigma=sigma):
        super().__init__()
        self.fourier = LearnedFourierFeatures(num_freq, sigma)
        self.real_net = nn.Sequential(
            nn.Linear(1 + 2 * num_freq, hidden),  # input: x + Fourier features
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        fourier_feats = self.fourier(x)
        x_cat = torch.cat([x, fourier_feats], dim=-1)
        return self.real_net(x_cat).squeeze(-1)


# initialize model
potential_net = PotentialNN().to(device)
# optimizer = torch.optim.AdamW(potential_net.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(potential_net.parameters(), lr=learning_rate, momentum=momentum)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training loop
num_epochs = num_epochs
steps = steps
dt = 1.0/steps
t_span = torch.arange(steps, dtype=torch.float32) * dt

psi0s = psi0s.to(device)
psi1s = psi1s.to(device)

loss_over_time = []
all_potentials = []
all_psi0_pred = []
x_bigger= torch.linspace(-1.0, 1.0, 200).to(device)

def fidelity(pred_psi, true_psi):
    return torch.real(1 - (pred_psi.conj() @ true_psi * dx))

for epoch in range(num_epochs):
    total_loss = 0.0
    start = time.time()
    current_losses = [] # Keep track of losses over all wavefunctions
    for psi0, psi1_true in zip(psi0s, psi1s):
        psi_all = odeint_adjoint_cn(psi0, t_span, potential_net)
        pred_psi1 = psi_all[-1]

        loss = fidelity(pred_psi1, psi1_true)
        current_losses.append((loss.item(), psi0, psi1_true))

        if torch.equal(psi0, psi0s[show_psi]):  # for visualization
            all_psi0_pred.append(pred_psi1)


    # only update based on k wavefunctions with highest loss
    current_losses.sort(key=lambda x: x[0], reverse=True)
    hardest_samples = current_losses[:top_k]
    optimizer.zero_grad()
    for _, psi0_hard, psi1_hard in hardest_samples:
        psi_all = odeint_adjoint_cn(psi0_hard, t_span, potential_net)
        pred_psi1 = psi_all[-1]

        loss = fidelity(pred_psi1, psi1_hard)

        loss.backward()
    optimizer.step()

    total_loss += loss.item()

    # Track loss
    V_learned = torch.Tensor.cpu(potential_net(x_bigger)).detach().numpy()
    all_potentials.append(V_learned) # for graphic
    loss_over_time.append(total_loss) # for graphic
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Time taken: {time.time()-start}")




# Animation stuff
# Prepare static data
x_np = x.detach().cpu().numpy()
x_bigger_np = x_bigger.detach().cpu().numpy()
V_true_np = npz['potentials'][0]
psi1_true_np = psi1s[show_psi].detach().cpu().numpy()

# Create figure with 3 subplots (1 row, 3 columns)
fig = plt.figure(figsize=(18, 5))
gs = fig.add_gridspec(1, 3)

# Loss plot (left)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(range(len(loss_over_time)), loss_over_time, label='Loss')
ax0.set_title('Loss over Epochs')
ax0.set_xlabel('Epoch')
ax0.set_ylabel('Loss')
ax0.legend()

# Potential animation (middle)
ax1 = fig.add_subplot(gs[0, 1])
pred_line_V, = ax1.plot([], [], label='Learned V(x)', lw=2)
true_line_V, = ax1.plot(x_np, V_true_np-V_true_np.mean(), 'r--', label='True V(x)', lw=2)
ax1.set_xlim(x_bigger_np[0], x_bigger_np[-1])
v_max = max([max(i) for i in all_potentials])
v_min = min([min(i) for i in all_potentials])
ax1.set_ylim(-4, v_max)
ax1.set_title('Potential V(x)')
ax1.set_xlabel('x')
ax1.legend()

# Wavefunction animation (right)
ax2 = fig.add_subplot(gs[0, 2])
pred_line_psi, = ax2.plot([], [], label='Predicted ψ₁(x)', lw=2)
true_line_psi, = ax2.plot(x_np, psi1_true_np, 'r--', label='True ψ₁(x)', lw=2)
ax2.set_xlim(x_np[0], x_np[-1])
ax2.set_ylim(-1, 1)
ax2.set_title('Final Wavefunction ψ₁(x)')
ax2.set_xlabel('x')
ax2.legend()

# Animation init
def init():
    pred_line_V.set_data([], [])
    pred_line_psi.set_data([], [])
    return pred_line_V, pred_line_psi, true_line_V, true_line_psi

# Frame update
def update(frame):
    V_pred = all_potentials[frame]
    psi_pred = all_psi0_pred[frame].detach().cpu().numpy()
    pred_line_V.set_data(x_bigger_np, V_pred)
    pred_line_psi.set_data(x_np, psi_pred)
    fig.suptitle(f'Epoch {frame}')
    return pred_line_V, pred_line_psi, true_line_V, true_line_psi

# Run animation
ani = animation.FuncAnimation(
    fig, update, frames=len(all_potentials),
    init_func=init, blit=True, interval=100
)

plt.tight_layout()
from matplotlib.animation import PillowWriter
ani.save("training_progress.gif", writer=PillowWriter(fps=10))
plt.show()
