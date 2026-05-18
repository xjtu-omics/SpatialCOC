import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import issparse, lil_matrix
from scipy.stats import zscore

class SineLayer(nn.Module):
    """
    Applies a sine activation function to the input, followed by a linear transformation to represent complex modalities' signals.

    Attributes:
        linear (nn.Linear): A linear transformation layer.
        omega_0 (torch.Tensor): A tensor representing the frequency of the sine function, repeated for each input feature.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 3, out_features)
        self.omega_0 = torch.tensor([[1], [2], [3]], dtype=torch.float32)
        self.omega_0 = self.omega_0.repeat(1, in_features).reshape(1, in_features * 3)

    def forward(self, input):
        input = input.repeat(1, 3)
        input = input * self.omega_0.to(input.device)
        return self.linear(torch.sin(input))

class INR(nn.Module):
    """
    The INR (Intuitive Neural Representation) class is designed to handle input coordinates and produce there construction representation of each modalities.

    Attributes:
        INR (nn.Sequential): A sequential container of PyTorch modules, including
            linear layers and SineLayer instances.
    """
    def __init__(self, out_dim):
        super().__init__()

        mid = 3000
        self.INR = nn.Sequential(nn.Linear(2, mid),
                                SineLayer(mid, mid),
                                SineLayer(mid, mid),
                                SineLayer(mid, out_dim))

    def forward(self, coord):
        return self.INR(coord)

def kernel_build(kerneltype, location, bandwidth):
    """
    Constructs a kernel matrix for the given location data and bandwidth.
    Returns:
    - K (numpy array): The kernel matrix.
    """
    if kerneltype == "gaussian":
        return np.exp(-squareform(pdist(location))**2 / bandwidth)
    raise ValueError(f"Unsupported kernel type: {kerneltype}")

def kernel_build_sparse(kerneltype, location, bandwidth, tol):
    """
    Creates a sparse representation of the kernel matrix.
    """
    K = kernel_build(kerneltype, location, bandwidth)
    K_sparse = lil_matrix(K)
    
    # Set elements below the tolerance to 0
    K_sparse[K_sparse < tol] = 0
    # Convert to csr_matrix for more efficient operations
    K_sparse = K_sparse.tocsr()
        
    return K_sparse

def spatial_build_kernel(location, kerneltype="gaussian", bandwidth_set_by_user=None, sparse_kernel=False, sparse_kernel_tol=1e-20, sparse_kernel_ncore=1):
    """
    Constructs a kernel matrix for spatial data, with options for normalization, sparsity, and bandwidth.
    Returns:
    - K (numpy array or scipy.sparse.lil_matrix): The kernel matrix, either dense or sparse.
    """
    location_normalized = zscore(location)
    bandwidth = bandwidth_set_by_user
    if not sparse_kernel:
        K = kernel_build(kerneltype, location_normalized, bandwidth)
    else:
        K = kernel_build_sparse(kerneltype, location_normalized, bandwidth, sparse_kernel_tol)
    
    return K

def _as_float_tensor(value, device):
    if torch.is_tensor(value):
        return value.float().to(device)
    if issparse(value):
        value = value.toarray()
    value = np.ascontiguousarray(value)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class SCM(nn.Module):
    def __init__(self, X, spatial_coord, device, learning_rate=1e-4, reg_par=1e-4, epoch_num=100, bandwidth=0.002, print_train_log_info=True):
        super().__init__()
        self.device = torch.device(device)
        self.X = _as_float_tensor(X, self.device)
        self.coords = _as_float_tensor(spatial_coord, self.device)
        self.learning_rate_ = learning_rate
        self.reg_par_ = reg_par
        self.epoch_num = epoch_num
        self.bandwidth = bandwidth
        self.print_train_log_info = print_train_log_info
        out_dim = self.X.shape[1]
        self.INR = INR(out_dim=out_dim).to(self.device)
        
        params = list(self.INR.parameters())
        self.optimizer_ = torch.optim.Adamax(params, lr=self.learning_rate_, weight_decay=reg_par)
        
        K = spatial_build_kernel(self.coords.cpu().numpy(), kerneltype="gaussian", bandwidth_set_by_user=self.bandwidth)
        K_centered = K - np.mean(K, axis=0) - np.mean(K, axis=1) + np.mean(K)
        self.K = torch.as_tensor(K_centered, dtype=torch.float32, device=self.device)
        self.KX = self.K @ self.X

    def fit(self):
        self.optimizer_ = torch.optim.Adamax(self.INR.parameters(), lr=self.learning_rate_, weight_decay=self.reg_par_)
        coord = self.coords
        train_losses = []
        epoch = 0
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_, T_0=50, T_mult=2)
        best_loss = float("inf")
        best_SCM_recon = None
        while epoch < self.epoch_num:
            
            SCM_recon = self.INR(coord)
            self.optimizer_.zero_grad()
            loss = torch.norm((SCM_recon - self.KX), p=2)
            loss = torch.sum(loss)
            train_losses.append(loss.item())
            loss.backward()
            self.optimizer_.step()

            scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_SCM_recon = SCM_recon.detach().clone()
            if self.print_train_log_info:
                info_string = "Epoch {:d}/{:d}, training_loss: {:.4f}, best_loss:{:4f}, learning_rate: {:.9f}"
                print('', info_string.format(epoch + 1, self.epoch_num, loss.item(), best_loss, self.optimizer_.param_groups[0]['lr']), end='\r')
            epoch += 1
        
        return best_SCM_recon.cpu().numpy()
