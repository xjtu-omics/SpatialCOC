from sklearn.exceptions import NotFittedError
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from abc import abstractmethod
from sklearn.base import BaseEstimator
from Model.utils import check_Xs
import torch.optim as optim
from tqdm import tqdm

class BaseEmbed(BaseEstimator):
    """
    A base class for embedding multi-modality data.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xs, y=None):
        """
        A method to fit model to multi-modality data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs shape: (n_modalities,)
            - Xs[i] shape: (n_samples, n_features_i)

        y : array, shape (n_samples,), optional

        Returns
        -------
        self: returns an instance of self.
        """

        return self

    @abstractmethod
    def transform(self, Xs):
        """
        Transform data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_modalities
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        Xs_transformed : list of numpy.ndarray
            - length: n_modalities
            - Xs_transformed[i] shape: (n_samples, n_components_i)
        """

        pass

    def fit_transform(self, Xs, y=None):
        """
        Fit an embedder to the data and transform the data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_modalities
            - Xs[i] shape: (n_samples, n_features_i)

        y : array, shape (n_samples,), optional
            Targets to be used if fitting the algorithm is supervised.

        Returns
        -------
        X_transformed : list of numpy.ndarray
            - X_transformed length: n_modalities
            - X_transformed[i] shape: (n_samples, n_components_i)
        """
        return self.fit(Xs=Xs, y=y).transform(Xs=Xs)


class Encoder(nn.Module):
    """
    Parameters
    ----------
    x : torch.Tensor
        Input data.

    Returns
    -------
    latent : torch.Tensor
        The representation of the input data in the latent space.
    x : torch.Tensor
        The output of the second fully connected layer.
    """
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_size)
        self.leaky_relu = nn.LeakyReLU(1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        latent = self.fc3(x)
        return latent, x


class Decoder(nn.Module):
    """
    Forward pass through the decoder network.

    Parameters
    ----------
    x : torch.Tensor
        Input data from the latent space.

    Returns
    -------
    output : torch.Tensor
        The reconstructed input data.
    """
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.leaky_relu = nn.LeakyReLU(1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        output = self.fc3(x)
        return output


class EncoderDecoderNet(nn.Module):
    """
    Combines an encoder and a decoder for learning representations.

    Attributes
    ----------
    encoder : Encoder
        The encoder network.
    decoder : Decoder
        The decoder network.
    """
    def __init__(self, input_size, latent_size):
        super(EncoderDecoderNet, self).__init__()
        self.encoder = Encoder(input_size, latent_size)
        self.decoder = Decoder(latent_size, input_size)

    def forward(self, x):
        latent, mid = self.encoder(x)
        output = self.decoder(latent)
        return output, latent, mid


class linear_cca():
    """
    Applies a linear transformation to the latent representation of the model output to further extract relevant features between modalities.

    Attributes
    ----------
    w_ : list
        The linear transformation matrices for each modality.
    m_ : list
        The mean vectors for each modality.
    """
    
    def __init__(self):
        self.w_ = [None, None]
        self.m_ = [None, None]

    def fit(self, H1, H2, n_components):
        """
        Fits the linear CCA model to the given data.

        Parameters
        ----------
        H1 : numpy.ndarray
            The latent representation of the first modality.
        H2 : numpy.ndarray
            The latent representation of the second modality.
        n_components : int
            The number of components to retain.
        """
        r1 = 1e-4
        r2 = 1e-4

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m_[0] = np.mean(H1, axis=0)
        self.m_[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m_[0], (m, 1))
        H2bar = H2 - np.tile(self.m_[1], (m, 1))

        # Compute covariance matrices
        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T,
                                              H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T,
                                              H2bar) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(
            np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(
            np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv,
                             SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w_[0] = np.dot(SigmaHat11RootInv, U[:, 0:n_components])
        self.w_[1] = np.dot(SigmaHat22RootInv, V[:, 0:n_components])
        D = D[0:n_components]

    def _get_result(self, x, idx):
        """
        Transform the data from a single modality using an already fitted linear transformation matrix (w_ calculated by the fit method)

        Parameters
        ----------
        x : nd-array, shape (n_samples, n_features)
        idx : int, 0 if modality 1. 1 if modality 2.

        Returns
        -------
        result : nd-array
        The result of the linear transformation of x.
        """
        result = x - self.m_[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w_[idx])
        return result

    def transform(self, H1, H2):
        return [self._get_result(H1, 0), self._get_result(H2, 1)]

class cca_loss():
    """
    Helps the model learn feature representations that capture the correlation between two different modalities of the data.

    Attributes
    ----------
    n_components_ : int
        The number of components to retain.
    use_all_singular_values_ : bool
        Whether to use all singular values or just the top n_components_.
    device_ : torch.device
        The device to use for computations.
    """
    def __init__(self, n_components, use_all_singular_values, device):
        self.n_components_ = n_components
        self.use_all_singular_values_ = use_all_singular_values
        self.device_ = device

    def loss(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        # Transpose matrices so each column is a sample
        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Compute covariance matrices and add diagonal so they are
        # positive definite
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + \
                     r1 * torch.eye(o1, device=self.device_)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + \
                     r2 * torch.eye(o2, device=self.device_)

        # Calculate the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO='L')
        [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO='L')

        # Additional code to increase numerical stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        # Compute sigma hat matrices using the edited covariance matrices
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        # Compute the T matrix, whose matrix trace norm is the loss
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values_:
            # all singular values are used to calculate the correlation (and
            # thus the loss as well)
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            corr = torch.sqrt(tmp)
        else:
            # just the top self.n_components_ singular values are used to
            # compute the loss
            # U, V = torch.symeig(torch.matmul(Tval.t(), Tval), eigenvectors=True)
            U, V = torch.linalg.eigh(torch.matmul(Tval.t(), Tval), UPLO='U')
            U = U.topk(self.n_components_)[0]
            corr = torch.sum(torch.sqrt(U))
        return self.n_components_-corr


class DeepPairedNetworks(nn.Module):
    """
    A deep neural network that learns paired representations from two modalities.

    Attributes
    ----------
    model1_ : EncoderDecoderNet
        The encoder-decoder network for the first modality.
    model2_ : EncoderDecoderNet
        The encoder-decoder network for the second modality.
    loss_ : callable
    The loss function to use.
    reloss : float
    The reconstruction loss.
    """
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2,
                 n_components, use_all_singular_values,
                 device=torch.device('cpu')):
        super(DeepPairedNetworks, self).__init__()
        self.model1_ = EncoderDecoderNet(input_size1, layer_sizes1[-1]).double()
        self.model2_ = EncoderDecoderNet(input_size2, layer_sizes2[-1]).double()
        self.loss_ = cca_loss(n_components, use_all_singular_values, device).loss
        self.reloss = 0

    def forward(self, x1, x2):
        out1, latent1, mid1 = self.model1_(x1)
        out2, latent2, mid2 = self.model2_(x2)
        ## 计算自重建损失
        reconstruction_loss_fn = nn.MSELoss()
        reconstruction_loss1 = reconstruction_loss_fn(out1, x1)
        reconstruction_loss2 = reconstruction_loss_fn(out2, x2)
        self.reloss = 0.5*(reconstruction_loss1 + reconstruction_loss2)

        return latent1, latent2

    def joint_loss(self, latent1, latent2):
        cca_loss = self.loss_(latent1, latent2)
        total_loss = self.reloss + cca_loss
        return total_loss


class DCCAE(BaseEmbed):
    """
    deep canonically correlated auto-encoder (DCCAE) class.

    This class implements the DCCAE algorithm, which learns representations that capture the correlation between two modalities of the data.

    Attributes
    ----------
    input_size1_ : int
        The number of features in the first modality.
    input_size2_ : int
        The number of features in the second modality.
    n_components_ : int
        The number of components to retain.
    use_all_singular_values : bool
        Whether to use all singular values or just the top n_components_.
    device_ : torch.device
        The device to use for computations.
    epoch_num : int
        The number of epochs to train for.
    batch_size_ : int
        The batch size to use during training.
    learning_rate_ : float
        The learning rate for the optimizer.
    reg_par_ : float
        The regularization parameter.
    print_train_log_info : bool
        Whether to print training log information.
    tolerance : float
        The tolerance for convergence.
    deep_model_ : DeepPairedNetworks
        The deep paired networks model.
    linear_cca_ : linear_cca
        The linear CCA model.
    model_ : nn.DataParallel
        The model wrapped in a data parallel module.
    optimizer_ : torch.optim.Optimizer
        The optimizer to use during training.
    is_fit : bool
        Whether the model has been fit to data.
    """
    def __init__(
            self, input_size1=None, input_size2=None, n_components=2,
            layer_sizes1=None, layer_sizes2=None,
            use_all_singular_values=False, device=torch.device('cpu'),
            epoch_num=200, batch_size=800, learning_rate=1e-3, reg_par=1e-5,
            tolerance=1e-5, print_train_log_info=True
    ):

        super().__init__()

        if layer_sizes1 is None:
            layer_sizes1 = [1000, n_components]
        if layer_sizes2 is None:
            layer_sizes2 = [1000, n_components]

        self._valid_inputs(input_size1, input_size2, n_components,
                           layer_sizes1, layer_sizes2,
                           use_all_singular_values, device,
                           epoch_num, batch_size, learning_rate, reg_par,
                           tolerance, print_train_log_info)

        self.input_size1_ = input_size1
        self.input_size2_ = input_size2
        self.n_components_ = n_components

        self.use_all_singular_values = use_all_singular_values
        self.device_ = device
        self.epoch_num = epoch_num
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate

        self.reg_par_ = reg_par
        self.print_train_log_info = print_train_log_info
        self.tolerance = tolerance

        self.deep_model_ = DeepPairedNetworks(layer_sizes1, layer_sizes2,
                                              input_size1, input_size2,
                                              n_components,
                                              use_all_singular_values,
                                              device=device)
        self.linear_cca_ = linear_cca()

        self.model_ = nn.DataParallel(self.deep_model_)
        # Move to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ = self.model_.to(device)

        self.model_.to(device)
        # self.loss_ = self.deep_model_.loss_
        self.loss_ = self.deep_model_.joint_loss
        self.optimizer_ = torch.optim.RMSprop(self.model_.parameters(),
                                              lr=self.learning_rate_,
                                              weight_decay=self.reg_par_)
        self.is_fit = False

    def fit(self, Xs, y=None):
        """
        Fits the DCCAE model to the given data.

        Parameters
        ----------
        Xs : list of numpy.ndarray
            The input data for both modalities.
        y : numpy.ndarray, optional
            The target data (not used in this implementation).
        """
        Xs = check_Xs(Xs)  # ensure valid input

        # Check valid shapes based on initialization
        if Xs[0].shape[1] != self.input_size1_:
            raise ValueError('modality 1 input data is incorrect shape based on'
                             ' self.input_size1_. Found {} features but'
                             'expected {}'.format(Xs[0].shape[1],
                                                  self.input_size1_))
        if Xs[1].shape[1] != self.input_size2_:
            raise ValueError('modality 2 input data is incorrect shape based on'
                             ' self.input_size2_. Found {} features but'
                             'expected {}'.format(Xs[1].shape[1],
                                                  self.input_size2_))

        x1 = torch.DoubleTensor(Xs[0])
        x2 = torch.DoubleTensor(Xs[1])
        x1 = x1.to(self.device_)
        x2 = x2.to(self.device_)

        data_size = x1.size(0)
        checkpoint = 'checkpoint.model'
        train_losses = []
        epoch = 0
        current_loss = np.inf
        train_loss = 1

        # Define learning rate scheduler (Cosine Annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_, T_0=10, T_mult=2)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_, T_max=self.epoch_num)
        # Initialize the progress bar
        pbar = tqdm(total=self.epoch_num, desc="Training Progress")

        # Training loop
        for epoch in range(self.epoch_num):
            self.model_.train()  # Set the model to training mode
            batch_idxs = list(BatchSampler(RandomSampler(range(data_size)),
                                        batch_size=self.batch_size_,
                                        drop_last=False))  # Create batch indices
            current_loss = train_loss  # Initialize current loss (if needed)

            train_losses = []  # Initialize list to store losses for the current epoch

            # Iterate over the batch indices
            for batch_idx in batch_idxs:
                self.optimizer_.zero_grad()  # Clear previous gradients
                batch_x1 = x1[batch_idx, :]  # Get the current batch of data from x1
                batch_x2 = x2[batch_idx, :]  # Get the current batch of data from x2

                l1, l2 = self.model_(batch_x1, batch_x2)  # Forward pass through the model
                loss = self.loss_(l1, l2)  # Calculate the loss
                train_losses.append(loss.item())  # Append the loss to the list
                loss.backward()  # Backpropagate the loss
                self.optimizer_.step()  # Update model parameters

            train_loss = np.mean(train_losses)  # Calculate the mean loss for the epoch

            # Update learning rate with scheduler
            scheduler.step()

            # Save model checkpoint
            torch.save(self.model_.state_dict(), checkpoint)

            # Update progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()
        print("model training finished!")
        # Check if converged before max iterations
        if epoch == self.epoch_num:
            message = 'Loss did not converge before {} epochs. Consider' \
                      ' increasing epoch_num to train for' \
                      ' longer.'.format(self.epoch_num)
            warnings.warn(message, Warning)

        # train_linear_cca
        losses, outputs = self._get_outputs(x1, x2)
        self._train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model_.load_state_dict(checkpoint_)

        self.is_fit = True
        return self

    def transform(self, Xs, return_loss=False):
        if not self.is_fit:
            raise NotFittedError("Must call fit function before transform")
        Xs = check_Xs(Xs)
        x1 = torch.DoubleTensor(Xs[0])
        x2 = torch.DoubleTensor(Xs[1])

        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)
            outputs = self.linear_cca_.transform(outputs[0], outputs[1])
            if return_loss:
                return outputs, np.mean(losses)
            return outputs

    def _train_linear_cca(self, x1, x2):
        self.linear_cca_.fit(x1, x2, self.n_components_)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model_.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(range(data_size)),
                                           batch_size=self.batch_size_,
                                           drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model_(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss_(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]

        return losses, outputs

    def _valid_inputs(self, input_size1, input_size2, n_components,
                      layer_sizes1, layer_sizes2,
                      use_all_singular_values, device,
                      epoch_num, batch_size, learning_rate, reg_par,
                      tolerance, print_train_log_info
                      ):
        # Check input_size parameters
        if (not isinstance(input_size1, int)) or \
                (not isinstance(input_size2, int)) or \
                input_size1 < 1 or input_size2 < 1:
            raise ValueError('input_size1 and input_size2 must be'
                             ' positive integers')

        # Check n_components
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError('n_components must be positive integer')

        # Check n_components vs last layer size
        if not (n_components <= layer_sizes1[-1]) and not \
                (n_components <= layer_sizes2[-1]):
            raise ValueError('n_components must be no greater than final'
                             ' layer size. Desired {} components but {}'
                             ' and {} dimensional final layers'
                             ''.format(n_components, layer_sizes1[-1],
                                       layer_sizes2[-1]))

        # Check layer_sizes
        if (isinstance(layer_sizes1, list)) or \
                (isinstance(layer_sizes2, list)):
            for elem in layer_sizes1:
                if not isinstance(elem, int) or elem < 1:
                    raise ValueError('All layer sizes must be positive'
                                     ' integers')
            for elem in layer_sizes2:
                if not isinstance(elem, int) or elem < 1:
                    raise ValueError('All layer sizes must be positive'
                                     ' integers')
        else:
            raise ValueError('layer_sizes1 and layer_sizes2 must be of type'
                             ' list')
        if layer_sizes1[-1] != layer_sizes2[-1]:
            raise ValueError('Output size of deep networks must match. Make'
                             ' sure layer_sizes1[-1] == layer_sizes2[-1]')

        # Check epoch_num
        if not isinstance(epoch_num, int) or epoch_num < 1:
            raise ValueError('epoch_num must be positive integer')

        # Check batch_size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('epoch_num must be positive integer')

        # Check learning_rate
        if learning_rate <= 0:
            raise ValueError('learning_rate must be positive')

        # Check reg_par
        if reg_par <= 0:
            raise ValueError('reg_par must be positive')

        # Check tolerance
        if tolerance <= 0:
            raise ValueError('tolerance must be positive')