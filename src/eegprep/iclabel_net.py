"""ICLabel neural network model for EEG artifact classification.

This module provides PyTorch implementations of the ICLabel neural network for
classifying EEG components as brain or artifact sources.
"""

import scipy.io
import torch
import scipy
import numpy as np

class Reshape(torch.nn.Module):
    """Custom reshape layer for PyTorch neural networks."""
    
    def __init__(self, shape):
        """Initialize reshape layer.

        Parameters
        ----------
        shape : tuple
            Target shape for reshaping.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """Forward pass for reshaping.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reshaped tensor.
        """
        return x.view(x.shape[0], *self.shape)

class Concatenate(torch.nn.Module):
    """Custom concatenation layer for PyTorch neural networks."""
    
    def __init__(self, dim):
        """Initialize concatenation layer.

        Parameters
        ----------
        dim : int
            Dimension along which to concatenate.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: list):
        """Forward pass for concatenation.

        Parameters
        ----------
        x : list
            List of tensors to concatenate.

        Returns
        -------
        torch.Tensor
            Concatenated tensor.
        """
        return torch.cat(x, dim=self.dim)


class ICLabelNet(torch.nn.Module):
    """ICLabel neural network for EEG component classification."""
    
    def __init__(self, mat_path):
        """Initialize ICLabel network from MATLAB weights.

        Parameters
        ----------
        mat_path : str
            Path to MATLAB .mat file containing network weights.
        """
        super().__init__()
        iclabel_matlab = scipy.io.loadmat(mat_path)
        params = iclabel_matlab['params'][0]
        # i = 11
        # print('shape of param', i, torch.tensor(params[i][1]).shape)
        self.discriminator_image_layer1_conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, dilation=1)
        # print(self.discriminator_image_layer1_conv.weight.shape)
        self.discriminator_image_layer1_conv.weight = torch.nn.Parameter(torch.tensor(params[0][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_image_layer1_conv.bias = torch.nn.Parameter(torch.tensor(params[1][1], dtype=torch.float32).squeeze())
        self.discriminator_image_layer1_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_image_layer2_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, dilation=1)
        self.discriminator_image_layer2_conv.weight = torch.nn.Parameter(torch.tensor(params[2][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_image_layer2_conv.bias = torch.nn.Parameter(torch.tensor(params[3][1], dtype=torch.float32).squeeze())
        self.discriminator_image_layer2_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_image_layer3_conv = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, dilation=1)
        self.discriminator_image_layer3_conv.weight = torch.nn.Parameter(torch.tensor(params[4][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_image_layer3_conv.bias = torch.nn.Parameter(torch.tensor(params[5][1], dtype=torch.float32).squeeze())
        self.discriminator_image_layer3_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_layer1_conv_conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_psdmed_layer1_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[6][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_psdmed_layer1_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[7][1], dtype=torch.float32).squeeze())
        self.discriminator_psdmed_layer1_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_layer2_conv_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_psdmed_layer2_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[8][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_psdmed_layer2_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[9][1], dtype=torch.float32).squeeze())
        self.discriminator_psdmed_layer2_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_layer3_conv_conv = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_psdmed_layer3_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[10][1], dtype=torch.float32).unsqueeze(3).permute(3, 2, 0, 1))
        self.discriminator_psdmed_layer3_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[11][1], dtype=torch.float32).squeeze(1))
        self.discriminator_psdmed_layer3_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_autocorr_layer1_conv_conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_autocorr_layer1_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[12][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_autocorr_layer1_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[13][1], dtype=torch.float32).squeeze())
        self.discriminator_autocorr_layer1_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_autocorr_layer2_conv_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_autocorr_layer2_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[14][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_autocorr_layer2_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[15][1], dtype=torch.float32).squeeze())
        self.discriminator_autocorr_layer2_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_autocorr_layer3_conv_conv = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_autocorr_layer3_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[16][1], dtype=torch.float32).unsqueeze(3).permute(3, 2, 0, 1))
        self.discriminator_autocorr_layer3_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[17][1], dtype=torch.float32).squeeze(1))
        self.discriminator_autocorr_layer3_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_reshape = Reshape((100, 1, 1))
        self.discriminator_psdmed_concat1 = Concatenate(dim=2)
        self.discriminator_psdmed_concat2 = Concatenate(dim=3)
        self.discriminator_autocorr_reshape = Reshape((100, 1, 1))
        self.discriminator_autocorr_concat1 = Concatenate(dim=2)
        self.discriminator_autocorr_concat2 = Concatenate(dim=3)
        self.discriminator_concat = Concatenate(dim=1)
        self.discriminator_conv = torch.nn.Conv2d(in_channels=712, out_channels=7, kernel_size=4, stride=1, padding=0, dilation=1)
        self.discriminator_conv.weight = torch.nn.Parameter(torch.tensor(params[18][1]).permute(3, 2, 0, 1))
        self.discriminator_conv.bias = torch.nn.Parameter(torch.tensor(params[19][1]).squeeze())
        self.discriminator_softmax = torch.nn.Softmax(dim=1)

    def forward(self, image, psdmed, autocorr):
        """Forward pass through the ICLabel network.

        Parameters
        ----------
        image : torch.Tensor
            Topographic image input.
        psdmed : torch.Tensor
            Power spectral density input.
        autocorr : torch.Tensor
            Autocorrelation input.

        Returns
        -------
        torch.Tensor
            Classification probabilities for each component type.
        """
        super().__init__()
        iclabel_matlab = scipy.io.loadmat(mat_path)
        params = iclabel_matlab['params'][0]
        # i = 11
        # print('shape of param', i, torch.tensor(params[i][1]).shape)
        self.discriminator_image_layer1_conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, dilation=1)
        # print(self.discriminator_image_layer1_conv.weight.shape)
        self.discriminator_image_layer1_conv.weight = torch.nn.Parameter(torch.tensor(params[0][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_image_layer1_conv.bias = torch.nn.Parameter(torch.tensor(params[1][1], dtype=torch.float32).squeeze())
        self.discriminator_image_layer1_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_image_layer2_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, dilation=1)
        self.discriminator_image_layer2_conv.weight = torch.nn.Parameter(torch.tensor(params[2][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_image_layer2_conv.bias = torch.nn.Parameter(torch.tensor(params[3][1], dtype=torch.float32).squeeze())
        self.discriminator_image_layer2_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_image_layer3_conv = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, dilation=1)
        self.discriminator_image_layer3_conv.weight = torch.nn.Parameter(torch.tensor(params[4][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_image_layer3_conv.bias = torch.nn.Parameter(torch.tensor(params[5][1], dtype=torch.float32).squeeze())
        self.discriminator_image_layer3_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_layer1_conv_conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_psdmed_layer1_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[6][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_psdmed_layer1_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[7][1], dtype=torch.float32).squeeze())
        self.discriminator_psdmed_layer1_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_layer2_conv_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_psdmed_layer2_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[8][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_psdmed_layer2_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[9][1], dtype=torch.float32).squeeze())
        self.discriminator_psdmed_layer2_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_layer3_conv_conv = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_psdmed_layer3_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[10][1], dtype=torch.float32).unsqueeze(3).permute(3, 2, 0, 1))
        self.discriminator_psdmed_layer3_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[11][1], dtype=torch.float32).squeeze(1))
        self.discriminator_psdmed_layer3_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_autocorr_layer1_conv_conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_autocorr_layer1_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[12][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_autocorr_layer1_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[13][1], dtype=torch.float32).squeeze())
        self.discriminator_autocorr_layer1_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_autocorr_layer2_conv_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_autocorr_layer2_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[14][1], dtype=torch.float32).permute(3, 2, 0, 1))
        self.discriminator_autocorr_layer2_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[15][1], dtype=torch.float32).squeeze())
        self.discriminator_autocorr_layer2_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_autocorr_layer3_conv_conv = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1), dilation=1)
        self.discriminator_autocorr_layer3_conv_conv.weight = torch.nn.Parameter(torch.tensor(params[16][1], dtype=torch.float32).unsqueeze(3).permute(3, 2, 0, 1))
        self.discriminator_autocorr_layer3_conv_conv.bias = torch.nn.Parameter(torch.tensor(params[17][1], dtype=torch.float32).squeeze(1))
        self.discriminator_autocorr_layer3_conv_relu = torch.nn.LeakyReLU(0.2)
        self.discriminator_psdmed_reshape = Reshape((100, 1, 1))
        self.discriminator_psdmed_concat1 = Concatenate(dim=2)
        self.discriminator_psdmed_concat2 = Concatenate(dim=3)
        self.discriminator_autocorr_reshape = Reshape((100, 1, 1))
        self.discriminator_autocorr_concat1 = Concatenate(dim=2)
        self.discriminator_autocorr_concat2 = Concatenate(dim=3)
        self.discriminator_concat = Concatenate(dim=1)
        self.discriminator_conv = torch.nn.Conv2d(in_channels=712, out_channels=7, kernel_size=4, stride=1, padding=0, dilation=1)
        self.discriminator_conv.weight = torch.nn.Parameter(torch.tensor(params[18][1]).permute(3, 2, 0, 1))
        self.discriminator_conv.bias = torch.nn.Parameter(torch.tensor(params[19][1]).squeeze())
        self.discriminator_softmax = torch.nn.Softmax(dim=1)

    def forward(self, image, psdmed, autocorr):
        """Forward pass through the ICLabelNet model.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        psdmed : torch.Tensor
            PSD median tensor.
        autocorr : torch.Tensor
            Autocorrelation tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after softmax.
        """
        x_image = self.discriminator_image_layer1_conv(image)
        x_image = self.discriminator_image_layer1_relu(x_image)
        x_image = self.discriminator_image_layer2_conv(x_image)
        x_image = self.discriminator_image_layer2_relu(x_image)
        x_image = self.discriminator_image_layer3_conv(x_image)
        x_image = self.discriminator_image_layer3_relu(x_image)
        # print('x_image', x_image.shape)

        x_psdmed = self.discriminator_psdmed_layer1_conv_conv(psdmed)
        x_psdmed = self.discriminator_psdmed_layer1_conv_relu(x_psdmed)
        x_psdmed = self.discriminator_psdmed_layer2_conv_conv(x_psdmed)
        x_psdmed = self.discriminator_psdmed_layer2_conv_relu(x_psdmed)
        x_psdmed = self.discriminator_psdmed_layer3_conv_conv(x_psdmed)
        x_psdmed = self.discriminator_psdmed_layer3_conv_relu(x_psdmed)
        x_psdmed = self.discriminator_psdmed_reshape(x_psdmed)
        x_psdmed = self.discriminator_psdmed_concat1([x_psdmed]*4)
        x_psdmed = self.discriminator_psdmed_concat2([x_psdmed]*4)
        # print('x_psdmed', x_psdmed.shape)

        x_autocorr = self.discriminator_autocorr_layer1_conv_conv(autocorr)
        x_autocorr = self.discriminator_autocorr_layer1_conv_relu(x_autocorr)
        x_autocorr = self.discriminator_autocorr_layer2_conv_conv(x_autocorr)
        x_autocorr = self.discriminator_autocorr_layer2_conv_relu(x_autocorr)
        x_autocorr = self.discriminator_autocorr_layer3_conv_conv(x_autocorr)
        x_autocorr = self.discriminator_autocorr_layer3_conv_relu(x_autocorr)
        x_autocorr = self.discriminator_autocorr_reshape(x_autocorr)
        x_autocorr = self.discriminator_autocorr_concat1([x_autocorr]*4)
        x_autocorr = self.discriminator_autocorr_concat2([x_autocorr]*4)
        # print('x_autocorr', x_autocorr.shape)

        x = self.discriminator_concat([x_image, x_psdmed, x_autocorr])
        x = self.discriminator_conv(x)
        # print('x', x.shape)
        # subtract max value to avoid overflow
        x = x - torch.max(x, dim=1, keepdim=True).values
        x = self.discriminator_softmax(x)
        
        return x
    
# if __name__ == "__main__":
#     model = ICLabelNet('netICL.mat')
#     image_mat = scipy.io.loadmat('net_vars.mat')['in_image']
#     psdmed_mat = scipy.io.loadmat('net_vars.mat')['in_psdmed']
#     autocorr_mat = scipy.io.loadmat('net_vars.mat')['in_autocorr']
#     # assuming third dimension is trivial and last dimension is channel. First two dimensions (32 x 32) are size of topoplot
#     image = torch.tensor(image_mat).permute(-1, 2, 0, 1)
#     print('image shape', image.shape)
#     psdmed = torch.tensor(psdmed_mat).permute(-1, 2, 0, 1)
#     print('psd shape', psdmed.shape)
#     autocorr = torch.tensor(autocorr_mat).permute(-1, 2, 0, 1)
#     print('autocorr shape', autocorr.shape)
#     output = model(image, psdmed, autocorr)
#     print(output.shape)
    
#     # save the output to a mat file
#     scipy.io.savemat('output4.mat', {'output': output.detach().numpy()})
