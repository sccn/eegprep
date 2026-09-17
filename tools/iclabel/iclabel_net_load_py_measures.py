"""MATLAB-parity harness: run the packaged ICLabelNet on reformatted features.

This is a development/parity tool, not part of the installed ``eegprep``
package. It is invoked by the MATLAB parity scripts under ``tests/matlab/``
via ``system('... iclabel_net_load_py_measures.py')``. It reuses the single
canonical :class:`~eegprep.plugins.ICLabel.iclabel_net.ICLabelNet` definition
so the parity harness exercises the same network as production ICLabel.

``netICL.mat`` is a repo dev asset, not a packaged runtime resource (the
wheel ships ``iclabel.onnx`` instead), so it is resolved from the source
tree rather than through ``importlib.resources``.
"""

import logging
from pathlib import Path

import scipy.io
import torch

from eegprep.plugins.ICLabel.iclabel_net import ICLabelNet

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    net_path = REPO_ROOT / "src" / "eegprep" / "plugins" / "ICLabel" / "netICL.mat"
    model = ICLabelNet(str(net_path))
    data = scipy.io.loadmat('python_temp_reformated.mat')
    image_mat = data['grid'][0][0]
    psdmed_mat = data['grid'][0][1]
    autocorr_mat = data['grid'][0][2]
    # assuming third dimension is trivial and last dimension is channel. First two dimensions (32 x 32) are size of topoplot
    image = torch.tensor(image_mat).permute(-1, 2, 0, 1)
    logger.debug("image shape: %s", image.shape)
    psdmed = torch.tensor(psdmed_mat).permute(-1, 2, 0, 1)
    logger.debug("psd shape: %s", psdmed.shape)
    autocorr = torch.tensor(autocorr_mat).permute(-1, 2, 0, 1)
    logger.debug("autocorr shape: %s", autocorr.shape)
    output = model(image, psdmed, autocorr)
    logger.debug("output shape: %s", output.shape)

    # save the output to a mat file
    scipy.io.savemat('output4_py.mat', {'output': output.detach().numpy()})
