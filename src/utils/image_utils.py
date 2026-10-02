from pathlib import Path

import matplotlib.pyplot as plt
import torch


def normalize_image_tensor(images):
    """
    Convert image tensors from [-1, 1] to [0, 1].

    Parameters
    ----------
    images : torch.Tensor
        Image tensor.

    Returns
    -------
    torch.Tensor
        Tensor clipped to the [0, 1] range.
    """
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def save_image_grid(images, output_path, nrow=4):
    """
    Save a batch of images as a grid.

    Parameters
    ----------
    images : torch.Tensor
        Image tensor with shape [N, C, H, W].
    output_path : str or Path
        Destination image path.
    nrow : int
        Number of images per row.

    Returns
    -------
    Path
        Path to the saved image.
    """
    images = normalize_image_tensor(images).detach().cpu()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = images.shape[0]
    ncols = min(nrow, count)
    nrows = (count + ncols - 1) // ncols

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * 3),
        squeeze=False,
    )

    axes = axes.flatten()

    for index in range(count):
        image = images[index]

        if image.shape[0] == 1:
            axes[index].imshow(image.squeeze(0), cmap="gray")
        else:
            axes[index].imshow(image.permute(1, 2, 0))

        axes[index].axis("off")

    for index in range(count, len(axes)):
        axes[index].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return output_path
