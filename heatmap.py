import tensorflow as tf
import matplotlib.pyplot as plt

def overlay_density_on_image(image, density_map, fname, alpha=0.4, cmap='jet'):
    """
    Overlay a density map onto an image.

    Args:
        image: Tensor of shape (256, 256, 1) or (256, 256, 3)
        density_map: Tensor of shape (256, 256, 1)
        alpha: Transparency of the density map (0 = only image, 1 = only density)
        cmap: Colormap to use for the density map
    """
    # Convert tensors to numpy
    image = image.numpy().squeeze()
    density_map = density_map.numpy().squeeze()

    # Normalize image to 0-1
    image = (image - image.min()) / (image.max() - image.min())

    # Normalize density map to 0-1
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())

    # Plot
    plt.imshow(image, cmap='gray')
    plt.imshow(density_map, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.savefig(f'images/{fname}.png', dpi=300, bbox_inches='tight')
