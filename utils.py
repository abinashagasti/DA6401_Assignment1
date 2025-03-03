import numpy as np
import matplotlib.pyplot as plt

def plot_sample_images(x, y):
    indices = [np.random.choice(np.where(y == c)[0]) for c in range(10)]
    # print(indices)

    m = x.shape[0]

    fig, axes = plt.subplots(2, 5, figsize=(10, 5)) 
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        axes[i].imshow(x[idx], cmap="gray")  # Display image
        axes[i].set_title(f"Class {i}")  # Set title
        axes[i].axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()