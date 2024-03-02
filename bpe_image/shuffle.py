import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
(mnist_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Select the first image for demonstration and normalize it
image = mnist_images[0].astype("float32") / 255.0


# Function to create distributed groupings
def create_distributed_groupings(image, gap_size=4, group_size=4):
    # Flatten the image for easier sampling
    flat_image = image.flatten()
    # Determine the step size based on the gap and group size
    step_size = (
        gap_size + 1
    )  # Assuming gap_size accounts for 'gap_size' pixels between the samples
    # Calculate the number of groups we can form
    total_pixels = len(flat_image)
    num_groups = total_pixels // (group_size * step_size)

    groupings = []
    for i in range(num_groups):
        group = flat_image[
            i * step_size : i * step_size + group_size * step_size : step_size
        ]
        # Reshape the group back to 2D if necessary or leave as 1D based on your preference
        groupings.append(group.reshape((2, 2)))  # Reshaping to 2x2 for visualization

    return groupings


# Create distributed groupings
groupings = create_distributed_groupings(image)


# Function to plot images in a grid
def plot_groupings(groupings, figure_size=(15, 7)):
    plt.figure(figsize=figure_size)
    for i, group in enumerate(groupings[:16]):  # Display first 16 groupings
        plt.subplot(4, 4, i + 1)
        plt.imshow(group, cmap="gray", interpolation="none")
        plt.title(f"Group {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Visualizing the first 16 distributed groupings
plot_groupings(groupings)
