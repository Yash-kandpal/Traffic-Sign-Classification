from Loading_dataset import train_images, train_labels
import numpy as np
import matplotlib.pyplot as plt

# Assuming there are 43 classes in total (from 0 to 42)
num_classes = 43

# Prepare a figure to display one random image per class
plt.figure(figsize=(15, 10))

for class_id in range(num_classes):
    # Find all indices where the label matches the current class_id
    class_indices = np.where(train_labels == class_id)[0]

    if len(class_indices) > 0:
        # Select a random index for this class
        idx = np.random.choice(class_indices)

        # Plot the image
        plt.subplot(7, 7, class_id + 1)  # Adjust layout based on num_classes (7x7 grid for 43 classes)
        plt.imshow(train_images[idx])
        plt.title(f"Class {class_id}")
        plt.axis('off')

plt.tight_layout()
plt.show()
