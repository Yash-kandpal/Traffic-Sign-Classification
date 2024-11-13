import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

# Define the class names for each index (replace these with actual class names)
class_names = [
"Speed limit (20km/h)",
"Speed limit (30km/h)",
"Speed limit (50km/h)",
"Speed limit (60km/h)",
"Speed limit (70km/h)",
"Speed limit (80km/h)",
"End of speed limit (80km/h)",
"Speed limit (100km/h)",
"Speed limit (120km/h)",
"No passing",
"No passing for vehicles over 3.5 metric tons",
"Right-of-way at the next intersection",
"Priority road",
"Yield",
"Stop",
"No vehicles",
"Vehicles over 3.5 metric tons prohibited",
"No entry",
"General caution",
"Dangerous curve to the left",
"Dangerous curve to the right",
"Double curve",
"Bumpy road",
"Slippery road",
"Road narrows on the right",
"Road work",
"Traffic signals",
"Pedestrians",
"Children crossing",
"Bicycles crossing",
"Beware of ice/snow",
"Wild animals crossing",
"End of all speed and passing limits",
"Turn right ahead",
"Turn left ahead",
"Ahead only",
"Go straight or right",
"Go straight or left",
"Keep right",
"Keep left",
"Roundabout mandatory",
"End of no passing",
"End of no passing by vehicles over 3.5 metric tons"

]


def preprocess_image(image_path):
    """
    Preprocess the image to match the model input requirements.
    """
    img = load_img(image_path, target_size=(32, 32))  # Resize image to 32x32
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def classify_image(image_path):
    """
    Classify an image using the trained model.
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Display the image and the prediction result
    plt.imshow(load_img(image_path))
    plt.axis('off')
    plt.title(f"Predicted Class: {class_names[predicted_class]}, Confidence: {confidence:.2f}")
    plt.show()


# Ask user to input the path of an image file
image_path = "C:/Users/yashk/Downloads/30speedlimit.jpeg"
classify_image(image_path)
