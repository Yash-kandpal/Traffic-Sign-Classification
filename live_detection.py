import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('best_model.keras')

# Define class labels (replace these with actual class names if available)
class_labels = [f"Class {i}" for i in range(43)]  # Customize with actual names if you have them

# Set input image size based on model requirements
input_size = (32, 32)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize the RGB image to match model's input size
    img_rgb = cv2.resize(img_rgb, input_size)

    # Normalize the image
    img_rgb = img_rgb / 255.0

    # Reshape for model input: (1, 32, 32, 3)
    img_rgb = img_rgb.reshape(1, input_size[0], input_size[1], 3)

    # Make a prediction
    predictions = model.predict(img_rgb)
    class_index = np.argmax(predictions)
    class_name = class_labels[class_index]

    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow("Traffic Sign Classification", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
