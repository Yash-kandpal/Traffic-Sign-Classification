import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from Loading_dataset import train_images, train_labels, valdn_images, valdn_labels,test_images,test_labels

num_classes = 43
train_labels = to_categorical(train_labels, num_classes)
valdn_labels = to_categorical(valdn_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)
# Load the pretrained VGG16 model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling layer to reduce the dimensions
x = Dense(128, activation='relu')(x)  # Dense layer with ReLU activation
x = Dropout(0.5)(x)  # Dropout for regularization

# Output layer for classification (for 43 classes)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the full model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model with an initial learning rate for frozen layers
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks for early stopping and model checkpointing
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Initial training with frozen layers
history = model.fit(
    train_images, train_labels,
    epochs=10, batch_size=32,
    validation_data=(valdn_images, valdn_labels),
    callbacks=[checkpoint, early_stopping]
)

# Unfreeze some of the last layers of the base model for fine-tuning
for layer in base_model.layers[-10:]:  # Adjust the number of layers to unfreeze as needed
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model with unfrozen layers
history_fine_tune = model.fit(
    train_images, train_labels,
    epochs=10, batch_size=32,
    validation_data=(valdn_images, valdn_labels),
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy:.2f}")