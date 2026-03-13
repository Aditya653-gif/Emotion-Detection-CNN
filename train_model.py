print("train_model.py STARTED")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths to dataset
train_dir = "train"
test_dir = "test"

# Image preprocessing (convert images to numbers)
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# Build CNN model
model = Sequential()

from tensorflow.keras.layers import Input

model.add(Input(shape=(48,48,1)))
model.add(Conv2D(32, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

# Save model
model.save("emotion_model.h5")

print("✅ Model training complete and saved as emotion_model.h5")
