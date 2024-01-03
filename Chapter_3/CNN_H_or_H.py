import tensorflow as tf
import keras as ks
from keras.preprocessing.image import ImageDataGenerator

training_dir = 'horse-or-human/training/'
validation_dir = 'horse-or-human/validation/'

# All images will be rescaled by 1/255
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300,300),
    class_mode='binary'
)

model = ks.models.Sequential([
    ks.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    ks.layers.MaxPooling2D(2, 2),
    ks.layers.Conv2D(32, (3,3), activation='relu'),
    ks.layers.MaxPooling2D(2,2),
    ks.layers.Conv2D(64, (3,3), activation='relu'),
    ks.layers.MaxPooling2D(2,2),
    ks.layers.Conv2D(64, (3,3), activation='relu'),
    ks.layers.MaxPooling2D(2,2),
    ks.layers.Conv2D(64, (3,3), activation='relu'),
    ks.layers.MaxPooling2D(2,2),
    ks.layers.Flatten(),
    ks.layers.Dense(512, activation='relu'),
    ks.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=ks.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)