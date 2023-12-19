#instalacja pakietu gdown
!pip install -U --no-cache-dir gdown --pre
#usuwanie i tworzenie pustego katalogu data
!rm -rf data && mkdir data
#pobieranie zip-a ze zdjęciami
!gdown 1BpccrvnDWO4XcOSQz08Xc4rGYQR7gOLq -O ai.zip
#rozpakowywanie pliku data.zip do katalogu data
!unzip -q ai.zip -d data

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import glob
import os
import xml.etree.ElementTree as ET

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tree = ET.parse('/content/data/ai/annotations.xml')
root = tree.getroot()

for child in root.iter('image'):
    image_name = child.attrib['name'].replace('ai/', '')
    image_label = child[0].attrib['label']

    if not os.path.isdir(f'/content/data/ai/{image_label}'):
        os.mkdir(f'/content/data/ai/{image_label}')

    os.replace(f'/content/data/ai/{image_name}', f'/content/data/ai/{image_label}/{image_name}')

cola_count = len(list(glob.glob('data/**/cola/*.jpg')))
fanta_count = len(list(glob.glob('data/**/fanta/*.jpg')))
sprite_count = len(list(glob.glob('data/**/sprite/*.jpg')))
print(f'{cola_count} examples of cola, {fanta_count} examples of fanta and {sprite_count} examples of sprite to train')

cola = list(glob.glob('data/ai/cola/*'))
PIL.Image.open(str(cola[3]))

batch_size = 32
class_count = 3

img_height = 64
img_width = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
  'data/ai',
  validation_split=0.2,
  subset='training',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'data/ai',
  validation_split=0.2,
  subset='validation',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(f'class names: {class_names}')

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(class_count)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

from sklearn.metrics import classification_report
def evaluate_model(val_ds, model):
    y_pred = []
    y_true = []

    for batch_images, batch_labels in val_ds:
        predictions = model.predict(batch_images, verbose=0)
        y_pred = y_pred + np.argmax(tf.nn.softmax(predictions), axis=1).tolist()
        y_true = y_true + batch_labels.numpy().tolist()
    print(classification_report(y_true, y_pred))

evaluate_model(val_ds, model)