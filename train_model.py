from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense ,Rescaling
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt


image_size = 100

datagen = ImageDataGenerator(rescale=1.0/255.0)
train_dataset = datagen.flow_from_directory('prepared data/train', class_mode='binary', batch_size=128, target_size=(image_size, image_size))
test_dataset = datagen.flow_from_directory('prepared data/test', class_mode='binary', batch_size=128, target_size=(image_size, image_size))


def define_model():

    model = Sequential()

    model.add(Conv2D(28, kernel_size=(3,3), input_shape=(image_size, image_size, 3),activation='relu'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model



model=define_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_dataset, steps_per_epoch=128,
                    validation_data=test_dataset, validation_steps=128, epochs=30, verbose=1)

_, acc = model.evaluate(test_dataset, steps=len(test_dataset), verbose=1)
print('> %.3f' % (acc * 100.0))

model.save('voice_recognition_model')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
