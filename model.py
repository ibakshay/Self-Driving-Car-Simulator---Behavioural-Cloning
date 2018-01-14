import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

lines = []
with open('./TrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        #filename = source_path.split("/")[-1]
        #current_path = "./TrainingDataTesting/IMG/" + filename
        #image = cv2.imread(source_pathaawawwwwwwwwaw)
        images.append(source_path)
        measurement = float(line[3])
        if i == 0:
            measurements.append(measurement)
        if i == 1:
            measurements.append(measurement+0.2)
        if i == 2:
            measurements.append(measurement-0.2)
X = np.array(images)
y = np.array(measurements)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

def generator(X_samples, y_samples, batch_size=32):
    num_samples = len(X_samples)
    while 1: # Loop forever so the generator never terminates
        #shuffle(X)
        for offset in range(0, num_samples, batch_size):
            batch_X_samples = X_samples[offset:offset+batch_size]
            batch_y_samples = y_samples[offset:offset+batch_size]
            images = []
            angles = []
            for x,y in zip(batch_X_samples,batch_y_samples):
                
                image = cv2.imread(x)
                center_angle = y
                #print(shape(image))
                
                #img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                #img = img[:, :, 2] # choose S channel
                #img = img.reshape(( img.shape[0], img.shape[1], 1))
                
                images.append(image)
                angles.append(center_angle)

            X_output = np.array(images)
            y_output = np.array(angles)
            #print(len(X_train))
            # print("X_train: ", X_train)
            # print("y_train: ", y_train)
            yield shuffle(X_output, y_output)

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=240)
#for x,y in generator(X_train, y_train, batch_size=32):
#    print(y)
validation_generator = generator(X_test, y_test, batch_size=240)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

print("X_train:",len(X_train)," y_train:", len(y_train), " X_test:", len(X_test), " y_test:", len(y_test))

model = Sequential()
h,w,c=32,64,3
#Preprocess
model.add(Lambda(lambda x : x/255 - 0.5, input_shape = (160,320, 3)))
model.add(Cropping2D(cropping = ((70, 25),(0,0))))

model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='same',
                            init = 'he_normal'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(16, 5, 5, subsample=(4, 4), border_mode="same",
                            init = 'he_normal'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same",
                            init = 'he_normal'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", 
                            init = 'he_normal'))
model.add(Flatten())
model.add(Dropout(.2))
#model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.2))
#model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(1))
print(model.layers[-1].output_shape)
#adam = Adam(lr=0.0001)
model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(train_generator, 
                    samples_per_epoch=len(X_train), 
                    validation_data=validation_generator,
                    nb_val_samples=len(X_test), nb_epoch=10)
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=5)

model.save("model.ib")
model.summary()
