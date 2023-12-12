import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

# plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20, validation_data=(testing_images,testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')


# img = cv.imread('deer.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# plt.imshow(img, cmap=plt.cm.binary)

# prediction = model.predict(np.array([img]) / 255)
# index = np.argmax(prediction)

# print(f"Prediction is {class_names[index]}")

# plt.show()















# import cv2 
# image = cv2.imread('image.png') 
# h, w = image.shape[:2] 
# print("Height = {}, Width = {}".format(h, w)) 





# (B, G, R) = image[100, 100] 
# print("R = {}, G = {}, B = {}".format(R, G, B)) 
# B = image[100, 100, 0] 
# print("B = {}".format(B)) 



# roi = image[100 : 500, 200 : 700] 


# resize = cv2.resize(image, (800, 800)) 




# ratio = 800 / w 
# dim = (800, int(h * ratio)) 
# resize_aspect = cv2.resize(image, dim) 
