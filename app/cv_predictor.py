import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from keras.preprocessing import image

from dog_names import dog_names
from extract_bottleneck_features import *

# load saved model
Resnet50_model = load_model('../saved_models/weights.best.Resnet50.hdf5')

# define ResNet50 model
ResNet50_model_ = ResNet50(weights='imagenet')

# extract pre-trained face detector
path = '../haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(path)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def breed(image_path):
    human = face_detector(image_path)
    dog = dog_detector(image_path)
    if not(dog) and not(human):
        return 'You have hidden very well, please provide a better image of your dog'
    breed = Resnet50_predict_breed(image_path)
    breed = breed.rsplit('.', 1)[1]
    if human:
        return 'This human looks like a {0}'.format(breed)
    if dog:
        return 'This dog looks like a {0}'.format(breed)
