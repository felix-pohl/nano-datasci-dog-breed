# Data Science Nanodegree Project: Dog Breed Classifier

## Project Overview

The goal of this project is providing a web app to make dog breed classification as easy as uploading an image and pressing send. To archive this feat this project will employ CNNs (convolutional neural networks). Classifications will be based on established Computer Vision Models such as VGG16 and ResNet50.

## Problem Statement

Idetifiying dog breeds from images alone can be quite problematic. Some breeds look very alike and can be misjudged by unexperienced people.

## Findings

Building a CNN from scratch can be quite demanding and not all architectures guarantee good accuracy. This project has started with building a small CNN from scracth with a accuracy of 8.9% when identifying breeds. Scaling this up to many more layers could have given better results but would be very ressource intensive. 

Rather than building our own model transfer learning from the estabslied ResNet50 was utilized to increase our accuracy to 81.7%. This has given an incredible boost while being very ressource friendly.

In the final step this trained model was transfered into a web app for conventient use by anyone. 

## Conslusion

Trying to build my own CNN seemed a good idea at first. But low accuracy and big time investment was very punishing. As image classification has already been explored by many other projects transfer learning seemed to be a good option. 

Overfitting and very slow learing rates we most problematic als changes in model arhictecture meant long training sessions. Changes were also not always guranteed to get better results and sometimes even stopped to improve error loss after a few epochs.

After trying VGG16 als a basis for the classification we moved on to ResNet50, wich gave better results. We could also have tried VGG19 or AlexNet, but the accuracy of over 80% seemed acceptable.

As the model was trained it had to be made accessable through the web app. A simple Flask app with a straight forward design and the sole focus on recognizing dog breeds was created. Transfering the model took little effort as it was saved in the training process.

Further improvements could be made to classification model by testing other architectures or investing more time into training and hyperparameter tuning.
An interesting improvement for the webapp could be expressing the top 3 most likely breeds oder showing a side by side view with a representational image of the identified breed.

## Further details

For further details in training and data exploration please see: [notebook](dog_app.ipynb) or [html version](dog_app.html)

## Web App

A web app is provided in the app directory.

The app consists of a cv_predictor that uses the pretrained model from the python notebook. Because the dog dataset is quite bit, I chose to include all dog categories as a list in dog_breeds.py for easy access.

### Required Dependencies

```
	tensorflow==2.5.0
	flask==2.0.1
	opencv-python==4.5.3.56
	keras==2.4.3
```

### Additional data before starting the App

Donwload the [Resnet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/project/bottleneck_features`.

### Running the App

Run the app by navigating to the app directory and execute 

```
cd app
python run.py
```

### Web App files

- `app.py` starts a flask app and provides routes for form submission and results
- `cv_predictor.py` uses the pretrained model to classify dog breeds. Humans and dogs are identified with ResNet and haarcascades.
- `dog_names.py` List of dog breeds as the original dogs dataset is not included
- `extract_bottleneck_features.py` utility for extracting features from different CNNs
- `run.py` starts the app
- `templates\identify.html` html markup providing a form and results