# Final-Year-Project
This project essentially focuses on classifying breast tumor as cancerous and non-cancerous. It further classifies the tumor into two more classes for each cancerous and non-cancerous.
So, basically there are 4 classes that we classify the tumor into.
We first classify the tumor into cancerous and non-cancerous. After that, if we get the image classified as cancerous, we further classify the image into in-situ or invasive. If we get the image classified as non-cancerous, we further classify the image into benign or normal.
To classify, we have trained three convolutional neural networks using keras as the framework with tensorflow as the backend. For any one particular image, two models out of the three perform operations on the images.
For classification, we have divided each image into 12 patches each of equal dimension. We classified each patch and based on the results of the classification, we label the images as predicted.
The classification strategy has an average accuracy of 89.21 % for non-cancerous tissue images and 88.80% for cancerous tissue images.
This entire classification model is uploaded on a web server so that people can feed images they want to classify directly into the web app and get their classification results in just a few seconds.
