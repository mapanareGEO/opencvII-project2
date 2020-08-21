# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:26:57 2020

@author: Gregorio Esteban Ortega     mapanare666@gmail.com   mapanare
"""
"""
Project 2: DoppelGanger Find your Celebrity Look-Alike 
We have seen how face Recognition algorithms work. In short, they convert a face 
into a high dimensional vector and then learn a feature space in that high dimension. 
Once the algorithm learns a robust feature space, you can add more people 
to the same space and they will form a different cluster.

Your Task
In this project, we will build a Fun Application - We will use the Face Embeddings 
to find a doppelganger or look-alike to a given person. 
Specifically, we will give you images of two persons and a celebrity dataset. 
Your task will be to find their celebrity look-alike. 
The first person is Sofia Solares who looks like the American Singer Selena Gomez 
and the second one is Shashikant Pedwal who looks like Indian Film Actor Amitabh Bachchan.

Given below is the Expected Output.

Expected Output
"""
"""
By now, you should be able to discern what the problem is, and how you are going to solve it. 
You simply need a dataset which has enough celebrity faces and use face embeddings 
to match the test image with the celebrity face embeddings. That's enough Hint!

The Dataset
There are many datasets which contain images of celebrities. Some of them are:

CelebA Dataset
VGGFace
VGGFace2
MS-Celeb-1M
Celebrity-Together Dataset
celeb_mini Dataset
Since most of the datasets are huge in size, we have curated a small dataset 
for this experiment.

We will use a subset of the Query set of the Celebrity Together Dataset in this project. 
The query dataset has around 1100 celebs and is ~400MB in size.

The subset we have created is using 5 images per celebrity and it has been saved 
in the ../resource/asnlib/publicdata/celeb_mini folder. It is ~70MB in size. 
We have also provided a celeb_mapping.npy file which has the mapping 
of the code name to the actual name of the celebrity.

The celeb_mini folder has the following structure:

celeb_mini
└───n00000001
│   └──n00000001_00000263.JPEG
│   └──n00000001_00000405.JPEG
│      ...
└───n00000003
│   └──n00000003_00000386.JPEG
│   └──n00000003_00000488.JPEG
│       ...
│
celeb_mapping.npy contains a dictionary which contains the mapping between 
the folder names  and the celebrity's actual name as show below:

{'n00000001': 'A.J. Buckley',
 'n00000002': 'A.R. Rahman',
 'n00000003': 'Aamir Khan',
 'n00000004': 'Aaron Staton',
 'n00000005': 'Aaron Tveit',
 'n00000006': 'Aaron Yoo',
 'n00000007': 'Abbie Cornish',
 .
 .
 .
}
Marks Distribution 
Implementation - 70 Marks
Video Explanation - 30 marks
Note that you are free to use any other dataset if you have the resources of space 
and compute to handle the large datasets.
"""


import os,random,glob
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'


"""
Load the model for face landmarks and face recognition
Initialize Dlib’s Face Detector, Facial Landmark Detector and Face Recognition neural network objects

You are free to use other face recognition models as well. 
"""

# Download dlib models
# Landmark model
#!wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 -O shape_predictor_68_face_landmarks.dat.bz2
#!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
# Face recognition model
#!wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2 -O dlib_face_recognition_resnet_model_v1.dat.bz2
#!bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
# Initialize face detector, facial landmarks detector 
# and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
faceRecognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


"""
Explore Data 
In this section, we provide sample code of how to load the images and use the label mapping for your reference. 
This should help you get started with handling images and finding the embeddings.
"""

#!wget https://www.dropbox.com/s/lulw2jwsblz687g/celeb_mini.zip?dl=1 -O celeb_mini.zip
#!unzip celeb_mini.zip

#!wget https://www.dropbox.com/s/m7kjjoa1z1hsxw6/celeb_mapping.npy?dl=1 -O celeb_mapping.npy

#!wget https://www.dropbox.com/s/3yi89blp4lhiw6y/test-images.zip?dl=1 -O test-images.zip
#!unzip test-images.zip

# Root folder of the dataset
faceDatasetFolder = 'celeb_mini'
# Label -> Name Mapping file
labelMap = np.load("celeb_mapping.npy", allow_pickle=True).item()

# Each subfolder has images of a particular celeb
subfolders = os.listdir(faceDatasetFolder)

# Let us choose a random folder and display all images
random_folder = random.choice(subfolders)
print("key for labelMap ",random_folder)
# Also find out the name of the celeb from the folder name and folder-> name mapping dictionary loaded earlier
celebname = labelMap[random_folder]

# Load all images in the subfolder
imagefiles = os.listdir(os.path.join(faceDatasetFolder, random_folder))

# Read each image and display along with the filename and celeb name
for file in imagefiles:
#     Get full path of each image file
    fullPath = os.path.join(faceDatasetFolder,random_folder,file)
    im = cv2.imread(fullPath)
    plt.imshow(im[:,:,::-1])
    plt.show()
#     Also print the filename and celeb name
    print("File path = {}".format(fullPath))
    print("Celeb Name: {}".format(celebname))
    
"""
TODO
You have to write the code for enrollment and testing to get the desired output as shown in the intro section.

You can take help from the code we had provided in week 4 material.

1. Enrollment of Celeb Images 
Process images one by and one.

Convert image from RGB to BGR, because Dlib uses RGB as default format.

Detect faces in the image and compute a face descriptor for each image and save them in an array so that it can be used during testing.
"""

#
#
# #YOUR CODE HERE
# read subfolders in folder "celeb_mini"
subfolders = []
for x in os.listdir(faceDatasetFolder):
  xpath = os.path.join(faceDatasetFolder, x)
  if os.path.isdir(xpath):
    subfolders.append(xpath)

# nameLabelMap is dict with keys as person's name
# and values as integer label assigned to this person
# labels contain integer labels for corresponding image in imagePaths
nameLabelMap = {}
labels = []
imagePaths = []
for i, subfolder in enumerate(subfolders):
  for x in os.listdir(subfolder):
    xpath = os.path.join(subfolder, x)
    if x.endswith('JPEG'):
      imagePaths.append(xpath)
      labels.append(i)
      nameLabelMap[xpath] = subfolder.split('/')[-1]

# Process images one by one
# We will store face descriptors in an ndarray (faceDescriptors)
# and their corresponding labels in dictionary (index)
index = {}
i = 0
faceDescriptors = None
for imagePath in imagePaths:
  print("processing: {}".format(imagePath))
  # read image and convert it to RGB
  img = cv2.imread(imagePath)

  # detect faces in image
  faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  print("{} Face(s) found".format(len(faces)))
  # Now process each face we found
  for k, face in enumerate(faces):

    # Find facial landmarks for each detected face
    shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

    # convert landmarks from Dlib's format to list of (x, y) points
    landmarks = [(p.x, p.y) for p in shape.parts()]

    # Compute face descriptor using neural network defined in Dlib.
    # It is a 128D vector that describes the face in img identified by shape.
    faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

    # Convert face descriptor from Dlib's format to list, then a NumPy array
    faceDescriptorList = [x for x in faceDescriptor]
    faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
    faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

    # Stack face descriptors (1x128) for each face in images, as rows
    if faceDescriptors is None:
      faceDescriptors = faceDescriptorNdarray
    else:
      faceDescriptors = np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)

    # save the label for this face in index. We will use it later to identify
    # person name corresponding to face descriptors stored in NumPy Array
    index[i] = nameLabelMap[imagePath]
    i += 1
"""
# Write descriors and index to disk
# faceDescriptors are 128 mumbers for each image, it is not necessary to save it
np.save('descriptors.npy', faceDescriptors)
try:
  import cPickle  # Python 2
except ImportError:
  import _pickle as cPickle  # Python 3
# index has image paths in same order as descriptors in faceDescriptors
with open('index.pkl', 'wb') as f:
  cPickle.dump(index, f)
"""

# we don't need to save to file and load again
# but, we do it this way as a shortcut to only enroll once in the jupiter notebook
faceDescriptorsEnrolled = faceDescriptors 
#
# end of my code

"""
2. Testing
Once you have enrolled the celebs, you should be able to use them to find 
the closest celeb to the given test image using minimum distance rule.

You need to find the embeddings of the test images and compare it with the enrolled celebs to find the look-alike.

Finally, display the celeb along side the input image. Partial code is given.
"""

matplotlib.rcParams['figure.figsize'] = (12.0,12.0)

# read image
testImages = glob.glob('test-images/*.jpg')

for test in testImages:
    im = cv2.imread(test)
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    #####################
    #  YOUR CODE HERE
    
    THRESHOLD = 0.6
    # load descriptors and index file generated during enrollment
    #index = np.load("index.pkl",allow_pickle=True)
    #faceDescriptorsEnrolled = np.load('descriptors.npy')
    #faceDescriptorsEnrolled = faceDescriptors

    faces = faceDetector(imDlib)
    # Now process each face we found
    for face in faces:

        # Find facial landmarks for each detected face
        shape = shapePredictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)

        # find coordinates of face rectangle
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        # Compute face descriptor using neural network defined in Dlib
        # using facial landmark shape
        faceDescriptor = faceRecognizer.compute_face_descriptor(im, shape)

        # Convert face descriptor from Dlib's format to list, then a NumPy array
        faceDescriptorList = [m for m in faceDescriptor]
        faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
        faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

        # Calculate Euclidean distances between face descriptor calculated on face dectected
        # in current frame with all the face descriptors we calculated while enrolling faces
        distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1)
        # Calculate minimum distance and index of this face
        argmin = np.argmin(distances)  # index
        minDistance = distances[argmin]  # minimum distance

        # Dlib specifies that in general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.

        # This threshold will vary depending upon number of images enrolled
        # and various variations (illuminaton, camera quality) between
        # enrolled images and query image
        # We are using a threshold of 0.5

        # If minimum distance if less than threshold
        # find the name of person from index
        # else the person in query image is unknown
        if minDistance <= THRESHOLD:
            label = index[argmin]
        else:
            label = 'unknown'


        # Draw a rectangle for detected face
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255))

        # Draw circle for face recognition
        center = (int((x1 + x2)/2.0), int((y1 + y2)/2.0))
        radius = int((y2-y1)/2.0)
        color = (0, 255, 0)
        cv2.circle(im, center, radius, color, thickness=1, lineType=8, shift=0)

        # Write text on image specifying identified person and minimum distance
        org = (int(x1), int(y1))  # bottom left corner of text string
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_color = (255, 0, 0)
        printLabel = '{} {:0.4f}'.format(label, minDistance)
        cv2.putText(im, printLabel, org, font_face, font_scale, text_color, thickness=2)

  
    
    
    # end of my code
    
    
    ####################
    
    plt.subplot(121)
    plt.imshow(imDlib)
    plt.title("test img")
    
    #TODO - display celeb image which looks like the test image instead of the black image. 
    plt.subplot(122)
    
    thePath= label.replace('\\', '/')
    print(thePath) # print the label for each image in testImages
   
    imageList=[]
    for img in glob.glob(thePath+"/*.JPEG"):
        imageList.append(img) 
        
    # read the image of the celeb look alike, assign it to celeb_img    
    celeb_img=cv2.imread(imageList[0])
    # change the space color from BGR to RGB to show it with plt.
    celeb_img=cv2.cvtColor(celeb_img,cv2.COLOR_BGR2RGB)
    plt.imshow(celeb_img)
    #plt.imshow(np.zeros((200,200,3)))
    # get from dict labelMap the celeb_name 
    my_string=label.split("\\", 1)[-1]
    print(my_string)
    celeb_name=labelMap[my_string]
    print(celeb_name)
    # print the celeb_name on the celeb image
    plt.title("Celeb Look-Alike={}".format(celeb_name))
    plt.show()
    cv2.destroyAllWindows()
    
