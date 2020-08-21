# opencvII-project2
In this project you get a lot of images from celebs in a file celeb_mini.zip, when unzip it, it has 1166 sub-folders and each sub-folder has five images of a celebrity. 
The names of the sub-folders are numbers, so you can get a file called celeb_mapping.npy, it is a dictionary, with the name(number) of the folder as a key, 
you can get the name of the celeb in the photos from that sub-folder. 
They sugest to use the library dlib, so you also get the files shape_predictor_68_face_landmarks.dat for get the landmarks and dlib_face_recognition_resnet_model_v1.dat 
for the recognition. For the face detector, we are going to use the dlib face detector.
Besides there is a folder with two images inside, the test-images.zip, we have to use these images to find a celebrity look alike to them.
There is this program  “enrollDlibFaceRec.py” that enrolls the celeb images in the face predictor de Dlib.and the program  “testDlibFaceRecImage.py”, 
to find the celebrity look-alike to the test-image.
