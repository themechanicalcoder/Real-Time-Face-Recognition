# Real-Time-Face-Recognition
It is a Machine Learning project that recognizes faces of multiple people using Google facenet and SVM algorithm to predict the face in real time 

## Input Image 
![gourav](https://user-images.githubusercontent.com/34737471/59551214-a3a56200-8f93-11e9-9698-8022bdbeb77d.jpg)


## Output Image
![Gourav1](https://user-images.githubusercontent.com/34737471/59551232-fed75480-8f93-11e9-8c5f-f3f94f4ae085.jpg)

### Dependencies
  Python 3<br/>
  Tensorflow<br/>
  Keras<br/>
  Open-cv<br/>
  Scikit Learn<br/>
  Numpy<br/>
  Matplotlib<br/>

## Algorithm
1. First the face is detected and  extracted from the training images of all classes using opencv [haar-cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)(face_preprocessing.py)<br/>
2. Then the images are converted to a particular  size and  fed into the Google facenet which returns a 128 length vector per image and the labels are encoded into one-hot arrays.<br/>
3. The list of preprocessed images and one-hot arrays are then fed into SVM (face_classification.py)  <br/>
4. The model is then tested for accuracy (test.py)
5. For live prediction using webcam input image is taken from open-cv and it is again converted into 128 length vector using steps 1 to 3 <br/>

## How to use this repository
Make the following directory structure  <br/>
Dataset->data->train->training classes with images of respective classes <br/>
Dataset->data->val->test classes with images of respective classes <br/>
![New Doc 2019-06-15 19 18 29](https://user-images.githubusercontent.com/34737471/59552299-bd01da80-8fa2-11e9-9493-18dc53c934a8.jpg)
 ```python
if __name__ == "__main__": 
    
    trainX, trainy = load_dataset('D:\\Dataset\\data\\train')
    print(trainX.shape, trainy.shape)
    
    # load test dataset
    testX, testy = load_dataset('D:\\Dataset\\data\\val')

```
and put the directory address in the load_dataset() in above code in face_preprocessing.py<br/>
Then run embeddings.py->face_classification.py->test.y (for getting the accuracy)<br/>
For usign webcam you can run the file real_time_recognizer.py<br/>
 

 

The model achieved an accuracy of around 95.6% which is quite good as there were a total of around 100 images per class<br/> 
References-[Google Facenet model](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)  <br/>
 
