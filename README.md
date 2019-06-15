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
  Open-cv
  Scikit Learn<br/>
  Numpy<br/>
  Matplotlib<br/>

## Algorithm
1.First the face is extracted from the image using [haar-cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)<br/>
2.Then the image is fed into the Google facenet which returns a 128 length <br/>
3.The list of images is then fed into SVM <br/>
4.For live prediction using webcam input image is taken from open-cv and it is again converted into a 128 length vector which is then passed then answer is predicted using the trained SVM model <br/>

References [Google Facenet model](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)  <br/>
The model achieved an accuracy of around 95.6% which is quite good as theere were a total of around 100 images per class<br/>  
