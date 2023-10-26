# Assignment 1: Eagle Tracking Algorithm

## Overview
For Assignment 1, the task was to develop an algorithm that continuously tracks an eagle as it flies through various scenarios. The goal was to implement a tracking system that maintains continuous tracking without losing the target.

## Solution
To tackle this problem, I utilized the YOLO (You Only Look Once) v8 model for bird detection in each frame. I combined the detection results with a tracking library called SORT (Simple Online and Realtime Tracking). While I also explored the use of the Norfair tracking library, the SORT library provided more accurate and reliable tracking results.

https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/908a5cf4-8a7d-4074-b1c7-bfc92ba529c2

https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/06d36461-56d2-4005-bcb3-914c6d9cea65


https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/8e4cde9f-1324-4ddf-bde0-7807cda91e93



# Assignment 2: Skeet Shooting Algorithm

## Overview
Assignment 2 involved creating an algorithm for predicting the trajectory of clay targets in a skeet shooting scenario. Additionally, the algorithm was required to provide an estimation on a screen to guide the user on where to fire to hit the target. The challenges in this assignment included the movement of the camera and the lack of proper video footage of skeet competitions with real clay targets.

## Solution
Due to the unavailability of proper video footage, I used two videos showing different scenarios with clay targets. These videos had cameras mounted on shotguns, providing a first-person perspective. I collected frames from these videos and annotated them using Roboflow. The annotated dataset was then used to train a custom object detection model on Google Colab.

However, implementing the trajectory prediction turned out to be challenging. The moving camera and the absence of accurate distance and speed information made it difficult to estimate the clay target's trajectory and guide the user to hit the target accurately.

![skeet_training](https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/281c9ed9-6f99-4af8-bc51-432441ad05e6)



https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/e78b513a-ba90-4a53-8d01-e8a41bb14cb2


https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/05236f51-772f-4ab8-9c8b-b989fe2bc3ce




https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/ff3e300e-81a2-43df-907e-96411830df47



https://github.com/ShreyNaik123/astrdefence_assignment/assets/61283238/cb42d30e-b667-4051-aa17-e2a6be7d257b




For both assignments, the YOLOv8 model was used for object detection in the video frames, and SORT (Simple Online and Realtime Tracking) was employed for tracking the detected objects. These solutions combined object detection with tracking to address the tracking requirements of the assignments.

While the second assignment faced challenges in predicting clay target trajectories, the overall approach incorporated the use of computer vision and machine learning techniques to address the given problems.

