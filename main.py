#from importlib.resources import path
import cv2 as cv
import numpy as np
import time

# k-NN model module
from knn_model import train_knn_model, predict_knn_model

# mediapipie drawing and pose library
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

# trained knn model
knn_model = train_knn_model(csv_file_path='assets//fitness_poses_csvs_out_basic.csv', neighbors=5)
# possible class of y ('lying_down', 'standing_sitting')
y_class = ['lying_down', 'standing_sitting']

# time counter flags
timer_enabled = False
start_time_activated = False
timeout = 5.

# create VideoCapture object
cap = cv.VideoCapture(0)

while cap.isOpened():

    # read frame and change color for mediapipe pose detection
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # initialize fresh pose tracker and run it
    with mp_pose.Pose() as pose_tracker:
        result = pose_tracker.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        pose_landmarks = result.pose_landmarks

    # if there are poses detected, draw them to frame
    if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)

    # save landmarks
    if pose_landmarks is not None:
        pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
        pose_landmarks *= np.array([frame_width, frame_height, frame_width])

        # k-NN prediction
        prediction = predict_knn_model(knn_model=knn_model, data=pose_landmarks)[0]

        # put prediction result to frame
        cv.putText(frame, str(prediction), (30,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)



        # time counter
        # start timer if a person is lying down
        # and stop timer if a person is standing or sitting

        # if prediction is 'lying down'
        if prediction == y_class[0]:
            
            # if timer is not enabled, toggle the flag and start timer
            if timer_enabled == False:
                start_time = time.time()
                timer_enabled = True
            
            # calculate time elapsed and show it on frame
            time_elapsed = time.time() - start_time
            cv.putText(frame, str(round(time_elapsed,2)), (30,70), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            # after timeout (elapsed 5 sec), make an emergency alarm
            if time_elapsed > timeout:
                cv.putText(frame, 'EMERGENCY!', (30,110), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # if prediction is 'standing_sitting'
        elif prediction == y_class[1]:
            timer_enabled = False

        
     # show vidwo frame on screen   
    cv.imshow('press q to quit', frame)
    if cv.waitKey(1) == ord('q'):
        break

# release some resource at the end of the program
cv.destroyAllWindows()
cap.release()