import cv2
import pandas
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import streamlit as st

video_path='data/parking_1920_1080_loop.mp4'
mask='parking-space-counter-master/mask_1920_1080.png'

# video_path='data/parking_1920_1080_loop_2.mp4'
# mask='parking-space-counter-master/masktrue2.png'



mask=cv2.imread(mask)
mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

connected_components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    

    spaces = []
    
    for i in range(1, totalLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT] )
        y1 = int(values[i, cv2.CC_STAT_TOP] )
        w = int(values[i, cv2.CC_STAT_WIDTH] )
        h = int(values[i, cv2.CC_STAT_HEIGHT])

        spaces.append([x1, y1, w, h])

    return spaces

spaces=get_parking_spots_bboxes(connected_components)

model = load_model('parking-space-counter-master/parking_model.keras')

empty=True
not_empty=False

def predictimage(img):
        img_resized=resize(img,(150,150,1))
        final_image=img_resized.reshape(1,150,150,1)
        result=model.predict(final_image)
        if result<0.5:
            return empty
        else:
            return not_empty



def calc_diff(im1,im2):
    return np.abs(np.mean(im1)-np.mean(im2))

cap=cv2.VideoCapture(video_path)

status=[None for j in spaces]
frame_number=0
step_size=60
diffs=[None for j in spaces]

prev_frame=None



while True:
    suc,frame = cap.read()
    if frame_number%step_size==0 and prev_frame is not None:
        for index,space in enumerate(spaces):
            x1,y1,w,h=space
            space_cropped=frame[y1:y1+h,x1:x1+w,:]

            diffs[index]=calc_diff(space_cropped,prev_frame[y1:y1+h,x1:x1+w,:])

        #print(diffs[j] for j in np.argsort(diffs)[::-1])


    if frame_number%step_size==0:
        if prev_frame is None:
            fr=range(len(spaces))
        else:
            fr=[j for j in np.argsort(diffs) if (diffs[j]/np.max(diffs))>0.55]

        for index in fr:
            space=spaces[index]
            x1,y1,w,h=space
            space_cropped=frame[y1:y1+h,x1:x1+w,:]
            space_status=predictimage(space_cropped)
            status[index]=space_status  


    
    if frame_number%step_size==0:
        prev_frame=frame.copy()
     

    for index,space in enumerate(spaces):
        x1,y1,w,h=spaces[index]
        space_status=status[index]
        if space_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)



    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame,f'Available Space: {str(sum(status))} / {str(len(spaces))}', (100,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.namedWindow('Window',cv2.WINDOW_NORMAL)
    cv2.imshow('Window',frame)
    if cv2.waitKey(25) & 0XFF==ord("q"):
        break
    frame_number+=1

cap.release()
cv2.destroyAllWindows()




