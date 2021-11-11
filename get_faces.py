import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
############################################################
###############   VARIABLES AND CONSTANTS   ################
############################################################
AREA = 4
IMAGE_SIZE = 256
# radius for black filter
r = 1 
h_padd = 30
w_padd = 30
lower_area_white = 50
upper_area_white = 40000
lower_area_yellow = 100
upper_area_yellow = 40000
k_size_blur = 5
k_size_color = 9
connect = 4
sensitivity_white = 50
sensitivity_black = 100
PATH = '/home/hasser/semestre_7/model_gener/simpsons_face_detector/'
input_videos = PATH + 'videos/videos2'
output_scenes = PATH + 'scenes/output_scenes2'
output_boxes = PATH + 'boxes/output_boxes2'
output_faces = PATH + 'faces/output_faces2'
csv_file = 'bounds.csv'
#YELLOW BOUNDS
lower_yellow = np.array([0, 193, 157], dtype="uint16")
upper_yellow = np.array([30, 255, 255], dtype="uint16")
#WHITE BOUNDS
lower_white = np.array([0,0,255-sensitivity_white], dtype="uint16")
upper_white = np.array([255,sensitivity_white,255], dtype="uint16")
#BLACK BOUNDS
lower_black = np.array([0,0,0], dtype="uint16")
upper_black = np.array([255,255,sensitivity_black], dtype="uint16")
df = pd.DataFrame(columns=['x', 'y', 'width', 'height', 'frame_id', 'face_num', 'video'])
############################################################
########################   PROGRAM   #######################
############################################################
for video_name in os.listdir(input_videos):
    cap = cv2.VideoCapture(input_videos + video_name)
    frame_rate = 30
    frame_num = 0
    while cap.isOpened():
        frame_num +=1
        ret, image = cap.read()
        if not ret:
            break
        if not frame_num % frame_rate == 0:
            continue
        ############################################################
        ###############   RGB, GRAY, BLURRED, HSV   ################
        ############################################################
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        blurred = cv2.medianBlur(rgb, k_size_blur)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_black_copy = mask_black.copy()
        height, width = mask_black_copy.shape
        for i in range(height):
            for j in range(width):
                try:
                    if (mask_black[i-r:i+r+1, j-r:j+r+1]==255).any():
                        mask_black_copy[i,j] = 255
                    else:
                        mask_black_copy[i,j] = 0
                except IndexError as e:
                    mask_black_copy[i,j] = 0
        mask_black = mask_black_copy
        ############################################################
        ###############   FILTER COLORS YELLOW, BLACK  #############
        ############################################################
        #YELLOW
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        blurred_yellow = cv2.medianBlur(mask_yellow, k_size_color)
        yellow_minus_black = np.where(mask_black==255, 0, blurred_yellow)
        #WHITE
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        blurred_white = cv2.medianBlur(mask_white, k_size_color)
        white_minus_black = np.where(mask_black==255, 0, blurred_white)
        n_white_comp, white_masks, white_stats, white_centroids = cv2.connectedComponentsWithStats(white_minus_black, connectivity=connect)
        n_yellow_comp, yellow_masks, yellow_stats, yellow_centroids = cv2.connectedComponentsWithStats(yellow_minus_black, connectivity=connect)
        if n_white_comp == 0 or n_yellow_comp == 0:
            continue
        ############################################################
        ###############   FILTER COMPONENTS BY AREA   ##############
        ############################################################
        white_indexes = [comp_index for comp_index in range(n_white_comp)
                        if white_stats[comp_index, AREA] >= lower_area_white and 
                        white_stats[comp_index, AREA] <= upper_area_white]
        yellow_indexes = [comp_index for comp_index in range(n_yellow_comp)
                        if yellow_stats[comp_index, AREA] >= lower_area_yellow and 
                        yellow_stats[comp_index, AREA] <= upper_area_yellow]
        if len(white_indexes) == 0 or len(yellow_indexes) == 0:
            continue
        white_comp_filt = np.where(white_masks==white_indexes[0], 255, 0)
        for index in white_indexes[1:]:
            white_comp_filt += np.where(white_masks==index, 255, 0)
        yellow_comp_filt = np.where(yellow_masks==yellow_indexes[0], 255, 0)
        for index in yellow_indexes[1:]:
            yellow_comp_filt += np.where(yellow_masks==index, 255, 0)
        ############################################################
        ###############         DETECT FACES        ################
        ############################################################
        faces = []
        for yellow_box in yellow_stats[yellow_indexes]:
            n_white_boxes = 0
            x,y,width,height,area = yellow_box
            y_x1, y_x2, y_y1, y_y2 = x, x+width, y, y+height
            for white_box in white_stats[white_indexes]:
                x,y,width,height,area = white_box
                w_x1, w_x2, w_y1, w_y2 = x, x+width, y, y+height
                # If white box contained in yellow box
                if w_x1 >= y_x1 and w_x2 <= y_x2 and y_y1 <= w_y1 and y_y2 >= w_y2:
                    n_white_boxes +=1
            if n_white_boxes == 2:
                faces.append(yellow_box)
        frame_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        for face_num, face in enumerate(faces):
            x,y,w,h,a = face
            try:
                cropped_image = image[y-h_padd : y+h+h_padd, x-w_padd : x+w+w_padd]
                box = {'x':x-w_padd,
                    'y':y-h_padd,
                    'width':w + 2*w_padd,
                    'height':h + 2*h_padd,
                    'frame_id':frame_id,
                    'face_num':face_num,
                    'video':video_name}
            except IndexError:
                cropped_image = image[y:y+h,x:x+w]
                box = {'x':x,
                    'y':y,
                    'width':w,
                    'height':h,
                    'frame_id':frame_id,
                    'face_num':face_num,
                    'video':video_name}
            finally:
                df = df.append(box, ignore_index=True)
                cropped_image = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
                cv2.imwrite(f"{output_faces}{frame_id}_{face_num}.png", cropped_image)
                cv2.imwrite(f"{output_scenes}{frame_id}.png", image)
    df.to_csv(output_boxes + csv_file)
    df.to_csv(output_boxes + video_name[:-4] + '_' + csv_file)