import cv2
import numpy as np
import matplotlib.pyplot as plt
############################################################
###############   VARIABLES AND CONSTANTS   ################
############################################################
AREA = 4
output_size = (254,254)
output_dir = './simpson_faces/'
video_name = 'sample.mp4'
r = 1 # radius for black filter
lower_area_white = 50
upper_area_white = 40000
lower_area_yellow = 100
upper_area_yellow = 40000
k_size_blur = 5
k_size_color = 9
connect = 4
sensitivity_white = 50
sensitivity_black = 100
PATH = '/home/hasser/semestre_7/model_gener/Simpsons-Face-Detector/'
#YELLOW BOUNDS
lower_yellow = np.array([22, 93, 0], dtype="uint16")
upper_yellow = np.array([45, 255, 255], dtype="uint16")
#WHITE BOUNDS
lower_white = np.array([0,0,255-sensitivity_white], dtype="uint16")
upper_white = np.array([255,sensitivity_white,255], dtype="uint16")
#BLACK BOUNDS
lower_black = np.array([0,0,0], dtype="uint16")
upper_black = np.array([255,255,sensitivity_black], dtype="uint16")

############################################################
########################   PROGRAM   #######################
############################################################
images = []
cap = cv2.VideoCapture(PATH + video_name)
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
    ############################################################
    ###############   FILTER COMPONENTS BY AREA   ##############
    ############################################################
    white_indexes = [comp_index for comp_index in range(n_white_comp)
                    if white_stats[comp_index, AREA] >= lower_area_white and 
                    white_stats[comp_index, AREA] <= upper_area_white]
    white_comp_filt = np.where(white_masks==white_indexes[0], 255, 0)
    for index in white_indexes[1:]:
        white_comp_filt += np.where(white_masks==index, 255, 0)
    #YELLOW
    yellow_indexes = [comp_index for comp_index in range(n_yellow_comp)
                    if yellow_stats[comp_index, AREA] >= lower_area_yellow and 
                    yellow_stats[comp_index, AREA] <= upper_area_yellow]
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
    for face in faces:
        x,y,w,h,a = face
        cropped_image = image[y:y+h,x:x+w]
        cropped_image = cv2.resize(cropped_image, output_size)
        cv2.imwrite(output_dir + video_name + f'_{frame_num}.png', cropped_image)
    print(f"Frame {frame_num} finished: {len(faces)} faces found")