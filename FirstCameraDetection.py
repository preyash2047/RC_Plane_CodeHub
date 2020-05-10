# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:39:24 2020

@author: preya
"""
#Sourcs: https://heartbeat.fritz.ai/detecting-objects-in-videos-and-camera-feeds-using-keras-opencv-and-imageai-c869fe1ebcdb

from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

#for ip address based campera
#camera = cv2.VideoCapture("http://192.168.43.1:8080/video")
camera = cv2.VideoCapture(0) 

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "H:\\Learning not to Uplode\\Dataset\\blue37 Dataset\\models\\yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "H:\\Learning not to Uplode\\Dataset\\blue37 Dataset\\Videos\\camera_detected_1")
                                , frames_per_second=1, log_progress=True)
print(video_path)