#Sourcs: https://heartbeat.fritz.ai/detecting-objects-in-videos-and-camera-feeds-using-keras-opencv-and-imageai-c869fe1ebcdb

from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "H:\\Learning not to Uplode\\Dataset\\blue37 Dataset\\models\\yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "H:\\Learning not to Uplode\\Dataset\\blue37 Dataset\\Videos\\1.mp4"),
                                output_file_path=os.path.join(execution_path, "H:\\Learning not to Uplode\\Dataset\\blue37 Dataset\Videos\\traffic_mini_detected_1")
                                , frames_per_second=24, log_progress=True)
print(video_path)