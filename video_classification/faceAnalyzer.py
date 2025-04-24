from feat import Detector
import os
import cv2
import torch
import csv
import datetime

class face_analyzer:

    def __init__(self):
        self.detector = Detector()
        self.face_detection_threshold = 0.8   
    
    def analyze_video(self, video_path):
        video_prediction = self.detector.detect(video_path, data_type="video", skip_frames=24, face_detection_threshold=self.face_detection_threshold)
        video_prediction.head()
        video_name = video_path[:len(video_path)-4] + ".csv"
        print(video_name)
        video_prediction.to_csv(video_name, index=False)