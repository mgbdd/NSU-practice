from feat import Detector
import os
import torch
from imageHandler import image_handler
import csv
from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor

class face_analyzer:
    def __init__(self, dir_path):
        self.detector = Detector()
        self.ih = image_handler()  
        self._dir_path = dir_path
        self.face_detection_threshold = 0.8
        self.output_file, self.output_writer = self.create_output_csv()
        
        try:
            self.output_file, self.output_writer = self.create_output_csv()
        except Exception as e:  
            raise RuntimeError(f"Failed to create CSV file for output: {e}") from e


    def analyze_image(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        tensor_image = torch.from_numpy(image_np).float()
        tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)

        print("Original image detection")
        orig_prediction = self.detector.detect(inputs=tensor_image, data_type="tensor", face_detection_threshold=self.face_detection_threshold,   progress_bar=True)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if len(orig_prediction.head()) != 1:
            self.save_csv_output(orig_prediction, None, None, image_name, len(orig_prediction.head()))
            return
            
        print("Aligning")
        tensor_image, nan_flag_rot = self.ih.rotate_tensor(tensor_image, orig_prediction.head())
        if nan_flag_rot:
            self.save_csv_output(orig_prediction, None, None, image_name, 0)
            return
        
        print("Aligned image detection")
        orig_prediction = self.detector.detect(inputs=tensor_image, data_type="tensor", face_detection_threshold=self.face_detection_threshold,  progress_bar=True)  
        
        print("Mirroring")
        left_mirrored_tensor, right_mirrored_tensor, nan_flag_mir = self.ih.mirror_faces(tensor_image, orig_prediction.head())
        if nan_flag_mir:
            self.save_csv_output(orig_prediction.head(), None, None, image_name, 0)
            return
        
        print("Left half detection")
        left_mirrored_prediction = self.detector.detect(inputs=left_mirrored_tensor, data_type="tensor",  face_detection_threshold=self.face_detection_threshold, progress_bar=True) 
        print("Right half detection")
        right_mirrored_prediction = self.detector.detect(inputs=right_mirrored_tensor, data_type="tensor", face_detection_threshold=self.face_detection_threshold, progress_bar=True) 

        self.save_csv_output(orig_prediction.head(), left_mirrored_prediction.head(), right_mirrored_prediction.head(), image_name, 1)

    def create_output_csv(self):
        try: 
            dir_name = os.path.basename(os.path.normpath(self._dir_path))
            csv_path = os.path.join(self._dir_path, f"{dir_name}.csv")
            csv_file = open(csv_path, 'w', newline='')  
            csv_writer = csv.writer(csv_file,  delimiter='\t')
            csv_writer.writerow(["FileName", "FaceNumber", "FaceScore","Pitch", "Roll", "Yaw", 
                                "WH_AU01", "WH_AU02", "WH_AU04", "WH_AU05", "WH_AU06", "WH_AU07", "WH_AU09", "WH_AU10", "WH_AU11", "WH_AU12", 
                                "WH_AU14", "WH_AU15", "WH_AU17", "WH_AU20", "WH_AU23", "WH_AU24", "WH_AU25", "WH_AU26", "WH_AU28", "WH_AU43", 
                                "WH_anger", "WH_disgust", "WH_fear", "WH_happiness", "WH_sadness", "WH_surprise", "WH_neutral", 
                                "LH_AU01", "LH_AU02", "LH_AU04", "LH_AU05", "LH_AU06", "LH_AU07", "LH_AU09", "LH_AU10", "LH_AU11", "LH_AU12", 
                                "LH_AU14", "LH_AU15", "LH_AU17", "LH_AU20", "LH_AU23", "LH_AU24", "LH_AU25", "LH_AU26", "LH_AU28", "LH_AU43",
                                "LH_anger", "LH_disgust", "LH_fear", "LH_happiness", "LH_sadness", "LH_surprise", "LH_neutral",
                                "RH_AU01", "RH_AU02", "RH_AU04", "RH_AU05", "RH_AU06", "RH_AU07", "RH_AU09", "RH_AU10", "RH_AU11", "RH_AU12", 
                                "RH_AU14", "RH_AU15", "RH_AU17", "RH_AU20", "RH_AU23", "RH_AU24", "RH_AU25", "RH_AU26", "RH_AU28", "RH_AU43",
                                "RH_anger", "RH_disgust", "RH_fear", "RH_happiness", "RH_sadness", "RH_surprise", "RH_neutral",]) 
            return csv_file, csv_writer
        except Exception as e:
            return None, None

    def save_csv_output(self, orig_prediction, left_prediction, right_prediction, image_name, face_number):
        try:
            if face_number != 1:
                self.output_writer.writerow([image_name, face_number]) # если лицо не обнаружено или больше 1 лица, записываем пустую строку
            else:
                fase_score = orig_prediction.iloc[0]["FaceScore"]

                pitch_i = orig_prediction.columns.get_loc("Pitch")
                yaw_i = orig_prediction.columns.get_loc("Yaw") + 1
                pry_piece = orig_prediction.iloc[0:1, pitch_i:yaw_i].values.tolist()[0] 

                first_wh_i = orig_prediction.columns.get_loc("AU01")
                last_wh_i = orig_prediction.columns.get_loc("neutral") + 1
                wh_aus = orig_prediction.iloc[0:1, first_wh_i:last_wh_i].values.tolist()[0]

                first_lh_i = right_prediction.columns.get_loc("AU01")
                last_lh_i = right_prediction.columns.get_loc("neutral") + 1
                lh_aus = right_prediction.iloc[0:1, first_lh_i:last_lh_i].values.tolist()[0]

                first_rh_i = left_prediction.columns.get_loc("AU01")
                last_rh_i = left_prediction.columns.get_loc("neutral") + 1
                rh_aus = left_prediction.iloc[0:1, first_rh_i:last_rh_i].values.tolist()[0]

                row_to_write = [image_name, face_number, fase_score] + pry_piece + wh_aus + lh_aus + rh_aus

                self.output_writer.writerow(row_to_write)
        
        except Exception as e:
            print(f"Failed to write output for image {image_name} in csv output file: {e}")
        return

