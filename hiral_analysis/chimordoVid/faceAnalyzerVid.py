from feat import Detector
import os
import cv2
import torch
from frameHandler import frame_handler
import csv
import datetime

class face_analyzer:

    def __init__(self):
        self.detector = Detector()
        self.fh = frame_handler()
        self.face_detection_threshold = 0.8
        
    def analyze_frame(self, frame_tensor):
        frame_prediction = self.detector.detect(inputs=frame_tensor, data_type="tensor", face_detection_threshold=self.face_detection_threshold, progress_bar=True)
        return frame_prediction       
    
    def analyze_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        vid_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        curr_pos = 0
        debug=0
        start_pos=1
        if debug==1:
            while curr_pos < start_pos:
                ret, frame = video_capture.read()
                print("Skipping frame ", curr_pos)
                curr_pos += 1
                if not ret:
                    break
                

        csv_file, csv_writer = self.create_output_csv(video_path)
        if csv_writer is None:
            return

        for frame_count in range(curr_pos,vid_length):
            ret, frame = video_capture.read()
            if not ret:
                break

            milliseconds = frame_count * 1000 / fps
            delta = datetime.timedelta(milliseconds=milliseconds)
            hours, remainder = divmod(delta.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            millis = delta.microseconds // 1000
            timecode = f"{int(hours)}:{int(minutes)}:{int(seconds)}.{millis:03}"

            # преобразуем фрейм в тензор
            frame_tensor = torch.from_numpy(frame).float()
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
            print("\nFrame ", frame_count, "/", vid_length)

            # 1. анализ исходного тензора
            print("Original frame detection")
            orig_prediction = self.analyze_frame(frame_tensor)

            if len(orig_prediction.head()) != 1:
                self.save_csv_output(csv_writer, orig_prediction, None, None, frame_count, timecode, len(orig_prediction.head()))
                #frame_count += 1
                continue

            # 2. поворот тензора
            print("Aligning")
            frame_tensor, nan_flag_rot = self.fh.rotate_tensor(frame_tensor, orig_prediction.head())
            if nan_flag_rot:
                self.save_csv_output(csv_writer, orig_prediction, None, None, frame_count, timecode, 0)
                continue

            # 3. анализ исправленного тензора
            print("Aligned frame detection")
            orig_prediction = self.analyze_frame(frame_tensor)

            # 4. отзеркаливаем
            print("Mirroring")
            left_mirrored_tensor, right_mirrored_tensor, nan_flag_mir = self.fh.mirror_faces(frame_tensor, orig_prediction.head())
            if nan_flag_mir:
                self.save_csv_output(csv_writer, orig_prediction.head(), None, None, frame_count, timecode, 0)
                continue
            
            # 5. анализируем
            print("Left half detection")
            left_mirrored_prediction = self.analyze_frame(left_mirrored_tensor)
            print("Right half detection")
            right_mirrored_prediction = self.analyze_frame(right_mirrored_tensor)
           
            self.save_csv_output(csv_writer, orig_prediction.head(), left_mirrored_prediction.head(), right_mirrored_prediction.head(), frame_count, timecode, 1)

        video_capture.release()
        cv2.destroyAllWindows()
        csv_file.close()

    def create_output_csv(self, video_path):
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]  
            csv_path = os.path.join(os.path.dirname(video_path), f"{video_name}.csv")
            csv_file = open(csv_path, 'w', newline='')  
            csv_writer = csv.writer(csv_file,  delimiter='\t')
            csv_writer.writerow(["Frame", "Timecode", "FaceNumber", "FaceScore","Pitch", "Roll", "Yaw", 
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
            print("Failed to create CSV output file for video ", video_path, " : " f"{e}")
            return None, None

    def save_csv_output(self, csv_writer, orig_prediction, left_prediction, right_prediction, frame_number, timecode, face_number):
        try:
            if face_number != 1:
                csv_writer.writerow([frame_number, timecode, face_number]) # если лицо не обнаружено или больше 1 лица, записываем пустую строку
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

                row_to_write = [frame_number, timecode, face_number, fase_score] + pry_piece + wh_aus + lh_aus + rh_aus

                csv_writer.writerow(row_to_write)
        
        except Exception as e:
            print(f"Failed to write output for frame №{frame_number} in csv file: {e}")
        return
