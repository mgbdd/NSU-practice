from PIL import Image
import math
import numpy as np
from torchvision import transforms 
import cv2

class image_handler:
    def __init__(self):
        self.x_start = 5                                    
        self.num_landmarks = 68        
        self.y_start = self.x_start + self.num_landmarks     

    def tensor_to_image(self, tensor):
        tensor = (tensor).byte()
        tensor = tensor.squeeze(0)
        tensor = tensor.permute(1, 2, 0)
        image = Image.fromarray(np.array(tensor).astype(np.uint8))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.asarray(image))
        return image

    def calc_face_angle(self, landmarks_df):
        landmark_pairs = [(39,42), (31, 35), (48, 54)]
        angles = []

        for pair in landmark_pairs:
            point1_index = pair[0]
            x1_col_index = self.x_start + point1_index
            y1_col_index = self.y_start + point1_index
            x1 = float(landmarks_df.iloc[0, x1_col_index])
            y1 = float(landmarks_df.iloc[0, y1_col_index])

            point2_index = pair[1]
            x2_col_index = self.x_start + point2_index
            y2_col_index = self.y_start + point2_index

            x2 = float(landmarks_df.iloc[0, x2_col_index])
            y2 = float(landmarks_df.iloc[0, y2_col_index])

            angle = math.atan2(y2 - y1, x2 - x1)
            angles.append(angle)
        average_angle = np.mean(angles)
        face_angle = math.degrees(average_angle)
        return face_angle
                                    
    def rotate_tensor(self, frame_tensor, landmarks):
        nan_flag_rot = False

        roll = self.calc_face_angle(landmarks)
        image = self.tensor_to_image(frame_tensor)

        if not np.isnan(roll):
            rotated_image = image.rotate(roll, resample=Image.BICUBIC, expand=True)
        else:
            nan_flag_rot = True
            print("Cannot see vertical alignment landmarks, skipping picture")
            return frame_tensor, nan_flag_rot
        
        rotated_image = cv2.cvtColor(np.asarray(rotated_image), cv2.COLOR_RGB2BGR)
        transform = transforms.ToTensor()
        rotated_tensor = transform(rotated_image).unsqueeze(0)
        return  rotated_tensor * 255, nan_flag_rot

    def mirror_faces(self, orig_tensor, landmarks):
        nan_flag_mir = False
        image = self.tensor_to_image(orig_tensor)
        width, height = image.size
        
        if np.isnan(landmarks.iloc[0, self.x_start + 8]) or np.isnan(landmarks.iloc[0, self.x_start + 27]):
            nan_flag_mir = True
            print("Cannot see middle line landmarks, skipping frame")
            left_mirrored_tensor = orig_tensor
            right_mirrored_tensor = orig_tensor
            return left_mirrored_tensor, right_mirrored_tensor, nan_flag_mir
            
        chin_x = int(landmarks.iloc[0, self.x_start + 8])
        nose_top_x = int(landmarks.iloc[0, self.x_start + 27]) 
        midpoint_x = (chin_x + nose_top_x) // 2

        left_half = image.crop((0, 0, midpoint_x, height))
        right_half = image.crop((midpoint_x, 0, width, height))  

        left_mirror = left_half.transpose(Image.FLIP_LEFT_RIGHT)
        right_mirror = right_half.transpose(Image.FLIP_LEFT_RIGHT)

        new_image1 = Image.new('RGB', (left_half.width + left_mirror.width, height))  
        new_image2 = Image.new('RGB', (right_half.width + right_mirror.width, height))  

        new_image1.paste(left_half, (0,0))
        new_image1.paste(left_mirror, (left_half.width, 0))

        new_image2.paste(right_mirror, (0, 0))
        new_image2.paste(right_half, (right_mirror.width, 0))

        new_image1 = cv2.cvtColor(np.asarray(new_image1), cv2.COLOR_RGB2BGR)
        transform = transforms.ToTensor()
        left_mirrored_tensor = transform(new_image1).unsqueeze(0)

        new_image2 = cv2.cvtColor(np.asarray(new_image2), cv2.COLOR_RGB2BGR)
        right_mirrored_tensor = transform(new_image2).unsqueeze(0)

        return left_mirrored_tensor * 255, right_mirrored_tensor * 255
