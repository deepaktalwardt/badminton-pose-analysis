import os
import glob
import tensorflow as tf
import cv2
import sys
import numpy as np
import time
import csv

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
 
        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
 
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
 
    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()
 
        print("Elapsed Time:", end_time-start_time)
 
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))
 
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

class DetectPlayers:

    def __init__(self, 
                input_folder, 
                output_folder, 
                od_model_path, 
                threshold=0.95, 
                crop_width_ratio=0.3, 
                crop_height_ratio=0.6,
                padding=100):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.crop_height_ratio = crop_height_ratio
        self.crop_width_ratio = crop_width_ratio
        self.padding = padding
        self.threshold = threshold

        # Create output folder
        if not os.path.exists(self.output_folder):
            try:
                os.makedirs(self.output_folder)
            except:
                print('Could not create output folder at: ' + self.output_folder)
                return

        # Object Detection API
        self.od = DetectorAPI(path_to_ckpt=od_model_path)
    
    def _crop_frame(self, frame):
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])

        crop_width_min = int(frame_width * self.crop_width_ratio)
        crop_width_max = int(frame_width - (frame_width * self.crop_width_ratio))

        crop_height_min = int(frame_height * self.crop_height_ratio)

        frame = frame[crop_width_min : crop_width_max, crop_height_min : frame_height]
        return frame
    
    def _blacken_frame(self, frame):
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])

        crop_width_min = int(frame_width * self.crop_width_ratio)
        crop_width_max = int(frame_width - (frame_width * self.crop_width_ratio))

        crop_height_min = int(frame_height * self.crop_height_ratio)
        
        mask = np.zeros(frame.shape, np.uint8)
        mask[crop_width_min : crop_width_max, crop_height_min : frame_height] = frame[crop_width_min : crop_width_max, crop_height_min : frame_height]
        # frame = frame[crop_width_min : crop_width_max, crop_height_min : frame_height]
        return mask
    
    def _detect_from_frame_and_save_entire(self, frame, file_name, csv_file):
        # frame = self._blacken_frame(frame)
        boxes, scores, classes, num = self.od.processFrame(frame)
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                x_min = max(box[0] - self.padding, 0)
                x_max = min(box[2] + self.padding, frame.shape[0])

                y_min = max(box[1] - self.padding, 0)
                y_max = min(box[3] + self.padding, frame.shape[1])

                mask = np.zeros(frame.shape, np.uint8)
                mask[x_min : x_max, y_min : y_max] = frame[x_min : x_max, y_min : y_max]

                record_name = file_name + str(i) + ".png"
                cv2.imwrite(self.output_folder + '\\' + record_name, mask)
                to_write = [record_name, x_min, y_min, x_max, y_max]
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(to_write)


    def _detect_from_frame_and_save_cropped(self, frame, file_name):
        frame = self._crop_frame(frame)
        boxes, scores, classes, num = self.od.processFrame(frame)
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                x_min = max(box[0] - self.padding, 0)
                x_max = min(box[2] + self.padding, frame.shape[0])

                y_min = max(box[1] - self.padding, 0)
                y_max = min(box[3] + self.padding, frame.shape[1])
       
                mask = frame[x_min : x_max, y_min : y_max]
                # mask = np.zeros(frame.shape, np.uint8)
                # mask[x_min : x_max, y_min : y_max] = frame[x_min : x_max, y_min : y_max]

                # # cv2.imshow("player", mask)
                # if area > max_human_area:
                #     max_human_area = area
                #     max_human = mask
                cv2.imwrite(self.output_folder + '\\' + file_name + str(i) + ".png", mask)

    def detect_players(self, cropped_frame=False):
        frame_list = glob.glob(self.input_folder + '/*.png')
        print('Frames found: ' + str(len(frame_list)))

        if cropped_frame:
            for frame_name in frame_list:
                file_name = os.path.basename(frame_name)
                frame = cv2.imread(frame_name)
                self._detect_from_frame_and_save_cropped(frame, file_name)
        else :
            csv_file = self.output_folder + '\\' + 'detection_info.csv'
            if os.path.isfile(csv_file):
                os.remove(csv_file)
                try:
                    os.makedirs(csv_file)
                except:
                    print("ERROR: Couldn't create CSV file")
                    return
            for frame_name in frame_list:
                file_name = os.path.basename(frame_name)
                frame = cv2.imread(frame_name)
                self._detect_from_frame_and_save_entire(frame, file_name, csv_file)
