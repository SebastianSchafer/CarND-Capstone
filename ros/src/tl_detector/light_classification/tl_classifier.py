from styx_msgs.msg import TrafficLight
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
# from numpy import random
from PIL import Image
from utils import label_map_util
import random
import os
from time import time
from tensorflow.python.client import device_lib

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        # Only used for printing out the light state and predictions
        self.light_map = {0: 'red',
                          1: 'yellow',
                          2: 'green',
                          4: 'unknown'}

        # Exported classifier model files
#         MODEL_PATH = '/capstone/ros/src/tl_detector/light_classification/ssd_inception_v2/frozen_inference_graph.pb'
#         LABELS_MAP_PATH = '/capstone/ros/src/tl_detector/light_classification/label_map_common.pbtxt'
        NUM_CLASSES = 3
        dirname = os.path.dirname(__file__)
        MODEL_PATH = os.path.join(dirname, 'ssd_inception_v2/frozen_inference_graph.pb')
        LABELS_MAP_PATH = os.path.join(dirname, 'label_map_common.pbtxt')

        # Label mappings
        label_map = label_map_util.load_labelmap(LABELS_MAP_PATH)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Load the trained classifier model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Create session here instead of instantiating during inference
        self.sess = tf.Session(graph = self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        dev_str = device_lib.list_local_devices()
        if '/gpu:' in str(dev_str):
            print('Found GPU to use for inference.')
        else:
            print('Tensorflow did not find a GPU for inference! Clasifying images will be very slow.')


    # simple image scaling to (nR x nC) size
    def scale(self, im, nR, nC):
        nR0 = len(im)     # source number of rows 
        nC0 = len(im[0])  # source number of columns 
        return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
                    for c in range(nC)] for r in range(nR)]

    def load_image_into_numpy_array(self, image):
        # Retrieve the current height and width of the image
        im_height, im_width = image.shape[:2]
        image_dim = image.reshape((im_height, im_width, 3)).astype(np.uint8)
        image_dim_resized = self.scale(image_dim, 512, 512)
        return image_dim_resized

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # return TrafficLight.UNKNOWN

        predicted_state = TrafficLight.UNKNOWN

        # Save file for debugging purposes
        # Image.fromarray(image).save('{0}.png'.format(random.random() * 100000))
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        t0 = time()
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, 
            self.detection_scores, 
            self.detection_classes, 
            self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # print('time for inference: ', time()-t0)
        # Results
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Confidence level threshold. Only classify if over this threshold.
        min_score_thresh = .5
        current_score = 0

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                class_name = self.category_index[classes[i]]['name']
                # Use only highest score; this only is reasonable if all lights are supposed to have smae color
                if scores[i] > current_score:
                    current_score = scores[i]
                    # print('{}'.format(class_name), scores[i])
                    if class_name == 'Green':
                        predicted_state = TrafficLight.GREEN
                    elif class_name == 'Red':
                        predicted_state = TrafficLight.RED
                    elif class_name == 'Yellow':
                        predicted_state = TrafficLight.YELLOW

        return predicted_state
