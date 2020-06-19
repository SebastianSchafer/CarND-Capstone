from styx_msgs.msg import TrafficLight
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = load_model('model.h5')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # return TrafficLight.UNKNOWN

        predicted_state = self.model.predict_classes(image)
        return predicted_state
