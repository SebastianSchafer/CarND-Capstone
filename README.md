
## Udacity Self-driving car ND system integration project

### Team members
* Sebastian Schafer
* Sai Krishna Chada
* Hamidreza Mirkhani
* Eric Lok

### Project overview
This is the final project of the Udacit self-drivin car [ND](http://udacity.com/drive). This is a system-integration project, with the intent of completing a compete, ROS-based software suite that can safely drive both a [simulated](https://github.com/udacity/CarND-Capstone/releases) car around a track as well as guiding a real car. Unfortunately, the testing on Udacitys' Carla self-driving car (not to be confused with the autonomous vehicle simulator [carla](http://carla.org)) was not available due to Covid19 related restrictions when this project was finished, so this description focuses on the simulator implementation.
The [original](https://github.com/udacity/CarND-Capstone) Udacity repo contains most of the framework necessary for this implementation, including several ROS nodes, launch files, and [Autoware](https://github.com/Autoware-AI/autoware.ai) libraries.
![system](gfx/system_overview.png)
As part of this project, we implemented one ROS node for each of the Perception, Planning, and Control sections of the system.

### Perception: Traffic light detection node
The traffic light detection was implemented using a single deep neural network and transfer learning. The traffic light detection node receives pose and camera image data and returns the predicted traffic light state, whcih is then used to determine whether the car should proceed or slow down.  We chose and end-to-end network implementation using the SSD inception_v2 [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models) pretrained on the [COCO](https://cocodataset.org/#home) dataset. While one might prefer an approach of chained or segragated detectors in a real-world application, with an initial object detector returning bounding boxes around detected objects, followed by a dedicated light detection of only the area of the detected traffic light, this did not make much sense in the current project, as there were no other objects to detect. For transfer learning, we used [labelled](https://drive.google.com/drive/folders/1NXqHTnjVC1tPjAB5DajGc30uWk5VPy7C) data from a [team](https://github.com/marcomarasca/SDCND-SuperAI-Capstone/blob/master/README.md#traffic-light-detection) doing the project in 2018; the labelling process can be time-consuming wihtout adding much learning, so we were happy to find a good dataset available.
While the training process is fairly straightforward using the tensorflow-api, a few steps are crucial to get the model working with good inference performance. We chose to train the model on Google Colab, which provides an even more convenient environment for tasks like this than AWS EC2. One drawback is that the project requires a tensorflow version of 1.3.0, which is not compatible with the oldest version available on Colab (1.15). Fortunately, tensorflow [provides](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md) an api to export models to other tensorflow versions for inference. We initially found that the model needed ~2s for inference on a workspace with a Nvidia K80 GPU, which clearly would not be sufficient to run the model and is also not consistent with expected inference times on the order of [50ms](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models). While we could not confirm for certain, it seems that his might be related to an [issue](https://github.com/tensorflow/models/issues/3270) of frozen models not properly assigning an available gpu when used with an old version of tensorflow. While even trying to assign the gpu using `tf.device('gpu:0')` failed, intalling __only__ tensorflow-gpu 1.3 yielded a usable inference time of about 60ms.

### Planning: Waypoint updater node
...

### Control: Drive-by-wire node
...

### Summary
... Add link to either video or embed gif?


### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
