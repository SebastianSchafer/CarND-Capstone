#!/usr/bin/env python

import numpy as np
from scipy.spatial import KDTree

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number
MAX_DECELERATION = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # rospy.Subscriber('/traffic_waypoint', Lane, self.traffic_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # current pose of the ego car
        self.pose = None

        # the base lane waypoints of the map
        self.base_waypoints = None

        # these 2d (x, y) waypoints are used to efficiently find the closest waypoint to the ego car
        self.waypoints_2d = None

        # this KDTree object is used to calculate closest waypoint to the ego vehicle
        self.waypoints_tree = None

        # 
        self.stopline_wp_index = None

        self.loop()

        # rospy.spin()

    def loop(self):
        rate = rospy.Rate(10) # rates >10Hz cause severe lag on Udacity workspace
        
        while not rospy.is_shutdown():
            if (self.pose and self.base_waypoints and self.waypoints_tree and self.stopline_wp_index):
                # get the closest waypoint
                closest_waypoint_index = self.get_closest_waypoint_index()
                self.publish_waypoints(closest_waypoint_index)
            
            rate.sleep()

    def get_closest_waypoint_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_index = self.waypoints_tree.query([x, y], 1)[1]

        # check if the closest waypoint is ahead or behind the ego vehcile
        closest_coords = self.waypoints_2d[closest_index]
        prev_coords = self.waypoints_2d[closest_index - 1]

        # the vectors ahead and behind the ego car
        closest_vect = np.array(closest_coords)
        prev_vect = np.array(prev_coords)
        pose_vect = np.array([x, y])

        dot_prod = np.dot((closest_vect - prev_vect), (pose_vect - closest_vect))

        if (dot_prod > 0.0):
            closest_index = (closest_index + 1) % len(self.waypoints_2d)

        return closest_index

    def publish_waypoints(self, closest_waypoint_index):
        front_lane = self.generate_front_lane(closest_waypoint_index)
        self.final_waypoints_pub.publish(front_lane)

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, base_waypoints):
        # TODO: Implement
        self.base_waypoints = base_waypoints

        if (not self.waypoints_2d):
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in base_waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def generate_front_lane(self, closest_waypoint_index):
        front_lane  = Lane()
        front_lane.header = self.base_waypoints.header
        front_lane.header.stamp = rospy.Time.now()
        end_waypoint_index = closest_waypoint_index + LOOKAHEAD_WPS
        ref_waypoints = self.base_waypoints.waypoints[closest_waypoint_index : end_waypoint_index]

        if (self.stopline_wp_index >= end_waypoint_index or self.stopline_wp_index == -1):
            front_lane.waypoints = ref_waypoints
        else:
            front_lane.waypoints = self.generate_decelerate_waypoints(ref_waypoints, closest_waypoint_index)

        return front_lane

    def generate_decelerate_waypoints(self, waypoints, closest_index):
        result_waypoints = []
        for i, waypoint in enumerate(waypoints):
            temp_waypoint = Waypoint()
            temp_waypoint.pose = waypoint.pose

            stop_index = max(self.stopline_wp_index - closest_index - 2, 0)
            distance = WaypointUpdater.distance(waypoints, i, stop_index)
            vel = math.sqrt(2 * MAX_DECELERATION * distance)

            if (vel < 1.0):
                vel = 0

            temp_waypoint.twist.twist.linear.x = min(vel, waypoint.twist.twist.linear.x)
            result_waypoints.append(temp_waypoint)

        return result_waypoints

    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i

        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
