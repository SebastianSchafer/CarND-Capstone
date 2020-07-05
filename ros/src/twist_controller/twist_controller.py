from __future__ import division, print_function
import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                        wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        # init yaw controller, min_speed=0.1 given in project description
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # PID param values, mentioned in project description; possibly fine-tune:
        kp = 0.4
        ki = 0.2
        kd = 0.02
        min_throttle = 0.
        max_throttle = 0.3 # based on dbw_test rosbag
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)

        tau = 0.5 # for lp filter, given
        ts = 1/50.
        self.v_lpf = LowPassFilter(tau, ts)

        # need parameters to calculate accel/decel:
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.prev_t = rospy.get_time()


    def control(self, linear_v, angular_v, current_v, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # reset controller if dbw disabled to avoid accum. error
        if dbw_enabled == False:
            self.throttle_controller.reset()
            return 0, 0, 0

        current_v = self.v_lpf.filt(current_v)

        steer = self.yaw_controller.get_steering(linear_v, angular_v, current_v)

        delta_v = linear_v - current_v
        self.prev_v = current_v
        current_t = rospy.get_time()
        delta_t = current_t - self.prev_t
        self.prev_t = current_t

        throttle = self.throttle_controller.step(delta_v, delta_t)
        brake = 0 # set to 700Nm during init

        # Don't move below min_v, value given in project description:
        if current_v < 0.1 and linear_v == 0:
            throttle = 0
            brake = 700
        # elif below min throttle (given), and negative delta_v (i.e. want to brake)
        elif delta_v < 0 and throttle < 0.1:
            throttle = 0
            decel_actual = max(delta_v, self.decel_limit)
            if abs(decel_actual) < self.brake_deadband:
                decel_actual = 0
            # Assume that brake torque is calculated at wheel circumference (i.e. not at brake pad position)
            brake = abs(decel_actual) * self.vehicle_mass * self.wheel_radius

        
        return throttle, brake, steer
