import numpy as np
from numpy.core.defchararray import join
from numpy.lib.shape_base import vsplit

DOF = 7
DQ_MAX = np.array([2, 2, 2, 2, 2.5, 2.5, 2.5])
DDQ_MAX = np.array([5, 5, 5, 5, 5, 5, 5])


class MotionGenerator:
    def __init__(self, speed_factor, q_start, q_goal):
        assert speed_factor > 0 and speed_factor <= 1
        self.q_goal = q_goal
        self.q_start = q_start
        self.speed_factor = speed_factor
        self.dq_max = speed_factor * DQ_MAX
        self.ddq_max = speed_factor * DDQ_MAX
        self.dq_max_sync = np.zeros(DOF)
        self.delta_q = q_goal - q_start
        self.t1_sync = np.zeros(DOF)
        self.t2_sync = np.zeros(DOF)
        self.tf_sync = np.zeros(DOF)
        self.q1 = np.zeros(DOF)
        self.time = 0
        self.k_delta_q_motion_finished = 1e-6
        self.calculate_syncronized_values()

        #print(self.t1_sync)
        #print(self.t2_sync)
        #print(self.tf_sync)
        #print(self.q1)
        #print(self.dq_max_sync)
        #print(self.q_start)
        #print(self.delta_q)

    def __call__(self, time):
        self.time = time
        delta_q_d, motion_finished = self.calculate_desired_values()
        res = self.q_start + delta_q_d
        return res, motion_finished

    def calculate_desired_values(self):
        t = self.time
        delta_q_d = np.zeros(DOF)
        sign_delta_q = np.sign(self.delta_q)
        t_d = self.t2_sync - self.t1_sync
        delta_t2_sync = self.tf_sync - self.t2_sync
        joint_motion_finished = [False] * DOF

        for i in range(DOF):
            if np.abs(self.delta_q[i]) < self.k_delta_q_motion_finished:
                delta_q_d[i] = 0.0
                joint_motion_finished[i] = True
            else:
                if t < self.t1_sync[i]:
                    delta_q_d[i] = (
                        (-1.0 / (self.t1_sync[i] ** 3.0))
                        * self.dq_max_sync[i]
                        * sign_delta_q[i]
                        * (0.5 * t - self.t1_sync[i])
                        * (t ** 3.0)
                    )
                elif (t >= self.t1_sync[i]) and (t < self.t2_sync[i]):
                    delta_q_d[i] = (
                        self.q1[i]
                        + (t - self.t1_sync[i]) * self.dq_max_sync[i] * sign_delta_q[i]
                    )
                elif (t >= self.t2_sync[i]) and (t < self.tf_sync[i]):
                    #print("HERE", i)
                    # print("DEPENDS:")
                    # print(self.delta_q[i])
                    # print(delta_t2_sync[i])
                    # print(self.t1_sync[i])
                    # print(t_d[i])
                    # print(self.dq_max_sync[i])
                    # print(sign_delta_q[i])
                    #print(1.0 / (delta_t2_sync[i] ** 3.0))
                    #print(t - self.t1_sync[i] - 2.0 * delta_t2_sync[i] - t_d[i])
                    #print((t - self.t1_sync[i] - t_d[i]) ** 3.0)
                    #print(
                        #2.0 * t
                        #- 2.0 * self.t1_sync[i]
                        #- delta_t2_sync[i]
                        #- 2.0 * t_d[i]
                    #)
                    delta_q_d[i] = (
                        self.delta_q[i]
                        + 0.5
                        * (
                            1.0
                            / (delta_t2_sync[i] ** 3.0)
                            * (t - self.t1_sync[i] - 2.0 * delta_t2_sync[i] - t_d[i])
                            * ((t - self.t1_sync[i] - t_d[i]) ** 3.0)
                            + (
                                2.0 * t
                                - 2.0 * self.t1_sync[i]
                                - delta_t2_sync[i]
                                - 2.0 * t_d[i]
                            )
                        )
                        * self.dq_max_sync[i]
                        * sign_delta_q[i]
                    )
                    #print(delta_q_d[i])
                else:
                    delta_q_d[i] = self.delta_q[i]
                    joint_motion_finished[i] = True

        return delta_q_d, np.all(joint_motion_finished)

    def calculate_syncronized_values(self):
        dq_max_reach = self.dq_max
        t_f = np.zeros(DOF)
        delta_t_2 = np.zeros(DOF)
        t_1 = np.zeros(DOF)
        delta_t_2_sync = np.zeros(DOF)
        sign_delta_q = np.sign(self.delta_q)

        for i in range(DOF):
            if np.abs(self.delta_q[i]) > self.k_delta_q_motion_finished:
                crit = 1.5 * (self.dq_max[i] ** 2) / self.ddq_max[i]
                if np.abs(self.delta_q[i]) < crit:
                    dq_max_reach[i] = np.sqrt(
                        (4.0 / 3.0)
                        * self.delta_q[i]
                        * sign_delta_q[i]
                        * 0.5
                        * self.ddq_max[i]
                    )
                t_1[i] = 1.5 * dq_max_reach[i] / self.ddq_max[i]
                delta_t_2[i] = 1.5 * dq_max_reach[i] / self.ddq_max[i]
                t_f[i] = (
                    (t_1[i] / 2.0)
                    + (delta_t_2[i] / 2.0)
                    + np.abs(self.delta_q[i]) / dq_max_reach[i]
                )

        max_t_f = np.max(t_f)
        for i in range(DOF):
            if np.abs(self.delta_q[i]) > self.k_delta_q_motion_finished:
                a = 1.5 * self.ddq_max[i]
                b = -1.0 * max_t_f * self.ddq_max[i] ** 2
                c = np.abs(self.delta_q[i]) * self.ddq_max[i] ** 2
                delta = b ** 2 - 4 * a * c
                if delta < 0.0:
                    delta = 0.0
                self.dq_max_sync[i] = (-1.0 * b - np.sqrt(delta)) / (2.0 * a)
                self.t1_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max[i]
                delta_t_2_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max[i]
                self.tf_sync[i] = (
                    self.t1_sync[i] / 2.0
                    + 0.5 * delta_t_2_sync[i]
                    + np.abs(self.delta_q[i] / self.dq_max_sync[i])
                )
                self.t2_sync[i] = self.tf_sync[i] - delta_t_2_sync[i]
                self.q1[i] = (
                    self.dq_max_sync[i] * sign_delta_q[i] * (0.5 * self.t1_sync[i])
                )


if __name__ == "__main__":

    np.set_printoptions(precision=6, suppress=True)

    q_start = np.array(
        [
            3.379845005927556e-20,
            0.1,
            -7.671762790414419e-20,
            -1.2,
            -9.52880363982232e-19,
            1.6,
            0.7853981633974483,
        ]
    )
    q_goal = np.array(
        [
            -0.2348861429997454,
            0.9260749479828954,
            0.30295523230097565,
            -1.481000625331023,
            -0.3836450894030683,
            2.366284286732787,
            -0.5411521516948592,
        ]
    )
    gen = MotionGenerator(0.2, q_start, q_goal)
    t = 0
    while True:
        q, finished = gen(t)
        print(np.round(t, 3))
        print(q)
        if finished:
            break
        t += 1e-3
