import numpy as np
import math
import matplotlib.pyplot as plt


class MobileRobot:
    MAX_LINEAR_VELOCITY = 5
    MIN_LINEAR_VELOCITY = -5
    MAX_ANGULAR_VELOCITY = np.pi / 1
    MIN_ANGULAR_VELOCITY = -np.pi / 1

    def __init__(self, init_x, init_y, init_theta):
        """
        初始化移动机器人
        :param init_x: 机器人在X坐标的初始位置（m）
        :param init_y: 机器人在Y坐标系的初始位置(m)
        :param init_theta: 机器人沿着X轴正方向的初始角度(rad)
        """
        self.d_estimation = 0.5
        self.d_min = 0.2
        self.d_max = 1.4
        self.x = init_x
        self.y = init_y
        self.theta = init_theta
        self.v = 0
        self.w = 0
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_theta = []
        self.v_list = []
        self.w_list = []
        self.d_estimation_list = []

    def update_trajectory(self, v, w, dt):
        """
        更新轨迹记录，将当前位置和速度添加到轨迹列表中
        :param v: 当前速度
        :param w: 当前角速度
        :param dt: 采样时间
        """
        v = self._limit_velocity(v, self.MAX_LINEAR_VELOCITY, self.MIN_LINEAR_VELOCITY)
        w = self._limit_velocity(w, self.MAX_ANGULAR_VELOCITY, self.MIN_ANGULAR_VELOCITY)

        self._update_position(v, w, dt)
        self.v = v
        self.w = w
        self.trajectory_x.append(self.x)
        self.trajectory_y.append(self.y)
        self.trajectory_theta.append(self.theta)
        self.v_list.append(v)
        self.w_list.append(w)

    def _update_position(self, v, w, dt):
        """
        根据速度和时间更新机器人的位置和角度
        :param v: 当前速度
        :param w: 当前角速度
        :param dt: 采样时间
        """
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt

    def _limit_velocity(self, value, max_value, min_value):
        """
        限制速度在最小值和最大值之间
        :param value: 当前速度或角速度
        :param max_value: 允许的最大速度或角速度
        :param min_value: 允许的最小速度或角速度
        :return: 限制后的速度或角速度
        """
        if value > max_value:
            return max_value
        if value < min_value:
            return min_value
        return value


class ControlParameter:
    def __init__(self):
        """
        控制参数需要满足以下条件：
        abs(w_d)_max+k2+k3/2*abs(v_d/d_star)_max <= MAX_ANGULAR_VELOCITY
        k1+d_max*abs(v_d/d_star)_max <= MAX_LINEAR_VELOCITY
        """
        self.k1 = 2.2
        self.k2 = np.pi/4
        self.k3 = np.pi/4
        self.gamma = 3

def safe_zero(num, epsilon=1e-10):
    """
    确保输入的数值0，不为0，而是接近于0
    :param num:需要保证接近于0的数值
    :param epsilon:趋近于0的程度
    :return:趋近于0的值
    """
    if num==0:
        num = epsilon
    return num

if __name__ == '__main__':
    dt = 0.01  # 相机采样时间（s）
    t = 100  # 采样总时间
    d_star = 1  # 参考坐标的原点与特征点平面之间的距离(m)
    robotX = 4  # 机器人在X坐标的初始位置（m）
    robotY = 2  # 机器人在Y坐标系的初始位置(m)
    robotTheta = 0  # 机器人沿着X轴正方向的初始角度(rad)
    desiredRobot = MobileRobot(5, 5, np.pi / 2)
    realRobot = MobileRobot(robotX, robotY, robotTheta)
    parameter = ControlParameter()  # 控制器参数配置
    for d_t in np.arange(0, t, dt):
        # 基于全局坐标系的误差
        error_X = (desiredRobot.x - realRobot.x) / d_star
        error_Y = (desiredRobot.y - realRobot.y) / d_star
        error_Theta = desiredRobot.theta - realRobot.theta
        # 相对于实际机器人坐标系的误差
        inertiaMatrix = np.array([[np.cos(realRobot.theta), np.sin(realRobot.theta), 0],
                                  [-np.sin(realRobot.theta), np.cos(realRobot.theta), 0],
                                  [0, 0, 1]])
        errorMatrix = np.array([[error_X], [error_Y], [error_Theta]])
        errorMatrix_ForRobot = np.dot(inertiaMatrix, errorMatrix)
        error_X_ForRobot = errorMatrix_ForRobot[0][0]
        error_Y_ForRobot = errorMatrix_ForRobot[1][0]
        error_Theta_ForRobot = safe_zero(errorMatrix_ForRobot[2][0])
        # 期望机器人速度
        desiredRobot_V = 2
        desiredRobot_W = 1.2*np.sin(d_t)
        # 机器人速度发布
        proj = (parameter.k3 * error_X_ForRobot * desiredRobot_V / d_star * np.cos(error_Theta_ForRobot)) / (
                1 + error_X_ForRobot ** 2 + error_Y_ForRobot ** 2)
        if (realRobot.d_estimation == realRobot.d_min and proj < 0) or (
                realRobot.d_estimation == realRobot.d_max and proj > 0):
            proj = 0
        realRobot.d_estimation = realRobot.d_estimation + parameter.gamma * proj * dt
        realRobot.d_estimation_list.append(realRobot.d_estimation)
        realRobotV = parameter.k1 * np.tanh(
            error_X_ForRobot) + realRobot.d_estimation * desiredRobot_V / d_star * np.cos(error_Theta_ForRobot)
        realRobotW = desiredRobot_W + parameter.k2 * np.tanh(error_Theta_ForRobot) + desiredRobot_V / d_star * (
                    np.sin(error_Theta_ForRobot) / error_Theta_ForRobot) * (
                                 parameter.k3 * error_Y_ForRobot / (1 + error_X_ForRobot ** 2 + error_Y_ForRobot ** 2))
        #控制机器人以上述速度运动，更新轨迹
        realRobot.update_trajectory(realRobotV, realRobotW, dt)
        desiredRobot.update_trajectory(desiredRobot_V, desiredRobot_W, dt)
    print(realRobot.d_estimation)
    plt.figure()
    plt.title('Robot Trajectory')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.plot(realRobot.trajectory_x,realRobot.trajectory_y,label='real Robot Trajectory')
    plt.plot(desiredRobot.trajectory_x,desiredRobot.trajectory_y,linestyle=':',label='desired Robot Trajectory')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(0,t,dt),np.array(desiredRobot.trajectory_x)-np.array(realRobot.trajectory_x),label='x error')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(0,t,dt),np.array(desiredRobot.trajectory_y)-np.array(realRobot.trajectory_y),label='y error')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(0,t,dt),np.array(desiredRobot.trajectory_theta)-np.array(realRobot.trajectory_theta),label='theta error')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(0,t,dt),realRobot.MAX_LINEAR_VELOCITY*np.ones(int(t/dt)),linestyle='--',label='MAX_LINEAR_VELOCITY')
    plt.plot(np.arange(0, t, dt), realRobot.MIN_LINEAR_VELOCITY*np.ones(int(t/dt)), linestyle='--',label='MIN_LINEAR_VELOCITY')
    plt.plot(np.arange(0,t,dt),realRobot.v_list,label='real Robot V')
    plt.plot(np.arange(0,t,dt),desiredRobot.v_list,label='desired Robot V')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(0,t,dt),realRobot.MAX_ANGULAR_VELOCITY*np.ones(int(t/dt)),linestyle='--',label='MAX_ANGULAR_VELOCITY')
    plt.plot(np.arange(0, t, dt), realRobot.MIN_ANGULAR_VELOCITY*np.ones(int(t/dt)), linestyle='--',label='MIN_ANGULAR_VELOCITY')
    plt.plot(np.arange(0,t,dt),realRobot.w_list,label='real Robot W')
    plt.plot(np.arange(0,t,dt),desiredRobot.w_list,label='desired Robot W')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(0,t,dt),realRobot.d_estimation_list,label='real Robot d_estimation')
    plt.legend()
    plt.show()




