import math
import numpy as np
import random
import matplotlib.pyplot as plt  # clf()清图  # cla()清坐标轴  # close()关窗口
from scipy import interpolate

# 常量
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
S = 0.0572555  # 特征面积
g = 9.81

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

Ma2 = np.array([[0.4, 39.056, 0.4604, 39.072],
                [0.6, 40.801, 0.4682, 39.735],
                [0.8, 41.372, 0.4635, 39.242],
                [0.9, 42.468, 0.4776, 40.351]])


def load_atm(path):
    file = open(path)
    atm_str = file.read().split()
    atm = []
    for _ in range(0, len(atm_str), 3):
        atm.append([float(atm_str[_]), float(atm_str[_ + 1]), float(atm_str[_ + 2])])
    return np.array(atm)


class MISSILE_TrEN:
    def __init__(self, missile=None, target=None, k=None):
        if k is None:
            k=[3.0, 1.5, 1.0]
        if missile is None:
            missile = [0., 300., 180. / RAD, 20000., 10000, 200]  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
        if target is None:
            target = [0., 0., -60.]  # 目标x,y,落角

        self.Y = np.array(missile)
        self.xt, self.yt, self.qt = target[0], target[1], target[2] / RAD  # 目标信息

        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force
        Rx = self.xt - self.Y[3]
        Ry = self.yt - self.Y[4]
        self.R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        self.q = math.atan2(Ry, Rx)  # 弹目视线角
        self.Rdot = 0.
        self.qdot = 0.
        self.ac = 0.  # 制导指令
        self.am = 0.  # 制导指令

        # 历史制导指令
        self.ac_1 = 0.
        self.ac_2 = 0.

        self.alpha = 0.  # 攻角
        # self.tgo = (1 + (self.Y[2] - self.q) ** 2 / 10) * self.R / self.Y[1]  # T_go = (1-(theta-lambda)^2/10)*R_go/V

        # 创建插值函数
        atm = load_atm('atm2.txt')  # 大气参数
        self.f_ma = interpolate.interp1d(atm[:, 0], atm[:, 2], 'linear')
        self.f_rho = interpolate.interp1d(atm[:, 0], atm[:, 1], 'linear')

        self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1] * k[0], 'linear')
        self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2] * k[1], 'linear')
        self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3] * k[2], 'linear')

        # 全弹道历史信息
        self.reY, self.reac, self.ream, self.reab, self.reR = [], [], [], [], []

    def modify(self, missile=None):  # 修改导弹初始状态
        if missile is None:
            missile = [0.,
                       random.uniform(100, 200),
                       random.uniform(-180, -135) / RAD,
                       random.uniform(5000., 1000),
                       random.uniform(5000, 10000),
                       100]  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
        self.Y = np.array(missile)
        Rx = self.xt - self.Y[3]
        Ry = self.yt - self.Y[4]
        self.R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        self.q = math.atan2(Ry, Rx)  # 弹目视线角

        self.ac_1 = 0.
        self.ac_2 = 0.
        # self.tgo = (1 + (self.Y[2] - self.q) ** 2 / 10) * self.R / self.Y[1]  # T_go = (1-(theta-lambda)^2/10)*R_go/V

        # k = self.k
        # self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1] * k, 'linear')
        # self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2] * k, 'linear')
        # self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3] * k, 'linear')

        self.reY, self.reac, self.ream, self.reab, self.reR = [], [], [], [], []
        return self.collect()

    def terminate(self):
        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force
        self.R = 0.  # 弹目距离range
        self.q = 0.  # 弹目视线角
        self.Rdot = 0.
        self.qdot = 0.
        self.ac = 0.  # 制导指令
        self.alpha = 0.  # 攻角

    def get_ma(self, y, v):  # 计算马赫数
        y = max(0, y)
        sonic = self.f_ma(y)
        return v / sonic

    def get_rho(self, y):  # 计算空气密度
        y = max(0, y)
        return self.f_rho(y)

    def get_clalpha(self, ma):
        return self.f_clalpha(max(min(ma, 0.9), 0.4))

    def get_cd0(self, ma):
        return self.f_cd0(max(min(ma, 0.9), 0.4))

    def get_cdalpha(self, ma):
        return self.f_cdalpha(max(min(ma, 0.9), 0.4))

    def dery(self, Y):  # 右端子函数
        v = Y[1]
        theta = Y[2]
        m = Y[5]
        dy = np.array(Y)
        dy[0] = 1  # t
        dy[1] = - self.X / m - g * math.sin(theta)  # v
        dy[2] = (self.L - m * g * math.cos(theta)) / (v * m)  # theta
        dy[3] = v * math.cos(theta)  # x
        dy[4] = v * math.sin(theta)  # y
        dy[5] = 0.
        return dy

    def step(self, action=0):
        h = 0.01
        if self.Y[0] < 400:
            t = self.Y[0]
            v = self.Y[1]  # 速度
            theta = self.Y[2]  # 弹道倾角
            x = self.Y[3]  # 横向位置
            y = self.Y[4]  # 纵向位置
            m = self.Y[5]  # 弹重
            RHO = self.get_rho(y)  # 大气密度
            ma = self.get_ma(y, v)  # 马赫数

            Q = 0.5 * RHO * v ** 2  # 动压

            Rx = self.xt - x
            Ry = self.yt - y
            vx = -v * math.cos(theta)  # x向速度
            vy = -v * math.sin(theta)  # y向速度
            self.R = R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
            self.qdot = qdot = (Rx * vy - Ry * vx) / R ** 2
            self.q = math.atan2(Ry, Rx)  # 弹目视线角
            self.Rdot = (Rx * vx + Ry * vy) / R
            # self.tgo = tgo = (1 + (theta - q) ** 2 / 10) * R / v

            if R < 2:
                print("弹目最小距离={:.1f}".format(self.R))
                return True
            elif y < 0:
                print("弹已落地, 弹目距离={:.1f}".format(R))
                return True
            # elif Rdot >= 0:
            #     print("逐渐远离目标...")

            # 实际项目中，飞行系数插值的三个维度分别为马赫数mach，舵偏角delta，攻角alpha
            cl_alpha = self.get_clalpha(ma)

            # 制导指令
            m_max = 3 * g
            self.ac = ac = np.clip(3 * v * qdot +
                                   math.cos(theta) * g +
                                   np.sign(math.cos(theta)) * action, -m_max, m_max)

            # 自动驾驶仪二阶动力学 xi=0.707, w=2Hz, h=0.01
            if t > 2 * h:
                self.am = am = 0.01445 * ac + 1.823 * self.ac_1 - 0.8372 * self.ac_2
            else:
                self.am = am = ac
            self.ac_2 = self.ac_1
            self.ac_1 = am

            a_max = 20. / RAD
            self.alpha = alpha = np.clip((m * am) / (Q * S * cl_alpha), -a_max, a_max)  # 使用了sin(x)=x的近似，在10°以内满足这一关系

            cd = self.get_cd0(ma) + self.get_cdalpha(ma) * alpha ** 2  # 阻力系数
            cl = cl_alpha * alpha  # 升力系数

            self.X = cd * Q * S  # 阻力
            self.L = cl * Q * S  # 升力

            def rk4(func, Y, h=0.01):
                k1 = h * func(Y)
                k2 = h * func(Y + 0.5 * k1)
                k3 = h * func(Y + 0.5 * k2)
                k4 = h * func(Y + k3)
                output = Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return output

            self.Y = rk4(self.dery, self.Y, h)

            self.reY.append(self.Y)
            self.reac.append(self.ac)
            self.ream.append(self.am)
            self.reab.append(action)
            self.reR.append(self.R)
            return False
        else:
            print("超时！未击中目标！")
            return True

    def collect(self):
        t = self.Y[0]  # 时间
        v = self.Y[1]  # 速度
        theta = self.Y[2]  # 弹道倾角
        r = self.R  # 弹目距离
        q = self.q  # 弹目视线角
        x = self.Y[3]  # 弹横向位置
        y = self.Y[4]  # 弹纵向位置
        return v, theta, r, q, x, y, t

    def plot_data(self, figure_num=0):
        reY = np.array(self.reY)
        reac = np.array(self.reac)
        ream = np.array(self.ream)
        reab = np.array(self.reab)
        reR = np.array(self.reR)

        plt.figure(figure_num)
        plt.ion()
        plt.clf()
        # 弹道曲线
        plt.subplots_adjust(hspace=0.6)
        plt.subplot(2, 2, 1)
        plt.plot(reY[:, 3] / 1000, reY[:, 4] / 1000, 'k-')
        plt.xlabel('Firing Range (km)')
        plt.ylabel('altitude (km)')
        plt.title('弹道曲线')
        plt.grid()

        # 速度曲线
        plt.subplot(2, 2, 2)
        plt.plot(reY[:, 0], reY[:, 1], 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('速度')
        plt.grid()

        # 过载指令
        plt.subplot(2, 2, 3)
        plt.plot(reY[:, 0], reac, 'k-')
        plt.plot(reY[:, 0], ream, 'r-')
        plt.xlabel('time (s)')
        plt.ylabel('ac')
        plt.title('过载指令')
        plt.grid()

        # 偏置项
        plt.subplot(2, 2, 4)
        plt.plot(reY[:, 0], reab, 'k-')
        plt.xlabel('Time (s)')
        plt.ylabel('action')
        plt.title('偏置项')
        plt.grid()

        plt.pause(0.1)


if __name__ == '__main__':
    mis = MISSILE_TrEN()
    for i in range(100):
        mis.modify()
        done = False
        while done is False:
            done = mis.step(0)  # 单步运行
        mis.plot_data(2)
        print(mis.Y[0])
