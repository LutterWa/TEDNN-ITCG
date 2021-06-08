import sys
import numpy as np
import scipy.io
from missile import MISSILE
from missile_TrEN import MISSILE_TrEN

pi = 3.141592653589793
RAD = 180 / pi
dict_launch = {"state": [], "time": []}
TrEN = True  # 迁移-集成学习标志

if TrEN is True:
    missile = MISSILE_TrEN(k=[3.0, 1.5, 1.0])
else:
    missile = MISSILE(k=float(sys.argv[1]))
    print("code is {}, k={}, data_set={}".format(sys.argv[0], sys.argv[1], sys.argv[2]))

for launch_time in range(int(1e3)):
    missile.modify()
    print("========", launch_time + 1, "========")
    step = []
    runtime = np.array([])
    done = False
    while done is False:
        done = missile.step(action=0)
        v, theta, r, q, x, y, t = missile.collect()
        runtime = np.append(runtime, t)
        step.append([v, theta, r, q, x, y])
    time = t * np.ones([runtime.shape[0]]) - runtime
    dict_launch["state"].append(step)
    dict_launch["time"].append(time)

flight_data = {'flight_data': dict_launch}
if TrEN is True:
    scipy.io.savemat('flight_data_TrEN.mat', flight_data)
else:
    scipy.io.savemat('flight_data_{}.mat'.format(int(sys.argv[2])), flight_data)
