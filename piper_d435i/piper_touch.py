import time
import math

import mujoco
import mujoco.viewer
import cv2
import numpy as np

m = mujoco.MjModel.from_xml_path('./agilex_piper/scene_touch.xml')
d = mujoco.MjData(m)

names_str_list = [name.decode('utf-8') for name in m.names.split(b'\x00') if name]

def get_touch_data_adr(name:str,names_str_list:list):
  touch_data_adr = []
  touch_data_name = []
  for i in range(len(names_str_list)):
    if name in names_str_list[i]:
        sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, names_str_list[i])
        touch_data_adr.append(m.sensor_adr[sensor_id])
        touch_data_name.append(names_str_list[i])
  return touch_data_adr, touch_data_name

link7_touch_A_adr, _ = get_touch_data_adr("link7_touch_A_point",names_str_list)
link7_touch_B_adr, _ = get_touch_data_adr("link7_touch_B_point",names_str_list)
link8_touch_A_adr, _ = get_touch_data_adr("link8_touch_A_point",names_str_list)
link8_touch_B_adr, _ = get_touch_data_adr("link8_touch_B_point",names_str_list)
# print(link7_touch_A_adr)

def get_touch_data(touch_data_adr,max):
  touch_data = np.zeros(len(touch_data_adr))
  for i in range(len(touch_data_adr)):
    adr = touch_data_adr[i]
    data = mujoco.mju_norm3(d.sensordata[adr:adr+3])
    touch_data[i] = mujoco.mju_clip(data, 0.0, max) / max
  return touch_data


with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    
    step_start = time.time()
    mujoco.mj_step(m, d)
    
    link7_touch_A_date = (get_touch_data(link7_touch_A_adr,3.0)*255).reshape((15,15))
    link7_touch_B_date = (get_touch_data(link7_touch_B_adr,3.0)*255).reshape((15,12))
    link8_touch_A_date = (get_touch_data(link8_touch_A_adr,3.0)*255).reshape((15,15))
    link8_touch_B_date = (get_touch_data(link8_touch_B_adr,3.0)*255).reshape((15,12))
    link7_touch_A = cv2.resize(link7_touch_A_date,(150,150))
    link7_touch_B = cv2.resize(link7_touch_B_date,(150,120))
    link8_touch_A = cv2.resize(link8_touch_A_date,(150,150))
    link8_touch_B = cv2.resize(link8_touch_B_date,(150,120))
    # cv2.imshow("link7_touch_A",link7_touch_A)
    # cv2.imshow("link7_touch_B",link7_touch_B)
    # cv2.imshow("link8_touch_A",link8_touch_A)
    # cv2.imshow("link8_touch_B",link8_touch_B)
    # 8B上 7A上
    link7_touch = cv2.vconcat([link7_touch_A, link7_touch_B])
    link7_touch = cv2.flip(link7_touch, 0)
    link8_touch = cv2.vconcat([link8_touch_B, link8_touch_A])
    cv2.imshow("link7_touch",link7_touch)
    cv2.imshow("link8_touch",link8_touch)
    cv2.waitKey(1)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)