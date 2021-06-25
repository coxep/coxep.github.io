```python
# import os
# os.environ['ROS_HOSTNAME'] = "10.210.232.185"
# os.environ['ROS_MASTER_URI'] = "http://bar562:11311"
```


```python
import numpy as np
import rospy
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500
```


```python
from tf2_ros import BufferClient
```


```python
def quat_to_rot(q):
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])


def make_transform(q, t):
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = quat_to_rot(q)
    tf[0, 3] = t[0]
    tf[1, 3] = t[1]
    tf[2, 3] = t[2]
    return tf
```


```python
# while not rospy.get_node_uri():
#     rospy.init_node('test_node')
#     rospy.sleep(0.5)
```


```python
# client = BufferClient('tf2_buffer_server')
# client.wait_for_server()
```


```python
# Camera intrinsics matrix (with scaling error)
K = np.float64([[578.2210083007812, 0.0, 319.5],
               [0.0, 578.2210083007812, 239.5],
               [0.0, 0.0, 1.0]])
K_inv = np.linalg.inv(K)
P = np.hstack([K, np.zeros([3, 1], dtype=np.float64)])

# "K" focal distance is really off, and should be scaled by about 0.9 to be more accurate
K_true = np.copy(K)
K_true[0, 0] *= 0.9
K_true[1, 1] *= 0.9
K_inv_true = np.linalg.inv(K_true)
P_true = np.hstack([K_true, np.zeros([3, 1], dtype=np.float64)])

# Positions of each spoiler on the robot (in base_footprint frame)
l_spoiler = np.float64([-0.274, 0.145, 0.347, 1.])
r_spoiler = np.float64([-0.267, -0.192, 0.347, 1.])

dock = np.mean([l_spoiler, r_spoiler], axis=0)

# Define the TRUE camera pose
# ==> rosrun tf tf_echo /depthcam_proximity_back_rgb_optical_frame /base_footprint
# At time 0.000
# - Translation: [-0.069, 0.712, 1.742]
# - Rotation: in Quaternion [0.689, 0.711, -0.076, 0.115]

t = np.float64([-0.069, 0.712, 1.742])
q = np.float64([0.689, 0.711, -0.076, 0.115])
q /= np.linalg.norm(q)

robot_to_camera_true = make_transform(np.float64(q), np.float64(t))

# Define the WRONG camera pose
# - Translation: [-0.083, 0.728, 1.731]
# - Rotation: in Quaternion [0.686, 0.710, -0.045, 0.155]
t = np.float64([-0.083, 0.728, 1.731])
q = np.float64([0.686, 0.710, -0.045, 0.155])
q /= np.linalg.norm(q)

robot_to_camera_false = make_transform(q, t)

# Define the mapping from a 3D point in the robot to an image pixel coordinate
robot_to_pixel_true = P_true.dot(robot_to_camera_true)

```


```python
dock
```




    array([-0.2705, -0.0235,  0.347 ,  1.    ])




```python
np.linalg.inv(robot_to_camera_true)
```




    array([[-0.02336869,  0.963007  , -0.26846123, -0.21961397],
           [ 0.99799348,  0.03827798,  0.05043618, -0.04625219],
           [ 0.05884655, -0.26674393, -0.96196921,  1.86973245],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])




```python
# Position of camera in the robot frame
c_pos = np.linalg.inv(robot_to_camera_true)[:3, 3]
np.linalg.norm(c_pos[:3] - l_spoiler[:3])
```




    1.5356592595415155




```python
np.linalg.norm(robot_to_camera_true.dot(l_spoiler)[:3])
```




    1.5356592595415144




```python
l_pix = robot_to_pixel_true.dot(l_spoiler)
l_pix /= l_pix[2]
l_pix
```




    array([355.33277601, 365.70623512,   1.        ])




```python
r_pix = robot_to_pixel_true.dot(r_spoiler)
r_pix /= r_pix[2]
r_pix
```




    array([236.68754494, 365.14668441,   1.        ])




```python
d_pix = robot_to_pixel_true.dot(dock)
d_pix /= d_pix[2]
d_pix
```




    array([296.38856138, 365.42824436,   1.        ])



Given the TRUE pixel coordinates, project a camera-frame ray using both the correct and incorrect intrinsics.


```python
# Solve for the distances (correct and incorrect from the camera to the robot)
c_pos = np.linalg.inv(robot_to_camera_true)[0:3, 3]
l_dist_true = np.linalg.norm(robot_to_camera_true.dot(l_spoiler)[:3])
r_dist_true = np.linalg.norm(robot_to_camera_true.dot(r_spoiler)[:3])
d_dist_true = np.linalg.norm(robot_to_camera_true.dot(dock)[:3])

l_dist_false = l_dist_true / 0.9
r_dist_false = r_dist_true / 0.9
d_dist_false = d_dist_true / 0.9

# Given l_pix and r_pix, project the pixels to 3d rays, and compare the results with and without scaling to the correct height
l_ray_true = K_inv_true.dot(l_pix)
l_ray_true /= np.linalg.norm(l_ray_true)
l_ray_true *= l_dist_true

r_ray_true = K_inv_true.dot(r_pix)
r_ray_true /= np.linalg.norm(r_ray_true)
r_ray_true *= r_dist_true

d_ray_true = K_inv_true.dot(d_pix)
d_ray_true /= np.linalg.norm(d_ray_true)
d_ray_true *= d_dist_true

l_ray_false = np.copy(l_ray_true)
l_ray_false[2] /= 0.9

r_ray_false = np.copy(r_ray_true)
r_ray_false[2] /= 0.9

d_ray_false = np.copy(d_ray_true)
d_ray_false[2] /= 0.9
```


```python
# Transform the ray to the camera frame
l_bad_proj = np.linalg.inv(robot_to_camera_false).dot(np.hstack([l_ray_false, 1.0]))[:3]
r_bad_proj = np.linalg.inv(robot_to_camera_false).dot(np.hstack([r_ray_false, 1.0]))[:3]
d_bad_proj = np.linalg.inv(robot_to_camera_false).dot(np.hstack([d_ray_false, 1.0]))[:3]
```


```python
l_spoiler
```




    array([-0.274,  0.145,  0.347,  1.   ])




```python
def process_resize_position(tag, cam, z_ref=0.32, old=False):
    
    new_pt = np.zeros([3], dtype=np.float64)
    
    if old:
        scale = (cam[2] - z_ref)/(cam[2] - tag[2])
        
        new_pt[0] = tag[0] * scale
        new_pt[1] = tag[1] * scale
        new_pt[2] = z_ref
        
    else:
        scale = (z_ref - tag[2]) / (cam[2] - tag[2])
        new_pt[0] = tag[0] + scale * (cam[0] - tag[0])
        new_pt[1] = tag[1] + scale * (cam[1] - tag[1])
        new_pt[2] = tag[2] + scale * (cam[2] - tag[2])

    return new_pt
```


```python
bad_c_pos = np.linalg.inv(robot_to_camera_false)[:3, 3]

plt.scatter(l_spoiler[1], l_spoiler[2], c='k', s=0.1)
plt.scatter(r_spoiler[1], r_spoiler[2], c='k', s=0.1)
plt.scatter(r_bad_proj[1], r_bad_proj[2], c='red', s=0.1)
plt.scatter(l_bad_proj[1], l_bad_proj[2], c='red', s=0.1)
plt.scatter(d_bad_proj[1], d_bad_proj[2], c='red', s=0.1)

# Draw a line from the "bad" camera pose to the left projected spoiler
plt.plot([bad_c_pos[1], l_bad_proj[1]], [bad_c_pos[2], l_bad_proj[2]], c='red', linewidth=0.1)
plt.plot([bad_c_pos[1], r_bad_proj[1]], [bad_c_pos[2], r_bad_proj[2]], c='red', linewidth=0.1)

# Scale each fiducial the old way
l_proj_old = process_resize_position(l_bad_proj, bad_c_pos, old=True)
d_proj_old = process_resize_position(d_bad_proj, bad_c_pos, old=True)

l_proj_new = process_resize_position(l_bad_proj, bad_c_pos, old=False)
r_proj_new = process_resize_position(r_bad_proj, bad_c_pos, old=False)
d_proj_new = process_resize_position(d_bad_proj, bad_c_pos, old=False)

# Old process for performing projections
plt.scatter(l_proj_old[1], l_proj_old[2], c='green', s=0.1)
plt.scatter(r_proj_old[1], r_proj_old[2], c='green', s=0.1)

# New process for performing projections
plt.scatter(l_proj_new[1], l_proj_new[2], c='blue', s=0.1)
plt.scatter(r_proj_new[1], r_proj_new[2], c='blue', s=0.1)

# Projection of dock connector
plt.scatter(d_proj_new[1], d_proj_new[2], c='blue', s=0.1)

# Plot the camera coordinate
# plt.scatter(-0.04, 1.819377)
fig = plt.gcf()
# fig.size(2, 8)
plt.axis('scaled')
plt.xlim(-.25, .25)
plt.ylim(0, 2.0)
plt.title('bad news')

plt.show()
```


    
![png](docking_geometry_tests_files/docking_geometry_tests_19_0.png)
    



```python
# Compute errors associated with each process
# 1) Old (buggy) process, without projecting dock to correct height
error_orig = np.abs(np.mean([l_proj_old[1], r_proj_old[1]]) - d_bad_proj[1])
print("Original Error: " + str(error_orig))

# 2) Current (1.20) process, no projection of dock connector to correct height
error_current = np.abs(np.mean([l_proj_new[1], r_proj_new[1]]) - d_bad_proj[1])
print("Current Error: " + str(error_current))

# 3) Ideal process, with projecting dock to correct height
error_ideal = np.abs(np.mean([l_proj_new[1], r_proj_new[1]]) - d_proj_new[1])
print("Future Error: " + str(error_ideal))


```

    Original Error: 0.0027672924353168427
    Current Error: 0.014365889132082713
    Future Error: 0.001483242209739867



```python
error
```




    0.001483242209739867




```python
r_proj_new[1] - r_proj_old[1]
```




    -0.01901844281489562




```python
np.mean([l_proj_new[1], l_proj_new[1]])
```




    0.12551226774553792




```python
d_bad_proj[1]
```




    -0.01637689879443338




```python
np.mean([l_proj_old[1], r_proj_old[1]])
```




    -0.013609606359116538




```python
d_proj_new[1]
```




    -0.03222603013625596




```python
np.mean([l_proj_new[1], r_proj_new[1]])
```




    -0.030742787926516094




```python

```
