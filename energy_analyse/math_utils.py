import numpy as np
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
# Math Helpers
# -----------------------------------------------------------------------------
def quat_conjugate(q):
    # q: [w, x, y, z] -> [w, -x, -y, -z]
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    # q1 * q2
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_rotate(q, v):
    # Rotate vector v by quaternion q
    # v' = q * v * q_conj
    q_v = np.array([0, v[0], v[1], v[2]])
    q_inv = quat_conjugate(q)
    return quat_mul(quat_mul(q, q_v), q_inv)[1:]

def quat_diff_yaw(q_a, q_b):
    # Return yaw difference or similar?
    # Logic from IsaacLab subtract_frame_transforms: 
    # relative_quat = q_b * inv(q_a) (local frame) or inv(q_a) * q_b
    # Here we need (Ref - Robot) in Robot Frame
    # delta_pos = inv(q_robot) * (pos_ref - pos_robot)
    # delta_rot = inv(q_robot) * q_ref
    # Returns delta_pos, delta_rot
    pass

def matrix_from_quat(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() # Scipy uses [x, y, z, w]
