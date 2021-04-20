import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pybullet as p

def plot_gaussian_2D(mu, sigma,ax=None,color=[0.7,0.7,0.],alpha=1.0, label='label'):
    if ax is None:
        fig,ax = plt.subplots()
    eig_val, eig_vec = np.linalg.eigh(sigma)
    std = np.sqrt(eig_val)*2
    angle = np.arctan2(eig_vec[1,0],eig_vec[0,0])
    ell = Ellipse(xy = (mu[0], mu[1]), width=std[0], height = std[1], angle = np.rad2deg(angle))
    ell.set_facecolor(color)
    ell.set_alpha(alpha)
    ell.set_label(label)
    ax.add_patch(ell)
    return

def Rotz(angle):
    A = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    return A

def compute_covariance(radius, ori):
    A = Rotz(ori)
    Sigma = np.diag(radius**2)
    Sigma = A.T.dot(Sigma).dot(A) 
    return Sigma

def create_primitives(shapeType=2, rgbaColor=[1, 1, 0, 1], pos = [0, 0, 0], radius = 1, length = 2, halfExtents = [0.5, 0.5, 0.5], baseMass=1, basePosition = [0,0,0]):
    visualShapeId = p.createVisualShape(shapeType=shapeType, rgbaColor=rgbaColor, visualFramePosition=pos, radius=radius, length=length, halfExtents = halfExtents)
    collisionShapeId = p.createCollisionShape(shapeType=shapeType, collisionFramePosition=pos, radius=radius, height=length, halfExtents = halfExtents)
    bodyId = p.createMultiBody(baseMass=baseMass,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=visualShapeId,
                      baseCollisionShapeIndex=collisionShapeId,    
                      basePosition=basePosition,
                      useMaximalCoordinates=True)
    return visualShapeId, collisionShapeId, bodyId


def get_joint_limits(robot_id, dof):
    limit = np.zeros((2, dof))
    for i in range(dof):
        limit[0,i] = p.getJointInfo(robot_id, i)[8]
        limit[1,i] = p.getJointInfo(robot_id, i)[9]
    return limit

def plot_pose_matrix(ax,T,label=None,scale=1):
    base = T[:3,-1]
    orn = T[:3,:3]
    ax.plot([base[0],base[0]+scale*orn[0,0]],[base[1],base[1]+scale*orn[1,0]],[base[2],base[2]+scale*orn[2,0]],c='r')
    ax.plot([base[0],base[0]+scale*orn[0,1]],[base[1],base[1]+scale*orn[1,1]],[base[2],base[2]+scale*orn[2,1]],c='g')
    ax.plot([base[0],base[0]+scale*orn[0,2]],[base[1],base[1]+scale*orn[1,2]],[base[2],base[2]+scale*orn[2,2]],c='b')
    
    if label is not None:
        ax.text(base[0],base[1],base[2],label)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def quat2Mat(quat):
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)