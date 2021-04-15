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