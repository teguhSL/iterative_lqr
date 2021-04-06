import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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