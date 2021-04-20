import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from ocp_utils import quat2Mat

class CostModelQuadratic():
    def __init__(self, sys, Q = None, R = None, x_ref = None, u_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.Q, self.R = Q, R
        if Q is None: self.Q = np.zeros((self.Dx,self.Dx))
        if R is None: self.R = np.zeros((self.Du,self.Du))
        self.x_ref, self.u_ref = x_ref, u_ref
        if x_ref is None: self.x_ref = np.zeros(self.Dx)
        if u_ref is None: self.u_ref = np.zeros(self.Du)
            
    def set_ref(self, x_ref=None, u_ref=None):
        if x_ref is not None:
            self.x_ref = x_ref
        if u_ref is not None:
            self.u_ref = u_ref
    
    def calc(self, x, u):
        self.L = 0.5*(x-self.x_ref).T.dot(self.Q).dot(x-self.x_ref) + 0.5*(u-self.u_ref).T.dot(self.R).dot(u-self.u_ref)
        return self.L
    
    def calcDiff(self, x, u):
        self.Lx = self.Q.dot(x-self.x_ref)
        self.Lu = self.R.dot(u-self.u_ref)
        self.Lxx = self.Q.copy()
        self.Luu = self.R.copy()
        self.Lxu = np.zeros((self.Dx, self.Du))

class CostModelQuadraticOrientation():
    '''
    The quadratic cost model for the end-effector's orientation in quaternion
    '''
    def __init__(self, sys, W, ee_id, o_ref = None, R_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx,sys.Du
        self.W = W            
        self.ee_id = ee_id
        
        self.R_ref = R_ref #orientation of the target w.r.t. the world
        if self.R_ref is not None:
            self.W = self.R_ref.dot(self.W).dot(self.R_ref.T) #transform the cost coefficient to the object frame
        if o_ref is None: o_ref = np.array([1,0,0,0])
        self.set_ref(o_ref)


    def dQuatToDxJac(self,q):
        H = np.array([
            [-q[1] , q[0] , -q[3] , q[2]],
            [-q[2] , q[3] , q[0] , -q[1]],
            [-q[3] , -q[2] , q[1] , q[0]],
        ])
        return H
    
    def dist(self,x,y):
        dist = x.T @ y
        dist = 1 if dist > 1 else dist
        dist = -1 if dist < -1 else dist

        ac = np.arccos(dist)
        if ac < 0:
            ac -= np.pi
        return ac

    def logMap(self,base,y):
        temp = y - base.T @ y * base
        
        if(np.linalg.norm(temp)==0):
            return np.zeros(len(y))
        
        return self.dist(base,y) * temp / np.linalg.norm(temp)

    def set_ref(self,o_ref):
        self.o_ref = o_ref 
        
        if self.R_ref is not None:
            #o_ref is defined in the object coordinate system
            #Transform o_ref to the world coordinate system
            R_local = Rotation.from_quat(o_ref).as_matrix()
            R_world = self.R_ref.dot(R_local)
            o_world = Rotation.from_matrix(R_world).as_quat()
            self.o_ref = o_world
            
        self.o_ref = np.hstack(( self.o_ref[-1] , self.o_ref[:-1] ))

    def calc(self,x,u):
        _,orn = self.sys.compute_ee(x,self.ee_id)
        orn = np.hstack(( orn[-1] , orn[:-1] )) # XYZW to WXYZ
        orientation_error = 2 * self.dQuatToDxJac(self.o_ref) @ self.logMap(self.o_ref,orn)
        self.L = 0.5 * orientation_error.T @ self.W @ orientation_error
        return self.L

    def calcDiff( self,x,u):
        _,orn = self.sys.compute_ee(x,self.ee_id)
        orn = np.hstack(( orn[-1] , orn[:-1] )) # XYZW to WXYZ
        orientation_error = 2 * self.dQuatToDxJac(self.o_ref) @ self.logMap(self.o_ref,orn)

        self.J = self.sys.compute_Jacobian(x,self.ee_id)[-3:] # Rotational Jacobian
        self.Lx = self.J.T @ self.W @ orientation_error
        self.Lx = np.concatenate([ self.Lx , np.zeros(self.Dx//2)])
        self.Lu = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx,self.Dx))
        self.Lxx[:self.Dx//2,:self.Dx//2] = self.J.T @ self.W @ self.J
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))
        

class CostModelQuadraticTranslation():
    '''
    The quadratic cost model for the end effector, p = f(x)
    '''
    def __init__(self, sys, W, ee_id, p_ref = None, R_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.W = W
        self.p_ref = p_ref
        if p_ref is None: self.p_ref = np.zeros(3)
        self.ee_id = ee_id
        
        if R_ref is not None:
            self.R_ref = R_ref #orientation of the target w.r.t. the world
            self.W = self.R_ref.dot(self.W).dot(self.R_ref.T) #transform the cost coefficient to the object frame
        
    def set_ref(self, p_ref):
        self.p_ref = p_ref
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x, self.ee_id)
        self.L = 0.5*(p-self.p_ref).T.dot(self.W).dot(p-self.p_ref) 
        return self.L
    
    def calcDiff(self, x, u):
        p,_      = self.sys.compute_ee(x, self.ee_id)
        self.J   = self.sys.compute_Jacobian(x, self.ee_id)[:3] #Only use the translation Jacobian
        self.Lx  = self.J.T.dot(self.W).dot(p-self.p_ref)
        self.Lx = np.concatenate([self.Lx, np.zeros(int(self.Dx/2))])
        self.Lu  = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx, self.Dx))
        self.Lxx[:int(self.Dx/2), :int(self.Dx/2)] = self.J.T.dot(self.W).dot(self.J)
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))

class CostModelSum():
    def __init__(self, sys, costs):
        self.sys = sys
        self.costs = costs
        self.Dx, self.Du = sys.Dx, sys.Du
    
    def calc(self, x, u):
        self.L = 0
        for i,cost in enumerate(self.costs):
            cost.calc(x, u)
            self.L += cost.L
        return self.L
    
    def calcDiff(self, x, u):
        self.Lx = np.zeros(self.Dx)
        self.Lu = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx,self.Dx))
        self.Luu = np.zeros((self.Du,self.Du))
        self.Lxu = np.zeros((self.Dx,self.Du))
        for i,cost in enumerate(self.costs):
            cost.calcDiff(x, u)
            self.Lx += cost.Lx
            self.Lu += cost.Lu
            self.Lxx += cost.Lxx
            self.Luu += cost.Luu
            self.Lxu += cost.Lxu
            
class CostModelCollisionEllipsoid():
    '''
    The collision cost model between the end-effector and an ellipsoid obstacle
    '''
    def __init__(self, sys, p_obs, Sigma_obs, ee_id, w_obs = 1., d_thres = 1., R_obs = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.p_obs = p_obs #obstacle position
        self.Sigma_obs = Sigma_obs #obstacle ellipse covariance matrix       
        self.Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        
        
        if R_obs is not None:
            self.R_obs = R_obs #orientation of the obstacle w.r.t. the world
            self.Sigma_obs_inv = self.R_obs.dot(self.Sigma_obs_inv).dot(self.R_obs.T) #transform the cost coefficient to the object frame    

        self.w_obs = w_obs
        self.d_thres = d_thres
        self.obs_status = False
        self.ee_id = ee_id
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x, self.ee_id)
        self.normalized_d = (p-self.p_obs).T.dot(self.Sigma_obs_inv).dot(p-self.p_obs) 
        if self.normalized_d < self.d_thres:
            self.obs_status = True #very near to the obstacle
            self.L = 0.5*self.w_obs*(self.normalized_d-self.d_thres)**2
        else:
            self.obs_status = False
            self.L = 0
        return self.L
    
    def calcDiff(self, x, u, recalc = True):
        if recalc:
            self.calc(x, u)
        self.J   = self.sys.compute_Jacobian(x, self.ee_id)[:3]  #Only use the translation part
        p,_      = self.sys.compute_ee(x, self.ee_id)
        
        if self.obs_status:
            Jtemp = self.J.T.dot(self.Sigma_obs_inv).dot(p-self.p_obs)
            self.Lx = np.zeros(self.Dx)
            self.Lx[:int(self.Dx/2)]  = self.w_obs*Jtemp.dot(self.normalized_d-self.d_thres)
            self.Lxx = np.zeros((self.Dx, self.Dx))
            self.Lxx[:int(self.Dx/2), :int(self.Dx/2)] = self.w_obs*Jtemp.T.dot(Jtemp)
        else:
            self.Lx = np.zeros(self.Dx)
            self.Lxx = np.zeros((self.Dx, self.Dx))
        
        self.Lu  = np.zeros(self.Du)
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))
     
