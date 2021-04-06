import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv

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
            
class CostModelQuadraticTranslation():
    '''
    The quadratic cost model for the end effector, p = f(x)
    '''
    def __init__(self, sys, W, p_ref = None):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.W = W
        self.p_ref = p_ref
        if p_ref is None: self.p_ref = np.zeros(3)
            
    def set_ref(self, p_ref):
        self.p_ref = p_ref
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x)
        self.L = 0.5*(p-self.p_ref).T.dot(self.W).dot(p-self.p_ref) 
        return self.L
    
    def calcDiff(self, x, u):
        self.J   = self.sys.compute_Jacobian(x)
        p,_      = self.sys.compute_ee(x)
        self.Lx  = self.J.T.dot(self.W).dot(p-self.p_ref)
        self.Lx = np.concatenate([self.Lx, np.zeros(self.Dx/2)])
        self.Lu  = np.zeros(self.Du)
        self.Lxx = np.zeros((self.Dx, self.Dx))
        self.Lxx[:self.Dx/2, :self.Dx/2] = self.J.T.dot(self.W).dot(self.J)
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
    def __init__(self, sys, p_obs, Sigma_obs, w_obs = 1., d_thres = 1.):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.p_obs = p_obs #obstacle position
        self.Sigma_obs = Sigma_obs #obstacle ellipse covariance matrix
        self.Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        self.w_obs = w_obs
        self.d_thres = d_thres
        self.obs_status = False
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x)
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
        self.J   = self.sys.compute_Jacobian(x)
        p,_      = self.sys.compute_ee(x)
        
        if self.obs_status:
            Jtemp = self.J.T.dot(self.Sigma_obs_inv).dot(p-self.p_obs)
            self.Lx  = self.w_obs*Jtemp.dot(self.normalized_d-self.d_thres)
            self.Lx = np.concatenate([self.Lx, np.zeros(self.Dx/2)])
            self.Lxx = np.zeros((self.Dx, self.Dx))
            #self.Lxx[:self.Dx/2, :self.Dx/2] = -self.w_obs*self.J.T.dot(self.Sigma_obs_inv).dot(self.J)
            self.Lxx[:self.Dx/2, :self.Dx/2] = self.w_obs*Jtemp.T.dot(Jtemp)
            #self.Lxx = np.eye(self.Dx)
        else:
            self.Lx = np.zeros(self.Dx)
            self.Lxx = np.zeros((self.Dx, self.Dx))
        
        self.Lu  = np.zeros(self.Du)
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))
            
class CostModelCollisionEllipsoidOld():
    '''
    The collision cost model between the end-effector and an ellipsoid obstacle
    '''
    def __init__(self, sys, p_obs, Sigma_obs, w_obs = 1., d_thres = 1.):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.p_obs = p_obs #obstacle position
        self.Sigma_obs = Sigma_obs #obstacle ellipse covariance matrix
        self.Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        self.w_obs = w_obs
        self.d_thres = d_thres
        self.obs_status = False
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x)
        self.normalized_d = (p-self.p_obs).T.dot(self.Sigma_obs_inv).dot(p-self.p_obs) 
        if self.normalized_d < self.d_thres:
            self.obs_status = True #very near to the obstacle
            self.L = -0.5*self.w_obs*self.normalized_d
        else:
            self.obs_status = False
            self.L = 0
        return self.L
    
    def calcDiff(self, x, u, recalc = True):
        if recalc:
            self.calc(x, u)
        self.J   = self.sys.compute_Jacobian(x)
        p,_      = self.sys.compute_ee(x)
        
        if self.obs_status:
            Jtemp = self.J.T.dot(self.Sigma_obs_inv).dot(p-self.p_obs)
            self.Lx  = -self.w_obs*Jtemp
            self.Lx = np.concatenate([self.Lx, np.zeros(self.Dx/2)])
            self.Lxx = np.zeros((self.Dx, self.Dx))
            #self.Lxx[:self.Dx/2, :self.Dx/2] = -self.w_obs*self.J.T.dot(self.Sigma_obs_inv).dot(self.J)
            self.Lxx[:self.Dx/2, :self.Dx/2] = self.w_obs*Jtemp.T.dot(Jtemp)
            #self.Lxx = np.eye(self.Dx)
        else:
            self.Lx = np.zeros(self.Dx)
            self.Lxx = np.zeros((self.Dx, self.Dx))
        
        self.Lu  = np.zeros(self.Du)
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))
        
class CostModelCollisionCircle():
    '''
    The collision cost model between the end-effector and an ellipsoid obstacle
    '''
    def __init__(self, sys, activation, p_obs, Sigma_obs, w_obs = 1., d_thres = 1.):
        self.sys = sys
        self.Dx, self.Du = sys.Dx, sys.Du
        self.p_obs = p_obs #obstacle position
        self.Sigma_obs = Sigma_obs #obstacle ellipse covariance matrix
        self.Sigma_obs_inv = np.linalg.inv(Sigma_obs)
        self.w_obs = w_obs
        self.d_thres = d_thres
        self.obs_status = False
        self.activation = activation
        
    def calc(self, x, u):
        p,_ = self.sys.compute_ee(x)
        self.r = p - self.p_obs
        self.activation.calc(self.r)
        self.L = self.activation.a*self.w_obs
        return self.L
    
    def calcDiff(self, x, u, recalc = True):
        if recalc:
            self.calc(x, u)
        self.J   = self.sys.compute_Jacobian(x)
        p,_      = self.sys.compute_ee(x)
        
        ###Compute the cost derivatives###
        self.activation.calcDiff(self.r)
        self.Lx = np.vstack([self.J.T.dot(self.activation.Ar), np.zeros((2, 1))])
        self.Lxx = np.vstack([
              np.hstack([self.J.T.dot(self.activation.Arr).dot(self.J),
                      np.zeros((2, 2))]),
           np.zeros((2, 4))
        ])*self.w_obs
        self.Lx = self.Lx[:,0]*self.w_obs
        
        self.Lu  = np.zeros(self.Du)
        self.Lxu = np.zeros((self.Dx, self.Du))
        self.Luu  = np.zeros((self.Du, self.Du))
        
class ActivationCollision():
    def __init__(self, nr, threshold=0.3):
        self.threshold = threshold
        self.nr = nr
    def calc(self,  r):
        self.d = np.linalg.norm(r)

        if self.d < self.threshold:
            self.a = 0.5*(self.d-self.threshold)**2
        else:
            self.a = 0
        return self.a

    def calcDiff(self,  r, recalc=True):
        if recalc:
            self.calc(r)
        
        if self.d < self.threshold:
            self.Ar = (self.d-self.threshold)*r[:,None]/self.d
            self.Arr = np.eye(self.nr)*(self.d-self.threshold)/self.d + self.threshold*np.outer(r,r.T)/(self.d**3)
        else:
            self.Ar = np.zeros((self.nr,1))
            self.Arr = np.zeros((self.nr,self.nr))
        return self.Ar, self.Arr
    
class SphereSphereCollisionCost():
    def __init__(self, activation=None, nu=None, r_body = 0., r_obs = 0., pos_obs = np.array([0,0,0]), w= 1.):
        self.activation = activation 
        self.r_body = r_body
        self.r_obs = r_obs
        self.pos_obs = pos_obs
        self.w = w
        
    def calc(self, x, u):              
        #calculate residual
        pos_body = x[:3]
        self.r = pos_body-self.pos_obs
        self.activation.calc(self.r)
        self.L = self.activation.a*self.w
        return self.L

    def calcDiff(self,  x, u, recalc=True):
        if recalc:
            self.calc( x, u)

        #Calculate the Jacobian at p1
        self.J = np.hstack([np.eye(3), np.zeros((3,3))])
        ###Compute the cost derivatives###
        self.activation.calcDiff(self.r, recalc)
        self.Rx = np.hstack([self.J, np.zeros((3, 6))])
        self.Lx = np.vstack([self.J.T.dot(self.activation.Ar), np.zeros((6, 1))])
        self.Lxx = np.vstack([
              np.hstack([self.J.T.dot(self.activation.Arr).dot(self.J),
                      np.zeros((6, 6))]),
           np.zeros((6, 12))
        ])*self.w
        self.Lx = self.Lx[:,0]*self.w
        self.Lu = np.zeros(4)
        self.Luu = np.zeros((4,4))
        self.Lxu = np.zeros((12, 4))
        return self.Lx, self.Lxx