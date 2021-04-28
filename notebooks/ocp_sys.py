import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pybullet as p
import time
import crocoddyl

class LinearSystem():
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.Dx = A.shape[0]
        self.Du = B.shape[1]
        
    def reset_AB(self, A,B):
        self.A = A
        self.B = B
        
    def set_init_state(self,x0):
        self.x0 = x0
    
    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        return self.A,self.B
    
    def compute_ee(self,x, ee_id=1):
        #The end-effector for a point mass system is simply its position
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return x[:int(self.Dx/2)], None 
    
    def compute_Jacobian(self,x, ee_id=1):
        #The end-effector Jacobian for a point mass system is simply an identity matrix
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return np.eye(int(self.Dx/2)) 
    
    
    def step(self, x, u):
        return self.A.dot(x) + self.B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)

        
class TwoLinkRobot():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 4
        self.Du = 2
        self.dof = 2
        self.l1 = 1.5
        self.l2 = 1
        self.p_ref = np.zeros(2)
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def set_pref(self, p_ref):
        self.p_ref = p_ref
    
    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,2] = self.dt
        A[1,3] = self.dt
        
        B[2:,:] = np.eye(self.Du)
        
        self.A, self.B = A,B
        return A,B
    
    def compute_ee(self,x, ee_id=0):
        self.p1 = np.array([self.l1*np.cos(x[0]), self.l1*np.sin(x[0])])
        self.p2 = np.array([self.p1[0] + self.l2*np.cos(x[0] + x[1]), self.p1[1] + self.l2*np.sin(x[0] + x[1])])
        return self.p2, self.p1

    
    def compute_Jacobian(self, x, ee_id=0):
        J = np.zeros((2, 2))
        s1 = np.sin(x[0])
        c1 = np.cos(x[0])
        s12 = np.sin(x[0] + x[1])
        c12 = np.cos(x[0] + x[1])
        
        J[0,0] = -self.l1*s1 - self.l2*s12
        J[0,1] = - self.l2*s12
        J[1,0] =  self.l1*c1 + self.l2*c12
        J[1,1] =  self.l2*c12
        
        self.J = J
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    
    def plot(self, x, color='k'):
        self.compute_ee(x)
        
        line1 = plt.plot(np.array([0, self.p1[0]]),np.array([0, self.p1[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        line2 = plt.plot(np.array([self.p1[0], self.p2[0]]),np.array([self.p1[1], self.p2[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-1.5*(self.l1+self.l2), 1.5*(self.l1+self.l2)]
        #plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line1,line2

    def plot_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.plot(self.p_ref[0], self.p_ref[1], '*')
            plt.show()
            time.sleep(self.dt)
            
class URDFRobot():
    def __init__(self, dof, robot_id, joint_indices = None, dt = 0.01):
        self.dt = dt
        self.Dx = dof*2
        self.Du = dof
        self.dof = dof
        self.robot_id = robot_id
        if joint_indices is None:
            self.joint_indices = np.arange(dof)
        else:
            self.joint_indices = joint_indices
        
    def set_init_state(self,x0):
        self.x0 = x0
        self.set_q(x0)

    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[:self.dof, self.dof:] = np.eye(self.dof)*self.dt
        
        #B[self.dof:,:] = np.eye(self.Du)
        B[:self.dof,:] = np.eye(self.Du) * self.dt * self.dt /2
        B[-self.dof:,:] = np.eye(self.Du) * self.dt    

        self.A, self.B = A,B
        return A,B
    
    def compute_ee(self,x, ee_id):
        self.set_q(x)
        ee_data = p.getLinkState(self.robot_id, ee_id)
        pos = np.array(ee_data[0])
        quat = np.array(ee_data[1])
        return pos, quat
    
    def compute_Jacobian(self, x, ee_id):
        zeros = [0.]*self.dof
        Jl, Ja = p.calculateJacobian(self.robot_id, ee_id, [0.,0.,0.], x[:self.dof].tolist(),zeros,zeros)
        Jl, Ja = np.array(Jl), np.array(Ja)
        self.J = np.concatenate([Jl, Ja], axis=0)
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def set_q(self, x):
        q = x[:self.dof]
        for i in range(self.dof):
            p.resetJointState(self.robot_id, self.joint_indices[i], q[i])
        return 

    def vis_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.set_q(x)
            time.sleep(self.dt)
            
class ActionModelRobot(crocoddyl.ActionModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ActionModelAbstract.__init__(self, state, nu)
        
    def init_robot_sys(self,robot_sys, nr = 1):
        self.robot_sys = robot_sys
        self.Du = robot_sys.Du
        self.Dx = robot_sys.Dx
        self.Dr = nr
        
    def set_cost(self, cost_model):
        self.cost_model = cost_model
        
    def calc(self, data, x, u):
        #calculate the cost
        data.cost = self.cost_model.calc(x,u)
        
        #calculate the next state
        data.xnext = self.robot_sys.step(x,u)
        
    def calcDiff(self, data, x, u, recalc = False):
        if recalc:
            self.calc(data, x, u)

        #compute cost derivatives
        self.cost_model.calcDiff(x, u)
        data.Lx = self.cost_model.Lx.copy()
        data.Lxx = self.cost_model.Lxx.copy()
        data.Lu = self.cost_model.Lu.copy()
        data.Luu = self.cost_model.Luu.copy()
        
        #compute dynamic derivatives 
        A, B = self.robot_sys.compute_matrices(x,u)
        data.Fx = A.copy()
        data.Fu = B.copy()
        
    def createData(self):
        data = ActionDataRobot(self)
        return data

class ActionDataRobot(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self,model)