import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

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
        return self.A,self.B
    
    def compute_ee(self,x):
        #The end-effector for a point mass system is simply its position
        return x[:self.Dx/2], None 
    
    def compute_Jacobian(self,x):
        #The end-effector Jacobian for a point mass system is simply an identity matrix
        return np.eye(self.Dx/2) 
    
    
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

class Unicycle():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 3
        self.Du = 2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(3)
        A[0,2] = -u[0]*np.sin(x[2])*self.dt
        A[1,2] = u[0]*np.cos(x[2])*self.dt
        
        B = np.zeros((3,2))
        B[0,0] = np.cos(x[2])*self.dt
        B[1,0] = np.sin(x[2])*self.dt
        B[2,1] = 1*self.dt
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        #A,B = self.compute_matrices(x,u)
        x_next = np.zeros(3)
        
        x_next[0] = x[0] + u[0]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + u[0]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + u[1]*self.dt
        #pdb.set_trace()
        return x_next
        #return A.dot(x) + B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
class SecondUnicycle():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 5
        self.Du = 2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        A[0,2] = -x[3]*np.sin(x[2])*self.dt
        A[1,2] = x[3]*np.cos(x[2])*self.dt
        
        A[0,3] = np.cos(x[2])*self.dt
        A[1,3] = np.sin(x[2])*self.dt
        A[2,4] = 1*self.dt
        
        B = np.zeros((self.Dx,self.Du))
        B[3,0] = self.dt
        B[4,1] = self.dt
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        
        x_next[0] = x[0] + x[3]*np.cos(x[2])*self.dt
        x_next[1] = x[1] + x[3]*np.sin(x[2])*self.dt
        x_next[2] = x[2] + x[4]*self.dt
        x_next[3] = x[3] + u[0]*self.dt
        x_next[4] = x[4] + u[1]*self.dt
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
class Pendulum():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 2
        self.Du = 1
        self.b = 1
        self.m = 1
        self.l = 1
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,1] = self.dt
        A[1,0] = 0.5*9.8*self.dt*np.cos(x[0])/self.l
        A[1,1] = 1 - self.dt*self.b/(self.m*self.l**2)
        
        B[1,0] = self.dt/(self.m*self.l**2)
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        x_next[0] = x[0] + x[1]*self.dt
        x_next[1] = (1-self.dt*self.b/(self.m*self.l**2))*x[1] + 0.5*9.8*self.dt*np.sin(x[0])/self.l + self.dt*u/(self.m*self.l**2) 
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
        px = np.array([0, -self.l*np.sin(x[0])])
        py = np.array([0, self.l*np.cos(x[0])])
        line = plt.plot(px, py, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-2*self.l, 2*self.l]
        plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line

    def plot_traj(self, xs, dt = 0.1, filename = None):
        for i,x in enumerate(xs):
            clear_output(wait=True)
            self.plot(x)
            if filename is not None:
                plt.savefig('temp/fig'+str(i)+'.png')
            plt.show()
            time.sleep(dt)
    
    
class Bicopter():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 6
        self.Du = 2
        
        self.m = 2.5
        self.l = 1
        self.I = 1.2
        
    def set_init_state(self,x0):
        self.x0 = x0

    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        A[:3, 3:] = np.eye(3)*self.dt
        A[3,2] = -self.dt*(u[0]+u[1])*np.cos(x[2])/self.m
        A[4,2] = -self.dt*(u[0]+u[1])*np.sin(x[2])/self.m
        
        B = np.zeros((self.Dx,self.Du))
        B[3,0] = -self.dt*np.sin(x[2])/self.m
        B[3,1] = B[3,0]
        
        B[4,0] = self.dt*np.cos(x[2])/self.m
        B[4,1] = B[4,0]
        
        B[5,0] = self.dt*self.l*0.5/self.I
        B[5,1] = -B[5,0]
        
        self.A, self.B = A,B
        return A,B
        
    def step(self, x, u):
        x_next = np.zeros(self.Dx)
        
        x_next[0] = x[0] + x[3]*self.dt
        x_next[1] = x[1] + x[4]*self.dt
        x_next[2] = x[2] + x[5]*self.dt
        
        x_next[3] = x[3] - (u[0]+u[1])*np.sin(x[2])*self.dt/self.m
        x_next[4] = x[4] + (u[0]+u[1])*np.cos(x[2])*self.dt/self.m - 9.8*self.dt
        x_next[5] = x[5] + (u[0]-u[1])*self.dt*self.l*0.5/self.I
        
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
        
    def plot(self, x, color = 'k'):
        pxs = np.array([x[0] + 0.5*self.l*np.cos(x[2]), x[0] - 0.5*self.l*np.cos(x[2])])
        pys = np.array([x[1] + 0.5*self.l*np.sin(x[2]), x[1] - 0.5*self.l*np.sin(x[2])])
        line = plt.plot(pxs, pys, marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        return line

    def vis_traj(self, xs, dt = 0.1, axes_lim = [-5,5,-5,5]):
        T = len(xs)
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.axes().set_aspect('equal')
            plt.axis(axes_lim)
            plt.show()
            time.sleep(dt)
            
        
class TwoLinkRobot():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 4
        self.Du = 2
        self.l1 = 1.5
        self.l2 = 1
        self.p_ref = np.zeros(2)
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def set_pref(self, p_ref):
        self.p_ref = p_ref
    
    def compute_matrices(self,x,u):
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,2] = self.dt
        A[1,3] = self.dt
        
        B[2:,:] = np.eye(self.Du)
        
        self.A, self.B = A,B
        return A,B
    
    def compute_Jacobian(self, x):
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
    
    def compute_ee(self,x):
        self.p1 = np.array([self.l1*np.cos(x[0]), self.l1*np.sin(x[0])])
        self.p2 = np.array([self.p1[0] + self.l2*np.cos(x[0] + x[1]), self.p1[1] + self.l2*np.sin(x[0] + x[1])])
        return self.p2, self.p1
    
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
            
