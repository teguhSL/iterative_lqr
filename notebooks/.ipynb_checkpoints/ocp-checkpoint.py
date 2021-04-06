import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv
from ocp_sys import *
   
class ILQR_Standard():
    '''
    ILQR Standard: uses the standard quadratic cost function Q, R, and Qf
    This class is kept only for educational purpose, as it is simpler than the one 
    using cost model
    '''
    def __init__(self, sys, mu = 1e-6):
        self.sys, self.Dx, self.Du = sys, sys.Dx, sys.Du
        self.mu = mu
        
    def set_timestep(self,T):
        self.T = T
        self.allocate_data()
        
    def set_reg(self,mu):
        self.mu = mu
        
    def set_ref(self, x_refs):
        self.x_refs = x_refs.copy()
        
    def allocate_data(self):
        self.Lx  = np.zeros((self.T+1, self.Dx)) 
        self.Lu  = np.zeros((self.T+1,   self.Du))
        self.Lxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Luu = np.zeros((self.T+1,   self.Du, self.Du))
        self.Fx  = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Fu  = np.zeros((self.T+1, self.Dx, self.Du))
        self.Vx  = np.zeros((self.T+1, self.Dx))
        self.Vxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Qx  = np.zeros((self.T,   self.Dx))
        self.Qu  = np.zeros((self.T,   self.Du))
        self.Qux = np.zeros((self.T,   self.Du, self.Dx))
        self.Qxx = np.zeros((self.T,   self.Dx, self.Dx))
        self.Quu = np.zeros((self.T,   self.Du, self.Du))
        self.k = np.zeros((self.T, self.Du))
        self.K = np.zeros((self.T, self.Du, self.Dx))
        
        self.xs = np.zeros((self.T+1, self.Dx))
        self.us = np.zeros((self.T+1, self.Du))
        self.x_refs = np.zeros((self.T+1, self.Dx))

    def set_cost(self, Q, R, Qf = None):
        if Q.ndim == 2:
            self.Q = np.array([Q]*(self.T+1))
            self.R = np.array([R]*(self.T+1)) #note: the last R is only created for convenience, u_T does not affect anything and will be zero
            if Qf is not None:
                self.Q[-1] = Qf
        elif Q.ndim == 3:
            self.Q = Q
            self.R = R
        else:
            print('Number of dimensions must be either 2 or 3')
            #raise()    
                
    def set_init_state(self,x0):
        self.x0 = x0.copy()
        
    def set_state(self, xs, us):
        self.xs = xs.copy()
        self.us = us.copy()
        
    def calc_diff(self):
        for i in range(self.T+1):
            self.Lx[i] = self.Q[i].dot(self.xs[i]- self.x_refs[i])
            self.Lxx[i] = self.Q[i]
            self.Luu[i] = self.R[i]
            self.Lu[i] = self.R[i].dot(self.us[i])
            self.Fx[i], self.Fu[i] = self.sys.compute_matrices(self.xs[i], self.us[i])
            
    def calc_cost(self, xs, us):
        running_cost_state = 0
        running_cost_control = 0
        cost = 0
        #for i in range(self.T):
        #    cost += (xs[i]- self.x_refs[i]).T.dot(self.Q[i]).dot(xs[i]- self.x_refs[i]) + us[i].T.dot(self.R[i]).dot(us[i])
        #cost += (xs[self.T]- self.x_refs[i]).T.dot(self.Q[self.T]).dot(xs[self.T]- self.x_refs[i])
        for i in range(self.T):
            running_cost_state += (xs[i]- self.x_refs[i]).T.dot(self.Q[i]).dot(xs[i]- self.x_refs[i])
            running_cost_control += us[i].T.dot(self.R[i]).dot(us[i])
        terminal_cost_state = (xs[self.T]- self.x_refs[i]).T.dot(self.Q[self.T]).dot(xs[self.T]- self.x_refs[i])
        self.cost = running_cost_state + running_cost_control + terminal_cost_state
        self.running_cost_state = running_cost_state
        self.running_cost_control = running_cost_control
        self.terminal_cost_state = terminal_cost_state
        return self.cost
        
    def forward_pass(self, max_iter = 10, method = 'batch'):
        cost0 = self.calc_cost(self.xs, self.us)
        print(cost0)
        alpha = 1.
        fac = 0.8
        cost = 5*cost0
        
        n_iter = 0
        while cost > cost0 and n_iter < max_iter  :
            if method == 'recursive':
                xs_new = []
                us_new = []
                x = self.x0.copy()
                xs_new += [x]
                for i in range(self.T):
                    u = self.us[i] + alpha*self.k[i] + self.K[i].dot(x-self.xs[i])
                    x = self.sys.step(x,u)
                    xs_new += [x]
                    us_new += [u]

                us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience
                
            elif method == 'batch':
                #use the actual dynamic for rollout
                xs_new = []
                us_new = []
                dus = self.del_us_ls.reshape(-1, self.Du)
                x = self.x0.copy()
                xs_new += [x]
                for i in range(self.T):
                    u = self.us[i] + alpha*dus[i]
                    x = self.sys.step(x,u)
                    xs_new += [x]
                    us_new += [u]
                us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience

                #use the linearized dynamic for rollout
#                 dxs  = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(alpha*self.del_us_ls)
#                 dus = alpha*self.del_us_ls.reshape(-1, self.Du)
#                 dxs = dxs.reshape(-1, self.Dx)
#                 xs_new = self.xs + dxs
#                 us_new = self.us + dus
                
            cost = self.calc_cost(xs_new,us_new)
            print(alpha,cost)
            alpha *= fac
            n_iter += 1
        self.xs, self.us = np.array(xs_new), np.array(us_new)
        
        
               
    def backward_pass(self, method = 'batch'):
        if method == 'recursive':
            self.Vx[self.T] = self.Lx[self.T]
            self.Vxx[self.T] = self.Lxx[self.T]
            for i in np.arange(self.T-1, -1,-1):
                self.Qx[i] = self.Lx[i]   + self.Fx[i].T.dot(self.Vx[i+1])
                self.Qu[i] = self.Lu[i]   + self.Fu[i].T.dot(self.Vx[i+1])
                self.Qxx[i] = self.Lxx[i] + self.Fx[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
                self.Quu[i] = self.Luu[i] + self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fu[i]) + self.mu*np.eye(self.Du)
                self.Qux[i] = self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
                Quuinv = inv(self.Quu[i])
                self.k[i] = -Quuinv.dot(self.Qu[i])
                self.K[i] = -Quuinv.dot(self.Qux[i])

                self.Vx[i] = self.Qx[i] - self.Qu[i].dot(Quuinv).dot(self.Qux[i])
                self.Vxx[i] = self.Qxx[i] - self.Qux[i].T.dot(Quuinv).dot(self.Qux[i])
                #ensure symmetrical Vxx
                self.Vxx[i] = 0.5*(self.Vxx[i] + self.Vxx[i].T)      
        
        elif method == 'batch':
            self.Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
            self.Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))

            for i in range(self.T+1):
                self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Lxx[i]
                self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.Luu[i]

            self.Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
            self.Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

            #### Calculate Sx and Su 
            i = 0
            self.Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
            for i in range(1, self.T+1):
                self.Sx[self.Dx*i:self.Dx*(i+1), :] =  self.Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.Fx[i-1])

            for i in range(1,self.T+1):
                self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.Fu[i-1]
                self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = self.Fx[i-1].dot(self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

            self.Lxs = self.Lx.flatten()
            self.Lus = self.Lu.flatten()

            #### Calculate X and U 
            self.Sigma_u_inv = (self.Su.T.dot(self.Qs.dot(self.Su)) + self.Rs)
            self.del_us_ls = -np.linalg.solve(self.Sigma_u_inv, self.Su.T.dot(self.Qs.dot(self.Sx.dot(-np.zeros(self.Dx)))) + self.Lxs.dot(self.Su) + self.Lus )
            self.del_xs_ls = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(self.del_us_ls)
    
    def solve(self, n_iter = 3, method = 'batch'):
        for i in range(n_iter):
            self.calc_diff()
            if method == 'recursive':
                self.backward_pass(method='recursive')
                self.forward_pass(method='recursive')
            elif method == 'batch':
                self.backward_pass(method='batch')
                self.forward_pass(method='batch')



class ILQR():
    def __init__(self, sys, mu = 1e-6):
        self.sys, self.Dx, self.Du = sys, sys.Dx, sys.Du
        self.mu = mu
        
    def set_timestep(self,T):
        self.T = T
        self.allocate_data()
        
    def set_reg(self,mu):
        self.mu = mu
        
    def allocate_data(self):
        self.Lx  = np.zeros((self.T+1, self.Dx)) 
        self.Lu  = np.zeros((self.T+1,   self.Du))
        self.Lxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Luu = np.zeros((self.T+1,   self.Du, self.Du))
        self.Fx  = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Fu  = np.zeros((self.T+1, self.Dx, self.Du))
        self.Vx  = np.zeros((self.T+1, self.Dx))
        self.Vxx = np.zeros((self.T+1, self.Dx, self.Dx))
        self.Qx  = np.zeros((self.T,   self.Dx))
        self.Qu  = np.zeros((self.T,   self.Du))
        self.Qux = np.zeros((self.T,   self.Du, self.Dx))
        self.Qxx = np.zeros((self.T,   self.Dx, self.Dx))
        self.Quu = np.zeros((self.T,   self.Du, self.Du))
        self.k = np.zeros((self.T, self.Du))
        self.K = np.zeros((self.T, self.Du, self.Dx))
        
        self.xs = np.zeros((self.T+1, self.Dx))
        self.us = np.zeros((self.T+1, self.Du))

    def set_cost(self, costs):
        self.costs = costs
                
    def set_init_state(self,x0):
        self.x0 = x0.copy()
        
    def set_state(self, xs, us):
        self.xs = xs.copy()
        self.us = us.copy()
        
    def calc_diff(self):
        for i in range(self.T+1):
            self.costs[i].calcDiff(self.xs[i], self.us[i])
            self.Lx[i]  = self.costs[i].Lx
            self.Lxx[i] = self.costs[i].Lxx
            self.Lu[i]  = self.costs[i].Lu
            self.Luu[i] = self.costs[i].Luu
            self.Fx[i], self.Fu[i] = self.sys.compute_matrices(self.xs[i], self.us[i])
            
    def calc_cost(self, xs, us):
        self.cost = np.sum([self.costs[i].calc(xs[i], us[i]) for i in range(self.T+1)])
        return self.cost
    
    def forward_pass(self, max_iter = 100, method = 'batch'):
        cost0 = self.calc_cost(self.xs, self.us)
        print(cost0)
        alpha = 1.
        fac = 0.8
        cost = 5*np.abs(cost0)
        
        n_iter = 0
        while cost > cost0 and n_iter < max_iter  :
            if method == 'recursive':
                xs_new = []
                us_new = []
                x = self.x0.copy()
                xs_new += [x]
                for i in range(self.T):
                    u = self.us[i] + alpha*self.k[i] + self.K[i].dot(x-self.xs[i])
                    x = self.sys.step(x,u)
                    xs_new += [x]
                    us_new += [u]

                us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience
                
            elif method == 'batch':
                #use the actual dynamic for rollout
                xs_new = []
                us_new = []
                dus = self.del_us_ls.reshape(-1, self.Du)
                x = self.x0.copy()
                xs_new += [x]
                for i in range(self.T):
                    u = self.us[i] + alpha*dus[i]
                    x = self.sys.step(x,u)
                    xs_new += [x]
                    us_new += [u]
                us_new += [np.zeros(self.Du)]  #add the last control as 0, for convenience

                #use the linearized dynamic for rollout
#                 dxs  = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(alpha*self.del_us_ls)
#                 dus = alpha*self.del_us_ls.reshape(-1, self.Du)
#                 dxs = dxs.reshape(-1, self.Dx)
#                 xs_new = self.xs + dxs
#                 us_new = self.us + dus
                
            cost = self.calc_cost(xs_new,us_new)
            print(alpha,cost)
            alpha *= fac
            n_iter += 1
        if n_iter == max_iter :
            print('Cannot find a good direction')
            raise Exception
        self.xs, self.us = np.array(xs_new), np.array(us_new)
    
    
    def backward_pass(self, method = 'batch'):
        if method == 'recursive':
            self.Vx[self.T] = self.Lx[self.T]
            self.Vxx[self.T] = self.Lxx[self.T]
            for i in np.arange(self.T-1, -1,-1):
                self.Qx[i] = self.Lx[i]   + self.Fx[i].T.dot(self.Vx[i+1])
                self.Qu[i] = self.Lu[i]   + self.Fu[i].T.dot(self.Vx[i+1])
                self.Qxx[i] = self.Lxx[i] + self.Fx[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
                self.Quu[i] = self.Luu[i] + self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fu[i]) + self.mu*np.eye(self.Du)
                self.Qux[i] = self.Fu[i].T.dot(self.Vxx[i+1]).dot(self.Fx[i])
                Quuinv = inv(self.Quu[i])
                self.k[i] = -Quuinv.dot(self.Qu[i])
                self.K[i] = -Quuinv.dot(self.Qux[i])

                self.Vx[i] = self.Qx[i] - self.Qu[i].dot(Quuinv).dot(self.Qux[i])
                self.Vxx[i] = self.Qxx[i] - self.Qux[i].T.dot(Quuinv).dot(self.Qux[i])
                #ensure symmetrical Vxx
                self.Vxx[i] = 0.5*(self.Vxx[i] + self.Vxx[i].T)      
        
        elif method == 'batch':
            self.Qs = np.zeros(((self.T+1)*self.Dx,(self.T+1)*self.Dx))
            self.Rs = np.zeros(((self.T+1)*self.Du,(self.T+1)*self.Du))

            for i in range(self.T+1):
                self.Qs[self.Dx*i:self.Dx*(i+1),self.Dx*i:self.Dx*(i+1)] = self.Lxx[i]
                self.Rs[self.Du*i:self.Du*(i+1),self.Du*i:self.Du*(i+1)] = self.Luu[i]

            self.Sx = np.zeros((self.Dx*(self.T+1),self.Dx))
            self.Su = np.zeros((self.Dx*(self.T+1),self.Du*(self.T+1)))

            #### Calculate Sx and Su 
            i = 0
            self.Sx[self.Dx*i:self.Dx*(i+1), :] = np.eye(self.Dx)
            for i in range(1, self.T+1):
                self.Sx[self.Dx*i:self.Dx*(i+1), :] =  self.Sx[self.Dx*(i-1):self.Dx*(i), :].dot(self.Fx[i-1])

            for i in range(1,self.T+1):
                self.Su[self.Dx*i:self.Dx*(i+1), self.Du*(i-1): self.Du*(i)] = self.Fu[i-1]
                self.Su[self.Dx*i:self.Dx*(i+1), :self.Du*(i-1)] = self.Fx[i-1].dot(self.Su[self.Dx*(i-1):self.Dx*(i), :self.Du*(i-1)])

            self.Lxs = self.Lx.flatten()
            self.Lus = self.Lu.flatten()

            #### Calculate X and U 
            self.Sigma_u_inv = (self.Su.T.dot(self.Qs.dot(self.Su)) + self.Rs)
            self.del_us_ls = -np.linalg.solve(self.Sigma_u_inv, self.Su.T.dot(self.Qs.dot(self.Sx.dot(-np.zeros(self.Dx)))) + self.Lxs.dot(self.Su) + self.Lus )
            self.del_xs_ls = self.Sx.dot(np.zeros(self.Dx)) + self.Su.dot(self.del_us_ls)

    def solve(self, n_iter = 3, method = 'batch'):
        for i in range(n_iter):
            self.calc_diff()
            if method == 'recursive':
                self.backward_pass(method='recursive')
                self.forward_pass(method='recursive')
            elif method == 'batch':
                self.backward_pass(method='batch')
                self.forward_pass(method='batch')