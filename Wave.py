#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:49:55 2019

@author: anass
"""
from dolfin import *
from mshr import *

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import factorized
from scipy.sparse import csr_matrix, csc_matrix, issparse
from scipy import integrate

from assimulo.solvers.sundials import CVode
from assimulo.solvers import RungeKutta34 
from assimulo.problem import Explicit_Problem 
from assimulo.problem import Implicit_Problem 
from assimulo.solvers import IDA,Radau5DAE              


import sys
import time


class Wave_2D:
    # Constructor of the domain
    def __init__(self):
        """
        Constructor for the Wave_2D class.
        """
        # Information related to the problem definition
        self.set_domain                = 0
        self.set_boundary_cntrl_space  = 0
        self.set_boundary_cntrl_time   = 0
        self.set_physical_parameters   = 0
        self.set_impedance             = 0
        self.set_initial_close_final_time    = 0
        self.problem_definition        = 0
        self.discontinous_boundary_values = 0
        self.discontinuous_impedance   = 0
        
        # Information related to the space-time discretization
        self.generate_mesh                  = 0
        self.set_FE_approximation_spaces    = 0
        self.apply_PFEM_formulation         = 0
        self.project_initial_data_FE_spaces = 0
        self.set_time_setting               = 0
        
        # Information related to the numerical approximation schemes
        self.dense  = 0
        self.sparse = 0
        
        # Information related to the post-processing
        self.interactive = False
        self.docker      = False
        self.notebook    = False

    ## Method
    #  @param self The object pointer.
    def my_mult(self, A, B):
        """
        Matrix multiplication.

        Multiplies A and B together via the "dot" method.

        Parameters
        ----------
        A : array_like
            first matrix in the product A*B being calculated
        B : array_like
            second matrix in the product A*B being calculated

        Returns
        -------
        array_like
            product of the inputs A and B
        """
        
        if issparse(B) and not issparse(A):
            # dense.dot(sparse) is not available in scipy.
            return B.T.dot(A.T).T
        else:
            return A.dot(B)

    ## Method.
    #  @param self The object pointer.
    def Check_Problem_Definition(self):
        """
        Check if the problem definition has been performed correctly
        """
        
        assert self.set_domain == 1, \
            'The domain has not been defined properly'
        
        assert self.set_boundary_cntrl_space == 1, \
            'Boundary control (space part) has not been set properly'
 
        assert self.set_boundary_cntrl_time == 1, \
            'Boundary control (time part) has not been set properly'
       
        assert self.set_physical_parameters == 1, \
            'The physical parameters have not been set properly'
            
        assert self.set_impedance == 1, \
            'The impedance has not been set properly' 
    
        assert self.set_initial_close_final_time == 1, \
            'The initial, close and final times have not been set' 
            
        self.problem_definition = 1
        
        return self.problem_definition

    ## Method.
    #  @param self The object pointer.
    def Set_rectangular_domain(self, x0, xL, y0, yL):
        """
        Set the dimensions of the rectangular domain
        """
        self.x0 = x0
        self.xL = xL
        self.y0 = y0
        self.yL = yL
        
        self.set_domain = 1
         
        return self.set_domain
    
    ## Method.
    #  @param self The object pointer.
    def Set_physical_parameters(self, Rho, T11, T12, T22, **kwargs):
        """
        Set the physical parameters as a FeniCS expression related to the PDE
        """
        self.rho = Expression(Rho, degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        self.T   = Expression( ( (T11, T12), (T12, T22) ), degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        
        self.set_physical_parameters = 1
        
        return self.set_physical_parameters
 
    ## Method.
    #  @param self The object pointer.
    def Set_impedance(self, Z, discontinuous=False, **kwargs):
        """
        Set the physical impedance as a regular FeniCS expression
        """
        self.z = Expression(Z, degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        
        if discontinuous : 
            self.discontinuous_impedance = 1 
        
        self.set_impedance = 1
        
        return self.set_impedance
    
    ## Method.
    #  @param self The object pointer.
    def Set_boundary_cntrl_space(self, boundary_cntrl_space, discontinuous=False, **kwargs):
        """
        Set the geometrical part of the boundary control as a regular FeniCS expression
        """
        self.boundary_cntrl_space = Expression(boundary_cntrl_space, degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        
        if discontinuous : 
            self.discontinous_boundary_values = 1
        
        self.set_boundary_cntrl_space = 1
        
        return self.set_boundary_cntrl_space

    ## Method.
    #  @param self The object pointer.
    def Set_boundary_cntrl_time(self, time_function=None):
        """
        Set the time part of the boundary control as a regular FeniCS expression
        """
        def unit_function(t,tclose) :
            return 1.
        
        if time_function == None:
            self.boundary_cntrl_time = unit_function
        else :
            self.boundary_cntrl_time = time_function
        
        self.set_boundary_cntrl_time = 1
        
        return self.set_boundary_cntrl_time
    
    ## Method.
    #  @param self The object pointer.
    def Set_initial_close_final_time(self, initial_time, close_time, final_time):
        """
        Set the initial, close and final times for defining the time domain
        """
        self.tinit  = initial_time 
        self.tclose = close_time
        self.tfinal = final_time
        
        self.set_initial_close_final_time = 1
        
        return self.set_initial_close_final_time   
    
    ## Method.
    #  @param self The object pointer. 
    def Evaluate_cntrl_function(self,t):
        """
        Evaluate the control function to be used at time t
        """
        assert self.set_time_setting == 1, "Set_time_setting must be called before."
        
        self.evaluate_cntrl_function = 1
        
        if t <= self.tclose:
            return np.sin(2*np.pi*t)
        else:
            return 0.
  
    #
    # Space and time discretization 
    #

    ## Method.
    #  @param self The object pointer.
    def Check_Space_Time_Discretization(self):
        """
        Check if the space and time discretizations have been performed correctly
        """
        
        assert self.generate_mesh == 1, \
            'The finite element mesh must be generated first'
        
        assert self.set_FE_approximation_spaces == 1, \
            'The FE approximation spaces must be selected first'
            
        assert self.apply_PFEM_formulation == 1, \
            'The PFEM formulation has to be applied' 
    
        assert self.project_initial_data_FE_spaces == 1, \
            'The initial data must be interpolated on the FE spaces' 

        assert self.set_time_setting == 1,\
            'The parameters for the time discretization must be set'
    
        self.space_time_discretization = 1
        
        return self.space_time_discretization
    
    
    ## Method.
    #  @param self The object pointer.   
    def Generate_mesh(self, rfn, structured_mesh=False):
        """
        Perform the mesh generation through the Fenics meshing functionalities
        """
        self.rfn = rfn  
        
        if structured_mesh: 
            self.Mesh = RectangleMesh(Point(self.x0,self.y0), Point(self.xL,self.yL), self.rfn, self.rfn, 'crossed')
        else:
            self.Mesh = generate_mesh(Rectangle(Point(self.x0,self.y0), Point(self.xL,self.yL)), self.rfn)
        
        self.norext   = FacetNormal(self.Mesh)
        
        self.generate_mesh = 1
        
        return self.generate_mesh

    ## Method.
    #  @param self The object pointer.   
    def Set_FE_Approximation_Spaces(self, family_q, family_p, family_b, rt_order, p_order, b_order):
        """
        Set the finite element approximation spaces related to the space discretization of the PDE
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        # Orders
        self.rt_order = rt_order
        self.p_order  = p_order
        self.b_order  = b_order
        
        # Spaces
        self.Vq = FunctionSpace(self.Mesh, family_q, self.rt_order+1)
        
        if self.p_order == 0 and family_p == 'P':
            self.Vp = FunctionSpace(self.Mesh, 'DG', 0)
        else :
            self.Vp = FunctionSpace(self.Mesh, family_p, self.p_order)

        if self.b_order == 0  and family_b == 'P':
            self.Vb = FunctionSpace(self.Mesh, 'CR', 1)
        else :
            self.Vb = FunctionSpace(self.Mesh, family_b, self.b_order)
            
        #self.Vp = FunctionSpace(self.Mesh, family_p, self.p_order)
        #self.Vb = FunctionSpace(self.Mesh, family_b, self.b_order)
        
        # DOFs
        self.Nq = self.Vq.dim()
        self.Np = self.Vp.dim()
        self.coord_q = self.Vq.tabulate_dof_coordinates()
        self.coord_p = self.Vp.tabulate_dof_coordinates()
        
        # p explicit coordinates (Lagrange in general)
        self.xp = self.coord_p[:,0]
        self.yp = self.coord_p[:,1]
        
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b  = self.Vb.tabulate_dof_coordinates()
        xb       = coord_b[:,0]
        yb       = coord_b[:,1]
        bndr_i_b = []
        for i in range(self.Vb.dim()) :
            if np.abs(xb[i] - self.x0) <= 1e-16 or np.abs(xb[i] - self.xL) <= 1e-16 or np.abs(yb[i] - self.y0) <= 1e-16 or np.abs(yb[i] - self.yL) <= 1e-16 : 
                 bndr_i_b.append(i)
        self.bndr_i_b = bndr_i_b
 
        # Exlpicit information about boundary DOFs 
        self.coord_b = coord_b[bndr_i_b,:]
        self.xb      = xb[bndr_i_b]
        self.yb      = yb[bndr_i_b]
        self.Nb      = len(self.bndr_i_b)
           
        # Corners indexes (boundary DOFs)
        self.Corner_indices = []
        for i in range(self.Nb) :
            if  ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) : 
                 self.Corner_indices.append(i)
         
        self.set_FE_approximation_spaces = 1
        
        return self.set_FE_approximation_spaces
    
    ## Method.
    #  @param self The object pointer.   
    def Apply_PFEM_formulation(self, formulation):
        """
        Perform the matrix assembly related to the PFEM formulation
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_FE_approximation_spaces == 1, \
                "The FE approximation spaces must be selected first"
        
        # Functions
        aq, ap, ab = TrialFunction(self.Vq), TrialFunction(self.Vp), TrialFunction(self.Vb)
        vq, vp, vb = TestFunction(self.Vq), TestFunction(self.Vp), TestFunction(self.Vb)
        
        # Mass matrices
        self.Mq = assemble( dot(aq, vq) * dx).array()
        self.Mp = assemble( ap * vp * dx).array()
        self.M  = linalg.block_diag(self.Mq, self.Mp)
        
        # Mass matrices with coefficients
        self.M_T   = assemble( dot(self.T*aq, vq) * dx).array()
        self.M_rho = assemble( 1/self.rho * ap * vp * dx).array()
        self.M_X   = linalg.block_diag(self.M_T, self.M_rho)

        # Stiffness matrices
        if formulation == 'div' :
            self.D = assemble( - ap * div(vq) * dx).array()
        elif formulation == 'grad' :
            self.D = assemble( dot(grad(ap), vq) * dx).array()

        self.J = np.vstack([np.hstack([ np.zeros((self.Nq,self.Nq)),          self.D        ]), 
                            np.hstack([       -self.D.T       ,  np.zeros((self.Np,self.Np))]) ])
        
        # Physical paramater matrices
        self.Q_T   = linalg.solve(self.Mq, self.M_T)
        self.Q_rho = linalg.solve(self.Mp, self.M_rho)
        self.Q     = linalg.block_diag(self.Q_T, self.Q_rho)
            
        # Boundary matrices
        self.Mb = assemble( ab * vb * ds).array()[self.bndr_i_b,:][:,self.bndr_i_b]
        
        if formulation == 'div' :
            self.B    = assemble( ab * dot(vq, self.norext) * ds).array()[:,self.bndr_i_b] 
            self.Bext = np.concatenate((self.B,np.zeros((self.Np,self.Nb))))
        if formulation == 'grad' :
            self.B    = assemble( ab * vp * ds).array()[:,self.bndr_i_b] 
            self.Bext = np.concatenate((np.zeros((self.Nq,self.Nb)), self.B))

        # Impedance matrices
        if self.set_impedance == 1 :
            self.Mz = assemble( ab * self.z * vb * ds).array()[self.bndr_i_b,:][:,self.bndr_i_b]
            if self.discontinuous_impedance :
                self.Mz[self.Corners_indexes,:][:,self.Corners_indexes] = self.Mz[self.Corners_indexes,:][:,self.Corners_indexes] / 2.
            self.Zd = linalg.solve(self.Mb, self.Mz)
            self.Rz = self.B @ self.Zd @ linalg.solve(self.Mb, self.B.T) 
            self.R  = linalg.block_diag(self.Rz, np.zeros((self.Np,self.Np)))
        else:
            self.R = np.zeros((self.Nq+self.Np, self.Nq+self.Np))
        
     
        self.apply_PFEM_formulation = 1
        
        return self.apply_PFEM_formulation
 
    
    ## Method.
    #  @param self The object pointer. 
    def Project_initial_data_FE_spaces(self, W0, Aq_0_1, Aq_0_2, Ap_0, **kwargs):
        """
        Project initial data on the FE spaces 
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_FE_approximation_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        # Expressions
        W_0  = Expression(W0, degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        Aq_0 = Expression((Aq_0_1, Aq_0_2), degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs) 
        Ap_0 = Expression('0', degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs) 
            
        # Vectors
        self.W0  = interpolate(W_0, self.Vp).vector()[:]
        self.Aq0 = interpolate(Aq_0, self.Vq).vector()[:]
        self.Ap0 = interpolate(Ap_0, self.Vp).vector()[:] * interpolate(self.rho, self.Vp).vector()[:]
        self.A0  = np.concatenate((self.Aq0, self.Ap0))
        
        self.project_initial_data_FE_spaces = 1
        
        return self.project_initial_data_FE_spaces
        
    ## Method.
    #  @param self The object pointer.           
    def Set_time_setting(self, time_step, theta, string_solver):
        """
        Specify the parameters related to the time integration
        string_solver specifies the method to be used.
        """
        self.dt     = time_step
        self.Nt     = int( np.floor(self.tfinal/self.dt) )
        self.tspan  = np.linspace(0,self.tfinal,self.Nt+1)
        self.Nclose = int(np.floor(self.tclose/self.dt) )
        self.theta  = theta
        
        self.time_method = string_solver
        
        self.set_time_setting = 1
        
        return self.set_time_setting
       
    #
    # Numerical approximation schemes
    #
    ## Method.
    #  @param self The object pointer.      
    def Convert_into_sparse_format(self):
        """
        Convert into sparse format
        """
        
        self.M      = csc_matrix(self.M)
        self.J      = csr_matrix(self.J)
        self.R      = csr_matrix(self.R)
        self.Q      = csr_matrix(self.Q)
        self.Bext   = csr_matrix(self.Bext)
        
        self.sparse = 1
        
        return self.sparse
    
    ## Method.
    #  @param self The object pointer.      
    def Time_integration(self, string_mode, **kwargs):
        """
        Wrapper method for the time integration
        """
        done = 0 
        
        self.Convert_into_sparse_format()
        
        if string_mode == 'ODE:Crank-Nicolson': 
            Aq, Ap, Ham, W, t_span = self.integration_theta_scheme_sparse()
            done           = 1
        
        if string_mode == 'ODE:Assimulo': 
            Aq, Ap, Ham, W, t_span = self.integration_assimulo(**kwargs)
            done           = 1  
            
        if string_mode == 'ODE:Scipy': 
            Aq, Ap, Ham, W, t_span = self.integration_scipy(**kwargs)
            done           = 1            
        
        if string_mode == 'DAE:Assimulo': 
            Aq, Ap, Ham, W, t_span = self.DAE_integration_assimulo(**kwargs)
            done           = 1 
        
        assert done == 1, "Unknown time discretization method in Time_integration"
        
        index_list   = np.where(abs(self.tspan-self.tclose)==abs(self.tspan-self.tclose).min())[0]
        self.Nclose  = index_list[0]
        
        return Aq, Ap, Ham, W, t_span
    
    
    ## Method.
    #  @param self The object pointer.       
    def integration_theta_scheme(self, theta, ctrl_time, close=False):
        """
        $\theta$-scheme for the numerical integration of the ODE system
        """
        if close:
            close_at = self.tclose
        
        # Control vector
        self.U = interpolate(self.boundary_cntrl_space, self.Vb).vector()[self.bndr_i_b]
        
        if self.discontinous_boundary_values == 1:
            self.U[self.Corner_indices] = self.U[self.Corner_indices]/2.
        
        # Solution and Hamiltonian versus time
        A_th   = np.zeros( (self.Nq+self.Np, self.Nt+1) )
        Ham_th = np.zeros(self.Nt+1)

        # Initialization
        A_th[:,0] = self.A0
        Ham_th[0] = 1/2 * A_th[:,0] @ self.M @ self.Q @ A_th[:,0]

        # Inversion of matrix system
        print('Matrices inversion for time-stepping')
        print('This may take a while\n')
        
        Sysopen = linalg.solve(self.M - self.dt*theta * self.J @ self.Q, self.M + self.dt * (1-theta) * self.J @ self.Q)
        
        if self.set_impedance : 
            Sysdis = linalg.solve(self.M - self.dt*theta * (self.J-self.R) @ self.Q, self.M + self.dt * (1-theta) * (self.J-self.R) @ self.Q)
        else : 
            Sysdis = Sysopen
            
        Sys_ctrl = linalg.solve(self.M - self.dt * theta * (self.J-self.R) @ self.Q, self.dt* self.Bext)

        # Time loop
        for n in range(self.Nt):   
            # Specify system
            if close :
                if n < int(close_at/self.tfinal * self.Nt) : 
                    Sys = Sysopen
                elif n >= int(close_at/self.tfinal * self.Nt) : 
                    Sys = Sysdis
            else : 
                Sys = Sysdis
            
            # Iterations
            A_th[:,n+1] = Sys @ A_th[:,n] + Sys_ctrl @ self.U * self.boundary_cntrl_time(self.tspan[n+1],self.tclose)
            Ham_th[n+1] = 1/2 * A_th[:,n+1] @ self.M @ self.Q @ A_th[:,n+1]
            
            # Progress bar
            perct = int(n/(self.Nt-1) * 100)  
            bar   = ('Time-stepping for energy variables \t' + '[' + '#' * int(perct/2) + ' ' + str(perct) + '%' + ']')
            sys.stdout.write('\r' + bar)
        print('\t')
        
        # Get q variables
        Aq_th = A_th[:self.Nq,:] 
        
        # Get p variables
        Ap_th = A_th[self.Nq:,:]

        # Get Hamiltonian
        self.Ham_th = Ham_th[:]
           
        # Get Deformation
        Rho = np.zeros(self.Np)
        for i in range(self.Np):
            Rho[i] = self.rho(self.coord_p[i])
            
        W_th      = np.zeros((self.Np,self.Nt+1))
        W_th[:,0] = self.W0[:]
        
        for n in range(self.Nt):
            W_th[:,n+1] = W_th[:,n] + self.dt * 1/Rho[:] * ( theta * Ap_th[:,n+1] + (1-theta) * Ap_th[:,n] ) 
            perct       = int(n/(self.Nt-1) * 100)  
            bar         = ('Time-stepping to get deflection \t' + '[' + '#' * int(perct/2) + ' ' + str(perct) + '%' + ']')
            sys.stdout.write('\r' + bar)

        print('Done')
        
        return Aq_th, Ap_th, Ham_th, W_th

    ## Method.
    #  @param self The object pointer.       
    def integration_theta_scheme_sparse(self):
        """
        $\theta$-scheme for the numerical integration of the ODE system
        """
        
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        theta     = self.theta
        time_span = self.tspan
        
        if self.tclose > 0:
            close    = True
            close_at = self.tclose
            
        else:
            close    = False
        
        # Control vector
        self.U = interpolate(self.boundary_cntrl_space, self.Vb).vector()[self.bndr_i_b]
        
        if self.discontinous_boundary_values == 1:
            self.U[self.Corner_indices] = self.U[self.Corner_indices]/2.
        
        # Solution and Hamiltonian versus time
        A_th   = np.zeros( (self.Nq+self.Np, self.Nt+1) )
        Ham_th = np.zeros(self.Nt+1)

        # Initialization
        A_th[:,0] = self.A0
        Ham_th[0] = 1/2 * A_th[:,0] @ self.M @ self.Q @ A_th[:,0]

        # Definition of the different matrices 
        
        C_open_imp  = self.M - self.dt*theta * self.J @ self.Q
        C_open_exp  = self.M + self.dt*(1.-theta) * self.J @ self.Q
        
        if close:
            C_close_imp = self.M - self.dt*theta * (self.J - self.R) @ self.Q
            C_close_exp = self.M + self.dt*(1.-theta) * (self.J - self.R) @ self.Q
                
        # LU factorization of the implicit matrices
        if close:
            my_solver_close = factorized(csc_matrix(C_close_imp))
            
        my_solver_open  = factorized(csc_matrix(C_open_imp))
            
        # Time loop
        for n in range(self.Nt):   
            # Specify system
            if close :
                if n < int(close_at/self.tfinal * self.Nt) : 
                    my_solver = my_solver_open
                    rhs = self.my_mult(C_open_exp,A_th[:,n]) + \
                          theta     * self.dt* self.my_mult(self.Bext, self.U * self.boundary_cntrl_time(self.tspan[n+1],self.tclose)) + \
                          (1.-theta)* self.dt* self.my_mult(self.Bext, self.U * self.boundary_cntrl_time(self.tspan[n],self.tclose))
                elif n >= int(close_at/self.tfinal * self.Nt) : 
                    my_solver = my_solver_close
                    rhs = self.my_mult(C_close_exp,A_th[:,n]) + \
                          theta     * self.dt* self.my_mult(self.Bext, self.U * self.boundary_cntrl_time(self.tspan[n+1],self.tclose)) + \
                          (1.-theta)* self.dt* self.my_mult(self.Bext, self.U * self.boundary_cntrl_time(self.tspan[n],self.tclose))
            else : 
                my_solver = my_solver_open
                rhs = self.my_mult(C_open_exp,A_th[:,n]) + \
                          theta     * self.dt* self.my_mult(self.Bext, self.U * self.boundary_cntrl_time(self.tspan[n+1],self.tclose)) + \
                          (1.-theta)* self.dt* self.my_mult(self.Bext, self.U * self.boundary_cntrl_time(self.tspan[n],self.tclose))
            
            # Iterations
            A_th[:,n+1] = my_solver(rhs)
            Ham_th[n+1] = 1/2 * A_th[:,n+1] @ self.M @ self.Q @ A_th[:,n+1]
            
            # Progress bar
            perct = int(n/(self.Nt-1) * 100)  
            bar   = ('Time-stepping for energy variables \t' + '[' + '#' * int(perct/2) + ' ' + str(perct) + '%' + ']')
            sys.stdout.write('\r' + bar )
        print('\t')
        
        # Get q variables
        Aq_th = A_th[:self.Nq,:] 
        
        # Get p variables
        Ap_th = A_th[self.Nq:,:]

        # Get Hamiltonian
        self.Ham_th = Ham_th[:]
           
        # Get Deformation
        Rho = np.zeros(self.Np)
        for i in range(self.Np):
            Rho[i] = self.rho(self.coord_p[i])
            
        W_th      = np.zeros((self.Np,self.Nt+1))
        W_th[:,0] = self.W0[:]
        
        for n in range(self.Nt):
            W_th[:,n+1] = W_th[:,n] + self.dt * 1/Rho[:] * ( theta * Ap_th[:,n+1] + (1-theta) * Ap_th[:,n] ) 
            perct       = int(n/(self.Nt-1) * 100)  
            bar         = ('Time-stepping to get deflection \t' + '[' + '#' * int(perct/2) + ' ' + str(perct) + '%' + ']')
            sys.stdout.write('\r' + bar)

        return Aq_th, Ap_th, Ham_th, W_th, time_span
   
    ## Method.
    #  @param self The object pointer.    
    def integration_assimulo(self, **kwargs):
        """
        Perform time integration for ODEs with the assimulo package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        if self.tclose > 0:
            close    = True
        else: 
            close    = False

        # Control vector
        self.U = interpolate(self.boundary_cntrl_space, self.Vb).vector()[self.bndr_i_b]
        if self.discontinous_boundary_values == 1:
            self.U[self.Corner_indices] = self.U[self.Corner_indices]/2

        # Definition of the sparse solver for the ODE rhs function to
        # be defined next
        #my_solver = factorized(csc_matrix(self.M))
        my_solver = factorized(self.M)
        #my_jac_o  = csr_matrix(my_solver(self.J @ self.Q))
        #my_jac_c  = csr_matrix(my_solver((self.J - self.R) @ self.Q))
                
        # Definition of the rhs function required in assimulo
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of assimulo
            """   
            if close:
                if t < self.tclose:
                    z = self.my_mult(self.J, self.my_mult(self.Q,y)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose))
                else:
                    z = self.my_mult((self.J - self.R), self.my_mult(self.Q,y))
            else:
                z = self.my_mult(self.J, self.my_mult(self.Q,y)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose)) 
            
            return my_solver(z)
 
        def jacobian(t,y):
            """
            Jacobian related to the ODE formulation
            """
            if close:
                if t < self.tclose:
                    my_jac = my_jac_o
                else:
                    my_jac = my_jac_c
            else:
                my_jac = my_jac_o
            
            return my_jac
        
        def jacv(t,y,fy,v):
            """
            Jacobian matrix-vector product related to the ODE formulation
            """
            if close:
                if t < self.tclose:
                    z = self.my_mult(self.J, self.my_mult(self.Q,v) )
                else:
                    z = self.my_mult((self.J - self.R), self.my_mult(self.Q,v))
            else:
                z = self.my_mult(self.J, self.my_mult(self.Q,v))
            
            return my_solver(z)
           
        print('ODE Integration using assimulo built-in functions:')

#
# https://jmodelica.org/assimulo/_modules/assimulo/examples/cvode_with_preconditioning.html#run_example
#
        
        model                     = Explicit_Problem(rhs,self.A0,self.tinit)
        #model.jac                 = jacobian
        model.jacv                = jacv
        sim                       = CVode(model,**kwargs)
        sim.atol                  = 1e-3 
        sim.rtol                  = 1e-3 
        sim.linear_solver         = 'SPGMR' 
        sim.maxord                = 3
        #sim.usejac                = True
        #sim                       = RungeKutta34(model,**kwargs)
        time_span, ODE_solution   = sim.simulate(self.tfinal)
        
        A_ode = ODE_solution.transpose()
        
        # Hamiltonian
        self.Nt    = A_ode.shape[1]
        self.tspan = np.array(time_span)
        
        Ham_ode = np.zeros(self.Nt)
        
        for k in range(self.Nt):
            #Ham_ode[k] = 1/2 * A_ode[:,k] @ self.M @ self.Q @ A_ode[:,k]
            Ham_ode[k] = 1/2 * self.my_mult(A_ode[:,k].T, \
                               self.my_mult(self.M, self.my_mult(self.Q, A_ode[:,k])))
      
        # Get q variables
        Aq_ode = A_ode[:self.Nq,:] 
        
        # Get p variables
        Ap_ode = A_ode[self.Nq:,:]

        # Get Deformation
        Rho = np.zeros(self.Np)
        for i in range(self.Np):
            Rho[i] = self.rho(self.coord_p[i])
            
        W_ode = np.zeros((self.Np,self.Nt))
        theta = .5
        for k in range(self.Nt-1):
            W_ode[:,k+1] = W_ode[:,k] + self.dt * 1/Rho[:] * ( theta * Ap_ode[:,k+1] + (1-theta) * Ap_ode[:,k] ) 

        self.Ham_ode = Ham_ode
    
        return Aq_ode, Ap_ode, Ham_ode, W_ode, np.array(time_span)
    
    
     ## Method.
    #  @param self The object pointer.    
    def integration_scipy(self, **kwargs):
        """
        Perform time integration for ODEs with the scipy.integrate.IVP package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        if self.tclose > 0:
            close    = True
        else: 
            close    = False

        # Control vector
        self.U = interpolate(self.boundary_cntrl_space, self.Vb).vector()[self.bndr_i_b]
        if self.discontinous_boundary_values == 1:
            self.U[self.Corner_indices] = self.U[self.Corner_indices]/2

        # Definition of the sparse solver for the ODE rhs function to
        # be defined next
        my_solver = factorized(csc_matrix(self.M))
        #my_jac_o  = csr_matrix(my_solver(self.my_mult(self.J,self.Q)))
        #my_jac_c  = csr_matrix(my_solver(self.my_mult(self.J - self.R, self.Q)))
        C_o       = self.my_mult(self.J,self.Q)
        C_c       = self.my_mult(self.J - self.R, self.Q)
        
        if self.sparse == 1:
            my_jac_c  = csr_matrix(my_solver(C_c.toarray()))
            my_jac_o  = csr_matrix(my_solver(C_o.toarray()))
        else:
            my_jac_c  = my_solver(C_c)
            my_jac_o  = my_solver(C_o)
        
        # Definition of the rhs function required in assimulo
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of IVP
            """   
            if close:
                if t < self.tclose:
                    z = self.my_mult(self.J, self.my_mult(self.Q,y)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose))
                else:
                    z = self.my_mult((self.J - self.R), self.my_mult(self.Q,y))
            else:
                z = self.my_mult(self.J, self.my_mult(self.Q,y)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose)) 
            
            return my_solver(z)
        
        def jacobian(t,y):
            """
            Jacobian matrix related to the ODE
            """
            if close:
                if t < self.tclose:
                    my_jac = my_jac_o
                else:
                    my_jac = my_jac_c
            else:
                my_jac = my_jac_o 
            
            return my_jac
            
        print('ODE Integration using scipy.integrate built-in functions:')

        ivp_ode    = integrate.solve_ivp(rhs, (self.tinit,self.tfinal), self.A0, **kwargs, jac= jacobian, atol=1.e-3)   
        
        A_ode      = ivp_ode.y
        self.tspan = np.array(ivp_ode.t)
        self.Nt    = len(self.tspan)
        
        print("Scipy: Number of evaluations of the right-hand side ",ivp_ode.nfev)
        print("Scipy: Number of evaluations of the Jacobian ",ivp_ode.njev)
        print("Scipy: Number of LU decompositions ",ivp_ode.nlu)
                        
        # Hamiltonian 
        Ham_ode    = np.zeros(self.Nt)
        
        for k in range(self.Nt):
            #Ham_ode[k] = 1/2 * A_ode[:,k] @ self.M @ self.Q @ A_ode[:,k]
            Ham_ode[k] = 1/2 * self.my_mult(A_ode[:,k].T, \
                               self.my_mult(self.M, self.my_mult(self.Q, A_ode[:,k])))
        # Get q variables
        Aq_ode = A_ode[:self.Nq,:] 
        
        # Get p variables
        Ap_ode = A_ode[self.Nq:,:]

        # Get Deformation
        Rho = np.zeros(self.Np)
        for i in range(self.Np):
            Rho[i] = self.rho(self.coord_p[i])
            
        W_ode = np.zeros((self.Np,self.Nt))
        theta = .5
        for k in range(self.Nt-1):
            W_ode[:,k+1] = W_ode[:,k] + self.dt * 1/Rho[:] * ( theta * Ap_ode[:,k+1] + (1-theta) * Ap_ode[:,k] ) 

        self.Ham_ode = Ham_ode
    
        return Aq_ode, Ap_ode, Ham_ode, W_ode, np.array(ivp_ode.t)
   
    ## Method.
    #  @param self The object pointer.    
    def DAE_integration_assimulo(self, **kwargs):
        """
        Perform time integration for DAEs with the assimulo package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        if self.tclose > 0:
            close    = True
        else:
            close    = False
            
        # Control vector
        self.U = interpolate(self.boundary_cntrl_space, self.Vb).vector()[self.bndr_i_b]
        if self.discontinous_boundary_values == 1:
            self.U[self.Corner_indices] = self.U[self.Corner_indices]/2

        # Definition of the sparse solver for the DAE res function to
        # be defined next M should be invertible !! 
        my_solver = factorized(csc_matrix(self.M))
        rhs       = self.my_mult(self.J, self.my_mult(self.Q,self.A0)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(0.,self.tclose))
        self.AD0  = my_solver(rhs) 
        
        # Definition of the rhs function required in assimulo
        def res(t,y,yd):
            """
            Definition of the residual function required in the DAE part of assimulo
            """   
            if close:
                if t < self.tclose:
                    z = self.my_mult(self.M,yd) - self.my_mult(self.J, self.my_mult(self.Q,y)) - self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose))
                else:
                    z = self.my_mult(self.M,yd) - self.my_mult((self.J - self.R), self.my_mult(self.Q,y))
            else:
                z = self.my_mult(self.M,yd) - self.my_mult(self.J, self.my_mult(self.Q,y)) - self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose)) 
            
            return z
  

        # Definition of the jacobian function required in assimulo
        def jac(c,t,y,yd):
            """
            Definition of the Jacobian matrix required in the DAE part of assimulo
            """  
            Matrix = csr_matrix(self.my_mult(self.J,self.Q))
            
            if close and t > self.tclose:
                    Matrix = csr_matrix(self.my_mult(self.J - self.R, self.Q))
            
            return c*csr_matrix(self.M) - Matrix
        
        # Definition of the jacobian matrix vector function required in assimulo
        def jacv(t,y,yd,res,v,c):
            """
            Jacobian matrix-vector product required in the DAE part of assimulo
            """  
            w = self.my_mult(self.Q, v)
            z = self.my_mult(self.J, w)
            
            if close and t > self.tclose:
                z -= self.my_mult(self.R, w)
                
            return c*self.my_mult(self.M,v) - z
        
        print('DAE Integration using assimulo built-in functions:')

        
        model                     = Implicit_Problem(res,self.A0,self.AD0,self.tinit)
        model.jacv                = jacv
        #sim                       = Radau5DAE(model,**kwargs)
        #
        # IDA method from Assimulo
        #
        sim                       = IDA(model,**kwargs)
        sim.algvar                = [1 for i in range(self.M.shape[0])]
        sim.atol                  = 1.e-6
        sim.rtol                  = 1.e-6
        sim.report_continuously   = True
        ncp                       = self.Nt
        sim.usejac                = True
        sim.suppress_alg          = True
        sim.inith                 = self.dt
        sim.maxord                = 5
        #sim.linear_solver         = 'SPGMR'
        time_span, DAE_y, DAE_yd  = sim.simulate(self.tfinal,ncp)
        
        #print(sim.get_options())
        print(sim.print_statistics())
        
        A_dae = DAE_y.transpose()
        
        # Hamiltonian
        self.Nt    = A_dae.shape[1]
        self.tspan = np.array(time_span)
        
        Ham_dae = np.zeros(self.Nt)
        
        for k in range(self.Nt):
            #Ham_dae[k] = 1/2 * A_dae[:,k] @ self.M @ self.Q @ A_dae[:,k]
            Ham_dae[k] = 1/2 * self.my_mult(A_dae[:,k].T, \
                               self.my_mult(self.M, self.my_mult(self.Q, A_dae[:,k])))
      
        # Get q variables
        Aq_dae = A_dae[:self.Nq,:] 
        
        # Get p variables
        Ap_dae = A_dae[self.Nq:,:]

        # Get Deformation
        Rho = np.zeros(self.Np)
        for i in range(self.Np):
            Rho[i] = self.rho(self.coord_p[i])
            
        W_dae = np.zeros((self.Np,self.Nt))
        theta = .5
        for k in range(self.Nt-1):
            W_dae[:,k+1] = W_dae[:,k] + self.dt * 1/Rho[:] * ( theta * Ap_dae[:,k+1] + (1-theta) * Ap_dae[:,k] ) 

        self.Ham_dae = Ham_dae
    
        return Aq_dae, Ap_dae, Ham_dae, W_dae, np.array(time_span)    
    #
    # POST-PROCESSING METHODS
    #
    
    ## Method.
    #  @param self The object pointer.
    def plot_mesh(self, **kwargs):
        """
        Plot the two-dimensional mesh with the Fenics plot method
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
     
        plot(self.Mesh, **kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.title("Finite element mesh")
        plt.savefig("Mesh.png")
        

    ## Method.
    #  @param self The object pointer.
    def plot_mesh_with_DOFs(self, **kwargs):
        """
        Plot the two-dimensional mesh with the Fenics plot method including DOFs
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"

        plt.figure()
        plot(self.Mesh, **kwargs)
        plt.plot(self.xp, self.yp, 'o', label='Dof of $p$ variables')
        plt.plot(self.coord_q[:,0], self.coord_q[:,1], '^', label='Dof of $q$ variables')
        plt.plot(self.coord_b[:,0], self.coord_b[:,1], 'k*', label='Dof of $\partial$ variables')
        plt.title('Mesh with associated DOFs, $Nq=$'+ str(self.Nq)+ ', $Np=$'+ str(self.Np)+ ', $N_\partial=$'+ str(self.Nb) )
        plt.legend()
        if not(self.notebook):
         plt.show()
        plt.savefig("Mesh_with_DOFs.png")
       
    ## Method.
    #  @param self The object pointer.
    def plot_Hamiltonian(self, tspan, Ham, **kwargs):
        """
        Plot the Hamiltonian function versus time 
        """
        plt.figure()
        plt.plot(tspan, Ham, **kwargs)
        plt.xlabel('Time $(s)$')
        plt.ylabel('Hamiltonian')
        plt.title('Hamiltonian')
        plt.grid(True)
        if not(self.notebook):
         plt.show()
        plt.savefig("Hamiltonian.png")

    # Set writer for video saving
    def set_video_writer(self, fps=128, dpi=256):
        """
        
        """
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata     = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
        self.writer  = FFMpegWriter(fps=fps, metadata=metadata)
        self.dpi     = dpi
    
    # Moving progressive plot
    def moving_plot(self, y, x,  step=1, title='', save=False):
        """
        Create a 2D animation with the plot command
        """
        fig = plt.figure()
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(y)*0.875, np.max(y)*1.125)
        
        # Save 
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                for i in range(0,self.Nt,step):
                    plt.plot(x[i:i+1], y[i:i+1], '.b')
                    plt.plot(x[0:i+1], y[0:i+1], '-b')
                    plt.title(title+ ' t='+np.array2string(x[i]) + '/' + np.array2string(x[-1]))
                    self.writer.grab_frame()
                    plt.pause(0.01)
        # Do not save
        else :
            for i in range(0,self.Nt,step):
                plt.plot(x[i:i+1], y[i:i+1], '.b')
                plt.plot(x[0:i+1], y[0:i+1], '-b')
                plt.title(title + ' t='+np.array2string(x[i]) + '/' + np.array2string(x[-1]))
                plt.pause(0.01)
    
    # Moving trisurf
    def moving_trisurf(self, time_space_Vector, step=1, title='', save=False):
        """
        Create a 3D animation with the plot_trisurf Matplotlib command
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #fig.colorbar(wfram, shrink=0.5, aspect=5)
        plt.title(title)
        wframe = None
        tstart = time.time()
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                for i in range(0, self.Nt, step):
                    if wframe:
                        ax.collections.remove(wframe)
                    wframe = ax.plot_trisurf(self.xp, self.yp, time_space_Vector[:,i], linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
                    ax.set_xlabel('x coordinate')
                    ax.set_ylabel('y coordinate')
                    ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]) ) 
                    ax.set_xlim(self.x0, self.xL)
                    ax.set_ylim(self.y0, self.yL)
                    ax.set_zlim(np.min(time_space_Vector), np.max(time_space_Vector))
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not save
        else:
            for i in range(0, self.Nt, step):
                if wframe:
                    ax.collections.remove(wframe)
                wframe = ax.plot_trisurf(self.xp, self.yp, time_space_Vector[:,i], linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
                ax.set_xlabel('x coordinate')
                ax.set_ylabel('y coordinate')
                ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]) ) 
                ax.set_xlim(self.x0, self.xL)
                ax.set_ylim(self.y0, self.yL)
                ax.set_zlim(np.min(time_space_Vector), np.max(time_space_Vector))
                plt.pause(.001)
        
    # Moving quiver
    def moving_quiver(self, time_space_Vector, step=1, with_mesh=False, title='', save=False):
        """
        Create a 2D animation with arrows on vector quantities
        """
        fig = plt.figure()
        ax = fig.gca()
        wframe = None
        tstart = time.time()
        temp_vec = Function(self.Vq)
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                if with_mesh : plot(self.Mesh, linewidth=.25)
                for i in range(0, self.Nt, step):
                    if wframe :
                        ax.collections.remove(wframe)
                    for k in range(self.Nq):
                        temp_vec.vector()[k] = time_space_Vector[k,i] 
                        
                    wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2)))
                    ax.set_xlabel('x coordinate')
                    ax.set_ylabel('y coordinate')
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not Save
        else :
            if with_mesh : plot(self.Mesh, linewidth=.25)
            for i in range(0, self.Nt, step):
                if wframe :
                    ax.collections.remove(wframe)
                for k in range(self.Nq):
                    temp_vec.vector()[k] = time_space_Vector[k,i] 
                    
                wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2))+'/'+str(self.tfinal))
                ax.set_xlabel('x coordinate')
                ax.set_ylabel('y coordinate')
                plt.pause(.001)
    
    # Plot contour at a chosen time
    def plot_contour(self, time_space_Vector, t, title):
        return 0
    
    # Plot 3D at a chosen time
    def plot_3D(self, time_space_Vector, t, title):
        """
        Create a 3D plot at a specific time t
        """
        # Find the index of the nearest time with t
        index_list   = np.where(abs(self.tspan-t)==abs(self.tspan-t).min())[0]
        i            = index_list[0]
        
        fig = plt.figure()
        ax  = fig.gca(projection='3d')
        ax.plot_trisurf(self.xp, self.yp, time_space_Vector[:,i], linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]))
        ax.set_title(title)
        if not(self.notebook):
         plt.show()
        plt.savefig("Space_Time_plot.png")
  
    
    
if __name__ == '__main__':
    
    #
    # Create the Wave_2D object by calling the constructor of the class
    #
    
    x0, xL, y0, yL = 0., 2., 0., 1.
    
    W = Wave_2D()
     
    #
    # Define Fenics Expressions
    #
    
    Rho = 'x[0]*x[0] * (xL-x[0])+ 1'
    T11 = 'x[0]*x[0]+1'
    T12 = 'x[1]'
    T22 = 'x[0]+1'
    Z   = '0.1'
    
    
    
    W0     = '0'
    Ap_0   = '0'
    Aq_0_1 = '0'
    Aq_0_2 = '0'
    
    boundary_cntrl_space='x[0] * ( sin(2*pi*x[1]) + 1)'
    
    # Time part of the control
    def boundary_cntrl_time(t,close_time):
        """
        Specify here the time part of the boundary control as a pure Python method. 
        """        
        if t <= close_time:
            return np.sin(2*np.pi*t)
        else:
            return 0.
    
    #
    # Define constants related to the time discretization
    #
    tinit  = 0.
    tfinal = 6.
    tclose = 1.
    dt     = 1.e-1
    theta  = 0.5
    
    #
    # Problem Definition
    #
    
    W.Set_rectangular_domain(x0, xL, y0, yL)
    
    W.Set_physical_parameters(Rho, T11, T12, T22)
    
    W.Set_impedance(Z)
    
    W.Set_boundary_cntrl_space(boundary_cntrl_space)
    
    W.Set_boundary_cntrl_time(boundary_cntrl_time)
    
    W.Set_initial_close_final_time(tinit,tclose,tfinal)
    
    assert W.Check_Problem_Definition() == 1, "Problem definition \
                to be checked"
    
    #
    # Finite Element Space and Time discretizations
    # 
    
    W.Generate_mesh(rfn=10)
    
    W.Set_FE_Approximation_Spaces(family_q='RT', family_p='P', family_b='P',\
                                  rt_order=0, p_order=1,b_order=1)
    
    W.Apply_PFEM_formulation(formulation='div')
    
    W.Project_initial_data_FE_spaces(W0, Aq_0_1, Aq_0_2, Ap_0)
    
    time_method = 'ODE:Assimulo'
    
    W.Set_time_setting(dt, theta, time_method)
    
    assert W.Check_Space_Time_Discretization() == 1, "Space_Time_Discretization \
                to be checked"
    
    #
    # Numerical Approximations
    #
    
    Aq, Ap, Ham, Deformation, t_span = W.Time_integration(time_method)
    
    #time_method = 'ODE:Scipy'
    #W.Set_time_setting(dt, theta, time_method)
    #Aq_s, Ap_s, Ham_s, Deformation_s, t_span_s = W.Time_integration(time_method,method='BDF')
    
   
    #
    # Post-processing
    # 
    W.plot_mesh_with_DOFs()    
    W.plot_Hamiltonian(t_span,Ham,marker='o')
    #W.plot_Hamiltonian(t_span_s,Ham_s,marker='s')
    #plt.figure()
    #plt.plot(t_span,Ham,'-o',t_span_s,Ham_s,'-s')
    #plt.show()
    W.plot_3D(Deformation,tfinal/2,'deflection at t='+str(tfinal/2))
    W.set_video_writer()
    W.moving_trisurf(Deformation,1,'movie_ode',save=True)
        
    #
    # End
    #
        
        
        
        
        
        
        
      
        
