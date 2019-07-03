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


class Heat_2D:
    # Constructor of the domain
    def __init__(self):
        """
        Constructor for the Heat_2D class.
        """
        # Information related to the problem definition
        self.set_domain                = 0
        self.set_boundary_cntrl_space  = 0
        self.set_boundary_cntrl_time   = 0
        self.set_physical_parameters   = 0
        self.set_isochoric             = 0
        self.set_initial_close_final_time    = 0
        self.problem_definition        = 0
        self.discontinous_boundary_values = 0
        self.discontinuous_isochoric   = 0
        
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
        
        self.debug       = 0

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
            
        assert self.set_isochoric == 1, \
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
    def Set_isochoric_heat_capacity(self, Z, discontinuous=False, **kwargs):
        """
        Set the isochoric heat capacity as a regular FeniCS expression
        """
        self.z = Expression(Z, degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        
        if discontinuous : 
            self.discontinuous_isochoric = 1 
        
        self.set_isochoric = 1
        
        return self.set_isochoric
    
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
        
        self.T = interpolate(self.T, TensorFunctionSpace(self.Mesh, 'P', 1))
        
        return self.generate_mesh

    ## Method.
    #  @param self The object pointer.   
    def Set_FE_Approximation_Spaces(self, family_q, family_p, family_b, family_l, rt_order, p_order, b_order, l_order):
        """
        Set the finite element approximation spaces related to the space discretization of the PDE
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        # Orders
        self.rt_order = rt_order
        self.p_order  = p_order
        self.b_order  = b_order
        self.l_order  = l_order
        
        # Spaces
        
        
        RT      = FiniteElement('RT', self.Mesh.ufl_cell(),self.rt_order+1)
        P       = FiniteElement('P' , self.Mesh.ufl_cell(),self.p_order)
        
        self.Vq = FunctionSpace(self.Mesh, RT)
        self.Vp = FunctionSpace(self.Mesh, P)
        self.Vl = FunctionSpace(self.Mesh, 'P', self.l_order)
        self.Vb = FunctionSpace(self.Mesh, 'P', self.b_order)
        
        #self.Vq = FunctionSpace(self.Mesh, family_q, self.rt_order+1)
        
        #if self.p_order == 0 and family_p == 'P':
        #    self.Vp = FunctionSpace(self.Mesh, 'DG', 0)
        #else :
        #    self.Vp = FunctionSpace(self.Mesh, family_p, self.p_order)

        #if self.b_order == 0  and family_b == 'P':
        #    self.Vb = FunctionSpace(self.Mesh, 'CR', 1)
        #else :
        #    self.Vb = FunctionSpace(self.Mesh, family_b, self.b_order)
            
        #if self.l_order == 0  and family_l == 'P':
        #    self.Vl = FunctionSpace(self.Mesh, 'CR', 1)
        #else :
        #    self.Vl = FunctionSpace(self.Mesh, family_l, self.l_order) 
        
        # DOFs
        self.Nq = self.Vq.dim()
        self.Np = self.Vp.dim()
        self.coord_q = self.Vq.tabulate_dof_coordinates()
        self.coord_p = self.Vp.tabulate_dof_coordinates()
        self.coord_b = self.Vb.tabulate_dof_coordinates()
        self.coord_l = self.Vl.tabulate_dof_coordinates()
        
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
        
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_l  = self.Vl.tabulate_dof_coordinates()
        xl       = coord_b[:,0]
        yl       = coord_b[:,1]
        bndr_i_l = []
        for i in range(self.Vl.dim()) :
            if np.abs(xl[i] - self.x0) <= 1e-16 or np.abs(xl[i] - self.xL) <= 1e-16 or np.abs(yl[i] - self.y0) <= 1e-16 or np.abs(yl[i] - self.yL) <= 1e-16 : 
                 bndr_i_l.append(i)
        self.bndr_i_l = bndr_i_l      
 
        # Exlpicit information about boundary DOFs 
        self.coord_l = coord_l[bndr_i_l,:]
        self.xl      = xl[bndr_i_b]
        self.yl      = yl[bndr_i_b]
        self.Nl      = len(self.bndr_i_l)
           
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
    def Apply_PFEM_formulation(self):
        """
        Perform the matrix assembly related to the PFEM formulation
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_FE_approximation_spaces == 1, \
                "The FE approximation spaces must be selected first"
        
        # Functions
        aq, ap, ab, al = TrialFunction(self.Vq), TrialFunction(self.Vp), TrialFunction(self.Vb), TrialFunction(self.Vl)
        vq, vp, vb, vl = TestFunction(self.Vq), TestFunction(self.Vp), TestFunction(self.Vb), TestFunction(self.Vl)
        
        # Mass matrices
        self.Mq        = assemble( dot(aq, vq) * dx).array()
        self.Mp        = assemble( ap * vp * dx).array()
        self.Mp_rho    = assemble( self.rho * ap * vp * dx).array()
        self.Mp_rho_Cv = assemble( self.rho * self.z * ap * vp * dx).array()
        #self.M         = linalg.block_diag(self.Mq, self.Mp)
        
        # Mass matrices with coefficients
        #self.M_T   = assemble( dot(self.T*aq, vq) * dx).array()
        #self.M_rho = assemble( 1/self.rho * ap * vp * dx).array()
        #self.M_X   = linalg.block_diag(self.M_T, self.M_rho)

        # Interconnection matrices
        self.D = assemble( - vp * div(aq) * dx).array()
        self.L = assemble( dot(self.T * aq, vq) * dx ).array()

        # self.J = np.vstack([np.hstack([ np.zeros((self.Nq,self.Nq)),          self.D        ]), 
        #                    np.hstack([       -self.D.T       ,  np.zeros((self.Np,self.Np))]) ])
        
        # Physical paramater matrices
        #self.Q_T   = linalg.solve(self.Mq, self.M_T)
        #self.Q_rho = linalg.solve(self.Mp, self.M_rho)
        #self.Q     = linalg.block_diag(self.Q_T, self.Q_rho)
            
        # Boundary matrices
        self.Mb = assemble( ab * vb * ds).array()[self.bndr_i_b,:][:,self.bndr_i_b]
        
        # Control and Lagrange multiplier related matrices
        self.Bp    = assemble( - ab * dot(vq, self.norext) * ds).array()[:,self.bndr_i_b] 
        self.Bl    = assemble( - ab * vl * ds).array()[self.bndr_i_l,:][:,self.bndr_i_b]
        
        if self.debug:
            print("Norms")
            print("Mq",np.linalg.norm(self.Mq))
            print("Mp",np.linalg.norm(self.Mp))
            print("Mp_rho_Cv",np.linalg.norm(self.Mp_rho_Cv))
            print("D",np.linalg.norm(self.D))
            print("L",np.linalg.norm(self.L))
            print("Bp",np.linalg.norm(self.Bp))
            print("Bl",np.linalg.norm(self.Bl))
        
        #self.Bext = np.concatenate((self.B,np.zeros((self.Np,self.Nb))))
        #if formulation == 'grad' :
        #    self.B    = assemble( ab * vp * ds).array()[:,self.bndr_i_b] 
        #    self.Bext = np.concatenate((np.zeros((self.Nq,self.Nb)), self.B))

        # Impedance matrices
        #if self.set_isochoric == 1 :
        #    self.Mz = assemble( ab * self.z * vb * ds).array()[self.bndr_i_b,:][:,self.bndr_i_b]
        #    if self.discontinuous_isochoric :
        #        self.Mz[self.Corners_indexes,:][:,self.Corners_indexes] = self.Mz[self.Corners_indexes,:][:,self.Corners_indexes] / 2.
        #    self.Zd = linalg.solve(self.Mb, self.Mz)
        #    self.Rz = self.B @ self.Zd @ linalg.solve(self.Mb, self.B.T) 
        #    self.R  = linalg.block_diag(self.Rz, np.zeros((self.Np,self.Np)))
        #else:
        #self.R = np.zeros((self.Nq+self.Np, self.Nq+self.Np))
        
        # Post-processing matrices
        self.BndCtrl     = linalg.solve(self.Mb, assemble( ap * vb * ds ).array()[self.bndr_i_b, :] )
        self.BndObsrv    = linalg.solve(self.Mb, assemble( - dot(self.T * grad(ap), self.norext) * vb * ds ).array()[self.bndr_i_b, :] )
        self.BndObsrvDAE = linalg.solve(self.Mb, assemble(dot(aq, self.norext) * vb * ds ).array()[self.bndr_i_b, :])
        self.BndObsrvLAG = linalg.solve(self.Mb, -self.Bl.T)

        # Dense matrices [TO BE IMPROVED LATER]
        self.A    = - self.D @ linalg.solve(self.Mq, self.L) @ linalg.solve(self.Mq, self.D.T)
        self.Bext =   self.D @ linalg.solve(self.Mq, self.L) @ linalg.solve(self.Mq, self.Bp)
        
        # Matrices related to the Lagrange multiplier variant
        self.M_class = assemble( self.rho * self.z * ap * vp * dx).array()
        self.D_class = assemble( dot( self.T * grad(ap), grad(vp) ) * dx).array()
        self.L_class = assemble( self.boundary_cntrl_space * vl * ds)[self.bndr_i_l]
        self.C_class = assemble( al * vp * ds ).array()[:,self.bndr_i_l]
        
        if self.debug:
            print("Norms classic")
            print(np.linalg.norm(self.M_class))
            print(np.linalg.norm(self.D_class))
            print(np.linalg.norm(self.L_class))
            print(np.linalg.norm(self.C_class))
        
        self.apply_PFEM_formulation = 1
        
        return self.apply_PFEM_formulation
 
    
    ## Method.
    #  @param self The object pointer. 
    def Project_initial_data_FE_spaces(self, T_0, Fq_0_1, Fq_0_2, Eq_0_1, Eq_0_2, **kwargs):
        """
        Project initial data on the FE spaces 
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_FE_approximation_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        # Expressions
        Tp_0 = Expression(T_0, degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
        Fq_0 = Expression((Fq_0_1, Fq_0_2), degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs) 
        Eq_0 = Expression((Eq_0_1, Eq_0_2), degree=2, x0=Constant(self.x0), xL=Constant(self.xL), y0=Constant(self.y0), yL=Constant(self.yL), **kwargs)
            
        # Vectors 
        self.Tp0  = interpolate(Tp_0, self.Vp).vector()[:]
        self.Fq0  = interpolate(Fq_0, self.Vq).vector()[:]
        self.Eq0  = interpolate(Eq_0, self.Vq).vector()[:]
        
        self.Y0   = np.concatenate((self.Tp0, self.Fq0, self.Eq0))
        self.YD0  = np.zeros(shape=self.Y0.shape[:])
        
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
        
        #self.Convert_into_sparse_format()
        
       
        if string_mode == 'ODE:Assimulo': 
            Ham, t_span = self.integration_assimulo(**kwargs)
            done           = 1  
            
        if string_mode == 'ODE:Scipy': 
            Ham, t_span  = self.integration_scipy(**kwargs)
            done           = 1            
        
        if string_mode == 'DAE:Assimulo': 
            Ham, t_span = self.DAE_integration_assimulo(**kwargs)
            done           = 1 
            
        if string_mode == 'DAE:Assimulo:Lagrangian': 
            Ham, t_span = self.DAE_integration_assimulo_lagrangian(**kwargs)
            done           = 1
        
        assert done == 1, "Unknown time discretization method in Time_integration"
        
        # TO CHECK IF NEEDED
        # index_list   = np.where(abs(self.tspan-self.tclose)==abs(self.tspan-self.tclose).min())[0]
        # self.Nclose  = index_list[0]
        
        return Ham, t_span
    
   
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
        
        my_solver = factorized(csc_matrix(self.Mp_rho_Cv))
        C         = self.A
        
        if self.sparse == 1:
            my_jac_o  = csr_matrix(my_solver(C.toarray()))
        else:
            my_jac_o  = my_solver(C)
                
        # Definition of the rhs function required in assimulo
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of assimulo
            """   
            
            z = self.my_mult(self.A, y) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(t,self.tclose)) 
            
            return my_solver(z)
            
            
            #z                                    = np.zeros(shape=y.shape[:])        
            #z[0:self.Np]                         = self.my_mult(self.Mp, yd[0:self.Np])              - self.my_mult(D, y[self.Np+self.Nq:self.Np+2*self.Nq]) 
            #z[self.Np:self.Np+self.Nq]           = self.my_mult(self.Mq, y[self.Np:self.Np+self.Nq]) + self.my_mult(D.T, y[0:self.Np]) - self.my_mult(self.Bp, self.U* self.boundary_cntrl_time(t,self.tclose))
            #z[self.Np+self.Nq:self.Np+2*self.Nq] = self.my_mult(self.Mq, y[self.Np+self.Nq:self.Np+2*self.Nq]) - self.my_mult(self.L,y[self.Np:self.Np+self.Nq])
            #return z
 
        def jacobian(t,y):
            """
            Jacobian related to the ODE formulation
            """
            my_jac = my_jac_o 
            
            return my_jac
        
        def jacv(t,y,fy,v):
            """
            Jacobian matrix-vector product related to the ODE formulation
            """
            return None
           
        print('ODE Integration using assimulo built-in functions:')

#
# https://jmodelica.org/assimulo/_modules/assimulo/examples/cvode_with_preconditioning.html#run_example
#
        
        model                     = Explicit_Problem(rhs,self.Tp0,self.tinit)
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
            Ham_ode[k] = 1/2 * self.my_mult(A_ode[:,k].T, \
                               self.my_mult(self.Mp_rho_Cv,  A_ode[:,k]))
      
        self.Ham_ode = Ham_ode
    
        return Ham_ode, np.array(time_span)
    
    
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

        self.UU = np.zeros(self.Nb)
        for i in range(self.Nb):
            self.UU[i] = self.boundary_cntrl_space(self.coord_b[i])

        # Definition of the sparse solver for the ODE rhs function to
        # be defined next
        my_solver = factorized(csc_matrix(self.Mp_rho_Cv))
        C         = self.A
        
        if self.sparse == 1:
            my_jac_o  = csr_matrix(my_solver(C.toarray()))
        else:
            my_jac_o  = my_solver(C)
        
        # Definition of the rhs function required in assimulo
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of IVP
            """   
            z = self.my_mult(self.A, y) + self.my_mult(self.Bext,self.UU* self.boundary_cntrl_time(t,self.tclose)) 
            
            return my_solver(z)
        
        def jacobian(t,y):
            """
            Jacobian matrix related to the ODE
            """
            my_jac = my_jac_o 
            
            return my_jac
            
        print('ODE Integration using scipy.integrate built-in functions:')

        ivp_ode    = integrate.solve_ivp(rhs, (self.tinit,self.tfinal), self.Tp0, **kwargs, jac= jacobian, atol=1.e-3)   
        
        Y_ode      = ivp_ode.y
        self.tspan = np.array(ivp_ode.t)
        self.Nt    = len(self.tspan)
        
        print("Scipy: Number of evaluations of the right-hand side ",ivp_ode.nfev)
        print("Scipy: Number of evaluations of the Jacobian ",ivp_ode.njev)
        print("Scipy: Number of LU decompositions ",ivp_ode.nlu)
                        
        # Hamiltonian 
        Ham_ode    = np.zeros(self.Nt)
        
        for k in range(self.Nt):
            Ham_ode[k] = 1/2 * Y_ode[:,k] @ self.Mp_rho_Cv @ Y_ode[:,k]
        
    
        return Ham_ode, np.array(ivp_ode.t)
   
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
        #my_solver = factorized(csc_matrix(self.M))
        #rhs       = self.my_mult(self.J, self.my_mult(self.Q,self.A0)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(0.,self.tclose))
        #self.AD0  = my_solver(rhs) 
        
        # Definition of the rhs function required in assimulo
        def res(t,y,yd):
            """
            Definition of the residual function required in the DAE part of assimulo
            """   
            z                                    = np.zeros(self.Np+2*self.Nq)        
            z[0:self.Np]                         = self.my_mult(self.Mp_rho_Cv, yd[0:self.Np])              - self.my_mult(self.D, y[self.Np+self.Nq:self.Np+2*self.Nq]) 
            z[self.Np:self.Np+self.Nq]           = self.my_mult(self.Mq, y[self.Np:self.Np+self.Nq]) + self.my_mult(self.D.T, y[0:self.Np]) - self.my_mult(self.Bp, self.U* self.boundary_cntrl_time(t,self.tclose))
            z[self.Np+self.Nq:self.Np+2*self.Nq] = self.my_mult(self.Mq, y[self.Np+self.Nq:self.Np+2*self.Nq]) - self.my_mult(self.L,y[self.Np:self.Np+self.Nq])
            
            return z
  
        # Definition of the jacobian function required in assimulo
        def jac(c,t,y,yd):
            """
            Definition of the Jacobian matrix required in the DAE part of assimulo
            """  
            #Matrix = csr_matrix(self.my_mult(self.J,self.Q))
            
            #if close and t > self.tclose:
            #        Matrix = csr_matrix(self.my_mult(self.J - self.R, self.Q))
            
            #return c*csr_matrix(self.M) - Matrix
            
            return None
        
        # Definition of the jacobian matrix vector function required in assimulo
        def jacv(t,y,yd,res,v,c):
            """
            Jacobian matrix-vector product required in the DAE part of assimulo
            """  
            #w = self.my_mult(self.Q, v)
            #z = self.my_mult(self.J, w)
            
            #if close and t > self.tclose:
            #    z -= self.my_mult(self.R, w)
                
            #return c*self.my_mult(self.M,v) - z
            
            return None
        
        print('DAE Integration using assimulo built-in functions:')

        
        model                     = Implicit_Problem(res,self.Y0,self.YD0,self.tinit)
        #model.jacv                = jacv
        #sim                       = Radau5DAE(model,**kwargs)
        #
        # IDA method from Assimulo
        #
        sim                       = IDA(model,**kwargs)
        sim.algvar                = list(np.concatenate((np.ones(self.Np), np.zeros(2*self.Nq) )) )
        sim.atol                  = 1.e-6
        sim.rtol                  = 1.e-6
        sim.report_continuously   = True
        ncp                       = self.Nt
        #sim.usejac                = True
        sim.suppress_alg          = True
        sim.inith                 = self.dt
        sim.maxord                = 5
        #sim.linear_solver         = 'SPGMR'
        sim.make_consistent('IDA_YA_YDP_INIT')
        
        time_span, DAE_y, DAE_yd  = sim.simulate(self.tfinal,0, self.tspan)
        
        #print(sim.get_options())
        print(sim.print_statistics())
        
        A_dae = DAE_y[:,0:self.Np].transpose()
        
        # Hamiltonian
        self.Nt    = A_dae.shape[1]
        self.tspan = np.array(time_span)
        
        Ham_dae = np.zeros(self.Nt)
        
        for k in range(self.Nt):
            Ham_dae[k] = 1/2 * self.my_mult(A_dae[:,k].T, \
                               self.my_mult(self.Mp_rho_Cv, A_dae[:,k]))
      
        self.Ham_dae = Ham_dae
    
        return Ham_dae, np.array(time_span)  
    
    ## Method.
    #  @param self The object pointer.    
    def DAE_integration_assimulo_lagrangian(self, **kwargs):
        """
        Perform time integration for DAEs with the assimulo package
        Lagrangian variant
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        if self.tclose > 0:
            close    = True
        else:
            close    = False
            
        # Control vector
        self.U = interpolate(self.boundary_cntrl_space, self.Vl).vector()[self.bndr_i_l]
        if self.discontinous_boundary_values == 1:
            self.U[self.Corner_indices] = self.U[self.Corner_indices]/2
            
        self.UU = np.zeros(self.Nb)
        for i in range(self.Nb):
            self.UU[i] = self.boundary_cntrl_space(self.coord_b[i])
        

        # Definition of the sparse solver for the DAE res function to
        # be defined next M should be invertible !! 
        #my_solver = factorized(csc_matrix(self.M))
        #rhs       = self.my_mult(self.J, self.my_mult(self.Q,self.A0)) + self.my_mult(self.Bext,self.U* self.boundary_cntrl_time(0.,self.tclose))
        #self.AD0  = my_solver(rhs) 
        
        # Definition of the rhs function required in assimulo
        def res(t,y,yd):
            """
            Definition of the residual function required in the DAE part of assimulo
            """   
            z                                    = np.zeros(self.Np+self.Nl)        
            z[0:self.Np]                         = self.my_mult(self.M_class, yd[0:self.Np])  + self.my_mult(self.D_class, y[0:self.Np]) - self.my_mult(self.C_class, y[self.Np:])
            z[self.Np:self.Np+self.Nl]           = self.my_mult(self.C_class.T, y[0:self.Np]) - self.L_class * self.boundary_cntrl_time(t,self.tclose)
            
            return z
  
        # Definition of the jacobian function required in assimulo
        def jac(c,t,y,yd):
            """
            Definition of the Jacobian matrix required in the DAE part of assimulo
            """  
            #Matrix = csr_matrix(self.my_mult(self.J,self.Q))
            
            #if close and t > self.tclose:
            #        Matrix = csr_matrix(self.my_mult(self.J - self.R, self.Q))
            
            #return c*csr_matrix(self.M) - Matrix
            
            return None
        
        # Definition of the jacobian matrix vector function required in assimulo
        def jacv(t,y,yd,res,v,c):
            """
            Jacobian matrix-vector product required in the DAE part of assimulo
            """  
            #w = self.my_mult(self.Q, v)
            #z = self.my_mult(self.J, w)
            
            #if close and t > self.tclose:
            #    z -= self.my_mult(self.R, w)
                
            #return c*self.my_mult(self.M,v) - z
            
            return None
        
        print('DAE Integration using assimulo built-in functions:')

        #def handle_result(solver, t ,y, yd):
        #    global order
        #    order.append(solver.get_last_order())
        # 
        #     solver.t_sol.extend([t])
        #    solver.y_sol.extend([y])
        #    solver.yd_sol.extend([yd]) 
            
        # The initial conditons
        y0  =  np.concatenate(( self.Tp0, np.zeros(self.Nl) )) 
        yd0 =  np.zeros(self.Np + self.Nl)     
        
        model                     = Implicit_Problem(res,y0,yd0,self.tinit)
        #model.handle_result       = handle_result
        #model.jacv                = jacv
        #sim                       = Radau5DAE(model,**kwargs)
        #
        # IDA method from Assimulo
        #
        sim                       = IDA(model,**kwargs)
        sim.algvar                = list(np.concatenate((np.ones(self.Np), np.zeros(self.Nl) )) )
        sim.atol                  = 1.e-6
        sim.rtol                  = 1.e-6
        sim.report_continuously   = True
        ncp                       = self.Nt
        #sim.usejac                = True
        sim.suppress_alg          = True
        sim.inith                 = self.dt
        sim.maxord                = 5
        #sim.linear_solver         = 'SPGMR'
        sim.make_consistent('IDA_YA_YDP_INIT')
        
        #time_span, DAE_y, DAE_yd  = sim.simulate(self.tfinal,ncp, self.tspan)
        time_span, DAE_y, DAE_yd  = sim.simulate(self.tfinal, 0, self.tspan)
        
        #print(sim.get_options())
        print(sim.print_statistics())
        
        A_dae = DAE_y[:,0:self.Np].transpose()
                
        # Hamiltonian
        self.Nt    = A_dae.shape[1]
        self.tspan = np.array(time_span)
        
        Ham_dae = np.zeros(self.Nt)
        
        for k in range(self.Nt):
            Ham_dae[k] = 1/2 * self.my_mult(A_dae[:,k].T, \
                               self.my_mult(self.Mp_rho_Cv, A_dae[:,k]))
      
        self.Ham_dae = Ham_dae
    
        return Ham_dae, np.array(time_span)   
    
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
        plt.plot(self.coord_b[:,0], self.coord_b[:,1], 's', label='Dof of $\partial$ variables')
        plt.plot(self.coord_l[:,0], self.coord_l[:,1], 'k+', label='Dof of $Lagrange$ multipliers')
        plt.title('Mesh with associated DOFs, $Nq=$'+ str(self.Nq)+ ', $Np=$'+ str(self.Np)+ ', $N_\partial=$'+ str(self.Nb) + ', $N_{\lambda}=$'+ str(self.Nl) )
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
    
    
    #global order
    #order = []
    
    #
    # Create the Heat_2D object by calling the constructor of the class
    #
    
    x0, xL, y0, yL = 0., 2., 0., 1.
    
    H = Heat_2D()
     
    #
    # Define Fenics Expressions
    #
    
    Rho = 'x[0]* (2.-x[0])+ x[1]* (1.-x[1]) + 1'
    T11 = '5 + x[0]*x[1]'
    T12 = '(x[0]-x[1])*(x[0]-x[1])'
    T22 = '3.+x[1]/(x[0]+1)'
    Cv  = '3.'
    
    # Initial conditions
    
    amplitude, sX, sY, X0, Y0  = 3., (xL-x0)/12., (yL-y0)/12., (xL-x0)/2., (yL-y0)/2. 
    T0     = 'amplitude * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
    Fq_0_1 = '0'
    Fq_0_2 = '0'
    Eq_0_1 = '0'
    Eq_0_2 = '0'
    
    # Spatial part of the control 
    
    boundary_cntrl_space='x[0] + x[1]'
    
    # Time part of the control
    def boundary_cntrl_time(t,close_time):
        """
        Specify here the time part of the boundary control as a pure Python method. 
        """        
        if t <= close_time:
            return 2*t/(t+1.)
        else:
            return 0.
    
    #
    # Define constants related to the time discretization
    #
    
    tinit  = 0.
    tfinal = 2.
    tclose = 2.
    dt     = 1.e-2
    theta  = 0.5
    
    #
    # Problem Definition
    #
    
    H.Set_rectangular_domain(x0, xL, y0, yL)
    
    H.Set_physical_parameters(Rho, T11, T12, T22)
    
    H.Set_isochoric_heat_capacity(Cv)
    
    H.Set_boundary_cntrl_space(boundary_cntrl_space)
    
    H.Set_boundary_cntrl_time(boundary_cntrl_time)
    
    H.Set_initial_close_final_time(tinit,tclose,tfinal)
    
    assert H.Check_Problem_Definition() == 1, "Problem definition \
                to be checked"
    
    #
    # Finite Element Space and Time discretizations
    # 
    
    H.Generate_mesh(rfn=4)
    
    H.Set_FE_Approximation_Spaces(family_q='RT', family_p='P', family_b='P', family_l='P',\
                                  rt_order=1, p_order=2, b_order=2, l_order=2)
    
    H.Apply_PFEM_formulation()
    
    H.Project_initial_data_FE_spaces(T0, Fq_0_1, Fq_0_2, Eq_0_1, Eq_0_2,   
                                     amplitude=Constant(amplitude), 
                                     sX=Constant(sX), sY=Constant(sY),
                                     X0=Constant(X0), Y0=Constant(Y0))
    
    time_method = 'ODE:Assimulo'
    
    H.Set_time_setting(dt, theta, time_method)
    
    assert H.Check_Space_Time_Discretization() == 1, "Space_Time_Discretization \
                to be checked"
    
    #
    # Numerical Approximations
    #
    
    Ham, t_span = H.Time_integration(time_method)
        
    time_method   = 'ODE:Scipy'
    H.Set_time_setting(dt, theta, time_method)
    Ham_s, t_span_s = H.Time_integration(time_method,method='BDF')

    time_method   = 'DAE:Assimulo'
    H.Set_time_setting(dt, theta, time_method)
    Ham_dae, t_span_dae = H.Time_integration(time_method)    
   
    time_method   = 'DAE:Assimulo:Lagrangian'
    H.Set_time_setting(dt, theta, time_method)
    Ham_dae_l, t_span_dae_l = H.Time_integration(time_method) 
    
    #
    # Post-processing
    # 
    H.plot_mesh_with_DOFs()    
    #H.plot_Hamiltonian(t_span,Ham,marker='o')
    #H.plot_Hamiltonian(t_span_s,Ham_s,marker='s')
    plt.figure()
    plt.plot(t_span,  Ham,  '-o',label='ODE:Assimulo')
    plt.plot(t_span_s,Ham_s,'-s',label='ODE:Scipy')
    plt.plot(t_span_dae,Ham_dae,'-*',label='DAE:Assimulo')
    plt.plot(t_span_dae_l,Ham_dae_l,'-k',label='DAE:Assimulo:Lagrange')
    plt.grid(True)
    plt.legend()
    plt.show()
    #W.plot_3D(Deformation,tfinal/2,'deflection at t='+str(tfinal/2))
    #H.set_video_writer()
    #H.moving_trisurf(Ham_dae,1,'movie_dae',save=False)
     
    # Plot Hamiltonians errors
    plt.figure()
    #plt.semilogy(t_span, np.abs(Ham-Ham_dae)/(np.abs(Ham) + np.abs(Ham_dae)) * 100, '-', label='ODE-PFEM vs DAE-PFEM')
    plt.semilogy(t_span_dae, np.abs(Ham_dae_l-Ham_dae)/(np.abs(Ham_dae_l) + np.abs(Ham_dae)) * 100, '--', label='DAE-PFEM vs ODE-FEM(Lag)', linewidth=2.5)
    #plt.semilogy(t_span, np.abs(Ham_dae_l-Ham)/(np.abs(Ham) + np.abs(Ham_dae_l)) * 100, '-',label='ODE-FEM(Lag) vs ODE-PFEM')

    print(t_span.shape[:])
    print(t_span_s.shape[:])
    print(t_span_dae.shape[:])
    print(t_span_dae_l.shape[:])

    plt.grid(True)
    plt.xlabel('Time $(s)$')
    plt.ylabel('relative error (%)')
    plt.legend()
    plt.title('HAMILTONIANS: relative errors')
    plt.show()
    #
    # End
    #
        
        
        
        
        
        
        
      
        
