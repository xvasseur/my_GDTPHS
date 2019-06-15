# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:33:34 2017

@author: vasseur
"""
## @package 
#  Documentation for this module.
#
#  

## Documentation for a class.
#
#  Class to handle classical model reduction techniques for port-Hamiltonian systems.
#
#  A port-Hamiltonian system is defined as
#  \f$\Sigma = (\Xi, H, \cal R, \cal C, \cal I, \cal D) \f$
#
#  \f$\Xi \f$: state space manifold
#  \f$\cal H \f$: Hamiltonian:  \f$\Xi \rightarrow \mathcal{R} \f$ corresponding to the energy storage port \f$\cal S \f$
#  \f$\cal R \f$: resistive port 
#  \f$\cal C \f$: control port
#  \f$\cal I \f$: interconnection port
#  \f$\cal D \f$: total Dirac structure 
#
#  See 
# 
#  van der Schaft A.J. (2013) Port-Hamiltonian Differential-Algebraic Systems. 
#  In: Ilchmann A., Reis T. (eds) Surveys in Differential-Algebraic Equations I. Differential-Algebraic Equations 
#  Forum. Springer, Berlin, Heidelberg
#
#  and 
#  
#  A.J. van der Schaft, "Port-Hamiltonian systems: an introductory survey", 
#  Proceedings of the International Congress of Mathematicians, Volume III, Invited Lectures, 
#  eds. Marta Sanz-Sole, Javier Soria, Juan Luis Verona, Joan Verdura, Madrid, Spain, pp. 1339-1365, 2006.
#
#  The global subsytem may be written as (with the additional possibility of a constraint matrix G_sys)
#
#                      M_sys xdot = (J-R)     GradH + B u + G_sys lambda  + s_sys
#                            y    = B^T       GradH + D u                 + s_o
#                            0    = G_sys^T   GradH
# 
#
# The role of M_sys has to be detailed later, up to now M_sys is equal to Identity. 
#
# The algorithms related to model reduction are described in 
#
# S. Chaturantabut, C. Beattie and S. Gugercin (2016) Structure-preserving model reduction for nonlinear 
# port-Hamiltonian systems, SIAM J. Sci. Comput. 38-5, pp. B837-B865.
#
# S. Gugercin, R. Polyuga, C. Beattie, A. van der Schaft (2012) Structure-preserving tangential interpolation for 
# model reduction of port-Hamiltonian systems, Automatica 48 1963-1974.
#
# In a first attempt we consider the linear case for the port-Hamiltonian systems.
# with a similar structure as in Chaturantabut et al. 


import numpy as np
from scipy import linalg 
from fbpca import pca,diffsnorm
from util  import cholesky_qr 
import math
import matplotlib.pyplot as plt

class Model_Reduction(object):
    """ Define the general methods to be used for the model reduction of a subsystem of Port-Hamiltonian system type """

    ## The constructor of the Model_Reduction class.
    #  @param self The object pointer.
    #  @param X    Collection of snapshots related to the trajectory x(t)
    #  @param F    Collection of snapshots related to the internal force \nabla_x H(x)

    def __init__(self,X,F):
        """
        Constructor for the Model_Reduction class.
        """
        ## @var X
        #  collection of snapshots related to the trajectory x(t) [Numpy array]
        #  @var F
        #  collection of snapshots related to the internal force \nabla_x H(x) [Numpy array]
        #  @var algorithm
        #  description related to the algorithm [string]

        # Set the default method 
        self.algorithm = 'SVD'
            
        # Check and set the problem dimensions    
        mx = X.shape[0]
        nx = X.shape[1]
        
        mf = F.shape[0]
        nf = F.shape[1]
        
        assert mx == mf, "The two snapshot matrices should have the same number of rows"       
        assert nx == nf, "The two snapshot matrices should have the same number of columns"       
        assert mx >= 0, "The number of rows of the trajectory snapshot matrix should be positive"       
        assert mf >= 0, "The number of rows of the internal force snapshot matrix should be positive"    
        assert nx >= 0, "The number of columns of the trajectory snapshot matrix should be positive" 
        assert nf >= 0, "The number of columns of the trajectory snapshot matrix should be positive" 

        self.m   = mx
        self.n   = nx

        # r is the variable storing the rank of the approximation        
        
        self.r                            = 0
        self.approximate_rank_upper_bound = min(self.m,self.n)
        
        ## @var X
        #  Snapshot matrix for the displacement [Numpy array]
        #  @var F
        #  Snapshot matrix for the force  [Numpy array]

        self.X        = np.zeros(shape=(self.m,self.n))
        self.F        = np.zeros(shape=(self.m,self.n))
        self.Weight   = np.zeros(shape=(self.m,self.m))
        
        # Shall we use a copy here ?   
        self.X  = X
        self.F  = F
        
        self.xs = np.zeros(min(self.m,self.n))
        self.fs = np.zeros(min(self.m,self.n))   
        
        
        ## @var istatus variables
        #  Flag related to the model reduction algorithm [integer]
        
        self.istatus_SVD_X   = 0
        self.istatus_SVD_F   = 0
        
        self.factorize_and_truncate = 0
        self.compute_projection_qr = 0
        self.compute_projection_svd = 0
        self.change_basis = 0
        self.pod  = 0
        self.algorithm_selected = 0
        self.weighted = 0
        
        ## @var actual ratio of the singular values
        #        
        self.threshold_F     = 0
        self.threshold_X     = 0
    
    ## Method.
    #  @param self The object pointer.
        
    def Specify_Weight_Matrix(self,M):
        """
        Set the weight matrix to be used for the model reduction
        """
        self.Weight = M
        
        self.weighted = 1
        
        return self.weighted

    
    ## Method.
    #  @param self The object pointer.
        
    def Specify_POD_Method(self,method):
        """
        Set the method to be used for the POD model reduction (method [string]). 
        By default we use the classical SVD method ('SVD' as keyword)
        """
        self.algorithm = method
        
        self.algorithm_selected = 1
        
        return self.algorithm_selected
  
    ## Method.
    #  @param self The object pointer.

    def Randomized_SVD(self,rank):
        """
        Perform the randomized SVD of the X and F snapshot matrices given a tentative approximate rank. 
        This is based on the Facebook implementation of the algorithm available in the fbpca module.
        (Public implementation)
        Keyword = "Randomized_SVD")
        """
        assert rank > 0, "The approximate rank should be strictly positive"
        
        # Impose a value if rank is too large
        approximate_rank = min(rank,self.approximate_rank_upper_bound)
        
        # Randomized SVD of X 
        XU, Xs, XVh          = pca(self.X, approximate_rank, True)
        self.istatus_SVD_X   = 1
        self.threshold_X     = Xs[-1]/Xs[0]
        self.xs[:approximate_rank] = Xs[:]
        print(diffsnorm(self.X, XU, Xs, XVh))
        
        # Randomized SVD of F
        FU, Fs, FVh          = pca(self.F, approximate_rank, True)
        self.istatus_SVD_F   = 1
        self.threshold_F     = Fs[-1]/Fs[0]
        self.fs[:approximate_rank] = Fs[:]
        print(diffsnorm(self.F, FU, Fs, FVh))
        
        # Deduce the approximation matrices
        self.r              = approximate_rank
        self.X_approx_basis = XU[:,0:self.r]
        self.F_approx_basis = FU[:,0:self.r]
        
        # The "truncation" here is due to the choice of the tentative rank
        self.factorize_and_truncate   = 1
        
        return self.factorize_and_truncate

      
    ## Method.
    #  @param self The object pointer.

    def Factorize_Truncate_SVD(self,relative_threshold):
        """
        Perform the economic SVD of the X and F snapshot matrices and deduce an approximation according to a relative threshold value
        """
        assert relative_threshold >= 0, "The relative threshold value should be positive"
        assert relative_threshold <= 1, "The relative threshold value should be less than one"
        
        # SVD of X 
        XU, Xs, XVh          = linalg.svd(self.X, full_matrices=False)
        self.istatus_SVD_X   = 1
        rx                   = sum(Xs >= relative_threshold*Xs[0])
        self.threshold_X     = Xs[rx-1]/Xs[0]
        #print(Xs[:]/Xs[0],rx, self.threshold_X)
        self.xs              = Xs
        
        # SVD of F
        FU, Fs, FVh          = linalg.svd(self.F, full_matrices=False)
        self.istatus_SVD_F   = 1
        rf                   = sum(Fs >= relative_threshold*Fs[0])
        self.threshold_F     = Fs[rf-1]/Fs[0]
        #print(Fs[:]/Fs[0], rf, self.threshold_F)
        self.fs              = Fs
        
        # Truncate according to the threshold value
        r                   = max(rx,rf)
        self.r              = r
        self.X_approx_basis = XU[:,0:self.r]
        self.F_approx_basis = FU[:,0:self.r]
        
        self.factorize_and_truncate   = 1
        
        return self.factorize_and_truncate
        
        
    def Factorize_Truncate_R_SVD(self,relative_threshold):
        """
        Perform the economic SVD due to Chan (R-SVD) of the X and F snapshot matrices and deduce an approximation according to a relative threshold value.
        This decomposition has to used if the matrices are sufficiently tall (m >= 1.5 n).
        See Chan. An improved algorithm for computing the Singular Value Decomposition. ACM Trans. Math. Softw., 8 :1, 84-88, 1982.
        """
        assert relative_threshold >= 0, "The relative threshold value should be positive"
        assert relative_threshold <= 1, "The relative threshold value should be less than one"
        
        # QR factorization of X        
        XQ, XR               = linalg.qr(self.X)        
        
        # SVD of XR 
        XRU, XRs, XRVh       = linalg.svd(XR)
        self.istatus_SVD_X   = 1
        rx                   = sum(XRs >= relative_threshold*XRs[0])
        self.threshold_X     = XRs[rx-1]/XRs[0]
        self.xs              = XRs
        
        # QR factorization of F        
        FQ, FR               = linalg.qr(self.F)        
       
        # SVD of FR
       
        FRU, FRs, FRVh       = linalg.svd(FR)
        self.istatus_SVD_F   = 1
        rf                   = sum(FRs >= relative_threshold*FRs[0])
        self.threshold_F     = FRs[rf-1]/FRs[0]
        self.fs              = FRs
        
        # Truncate according to the threshold value
        
        r                    = max(rx,rf)
        self.r               = r
        self.X_approx_basis  = np.dot(XQ,XRU[:,0:self.r])
        self.F_approx_basis  = np.dot(FQ,FRU[:,0:self.r])
        
        self.factorize_and_truncate   = 1
        
        return self.factorize_and_truncate
      
    def Factorize_Truncate_Cholesky_QR(self,relative_threshold):
        """
        Perform the Cholesky QR 2 algorithm (two steps) of the X and F snapshot matrices and deduce an approximation according to a relative threshold value.
        This decomposition has to used if the matrices are sufficiently tall (m >= 1.5 n).
        See [REF]
        This corresponds to the serial version of the algorithm. Map/Reduce techniques can be employed to implement in parallel this algorithm. 
        """
        assert relative_threshold >= 0, "The relative threshold value should be positive"
        assert relative_threshold <= 1, "The relative threshold value should be less than one"
        
        # Cholesky QR factorization of X        
        XQ, XR               = cholesky_qr(self.X)        
        
        # SVD of XR 
        XRU, XRs, XRVh       = linalg.svd(XR)
        self.istatus_SVD_X   = 1
        rx                   = sum(XRs >= relative_threshold*XRs[0])
        self.threshold_X     = XRs[rx-1]/XRs[0]
        self.xs              = XRs
        
        # Cholesky QR factorization of F        
        FQ, FR               = cholesky_qr(self.F)        
       
        # SVD of FR
       
        FRU, FRs, FRVh       = linalg.svd(FR)
        self.istatus_SVD_F   = 1
        rf                   = sum(FRs >= relative_threshold*FRs[0])
        self.threshold_F     = FRs[rf-1]/FRs[0]
        self.fs              = FRs
        
        # Truncate according to the threshold value
        
        r                    = max(rx,rf)
        self.r               = r
        self.X_approx_basis  = np.dot(XQ,XRU[:,0:self.r])
        self.F_approx_basis  = np.dot(FQ,FRU[:,0:self.r])
        
        self.factorize_and_truncate   = 1
        
        return self.factorize_and_truncate
        
    ## Method.
    #  @param self The object pointer.
     
    def Factorize_QR_Projected_Matrix(self): 
        """
        Perform the full QR factorization of the matrix (F_approx_basis^T X_approx_basis).
        Note: this procedure is fine in the weighted case if the condition number of self.Weight
        is not too large. Otherwise another decomposition with improved backward stability 
        must be used (two iterations of Modified Gram-Schmidt).
        """
        assert self.compute_projection_qr == 0, "The QR factorization of F_approx_basis^T M X_approx_basis has been already performed"       
        
        if self.weighted == 1:
            self.PQ,self.PR = linalg.qr(np.dot(self.F_approx_basis.T,self.Weight@self.X_approx_basis))
        else:
            self.PQ,self.PR = linalg.qr(np.dot(self.F_approx_basis.T,self.X_approx_basis))
         
        self.compute_projection_qr = 1 
        
        # Debug checks
        
        #print("Factorize_QR_Projected_Matrix:",np.dot(self.PQ.T,self.PQ))
        #print(np.allclose(np.dot(self.F_approx_basis.T,self.X_approx_basis),np.dot(self.PQ,self.PR)))
 
        return self.compute_projection_qr       

    ## Method.
    #  @param self The object pointer.
     
    def Factorize_SVD_Projected_Matrix(self): 
        """
        Perform the full SVD factorization of the matrix (F_approx_basis^T X_approx_basis).

        """
        assert self.compute_projection_svd == 0, "The SVD factorization of F_approx_basis^T M X_approx_basis has been already performed"       
        
        if self.weighted == 1:
            self.PU,self.Ps, self.PVh = linalg.svd(np.dot(self.F_approx_basis.T,self.Weight@self.X_approx_basis))
        else:
            self.PU,self.Ps, self.PVh = linalg.svd(np.dot(self.F_approx_basis.T,self.X_approx_basis))
        
        self.PD = np.zeros(shape=(self.Ps.shape[0],self.Ps.shape[0]))
        for loop in range(self.Ps.shape[0]):
            self.PD[loop,loop] = 1./math.sqrt(self.Ps[loop])
            
        self.compute_projection_svd = 1 
        
        # Debug checks
        
        #print("Factorize_QR_Projected_Matrix:",np.dot(self.PQ.T,self.PQ))
        #print(np.allclose(np.dot(self.F_approx_basis.T,self.X_approx_basis),np.dot(self.PQ,self.PR)))
 
        return self.compute_projection_svd

    ## Method.
    #  @param self The object pointer.
     
    def Change_Basis(self): 
        """
        Deduce the (reduced- or low) rank approximation matrices F and X such that F^T X = I_r.
        """
        
        assert self.compute_projection_qr == 1 or self.compute_projection_svd == 1, "The computation of the projection matrix has to be performed "
        assert self.change_basis      == 0, "Change of basis has been already performed "
        
        
        if self.compute_projection_svd == 1:
            self.F_approx_basis_change = self.F_approx_basis @ self.PU @ self.PD
            self.X_approx_basis_change = self.X_approx_basis @ self.PVh.T @ self.PD
            
        if self.compute_projection_qr == 1:
            self.F_approx_basis_change = np.dot(self.F_approx_basis,self.PQ)        
            W = linalg.solve_triangular(self.PR,self.X_approx_basis.T,trans='T')
            self.X_approx_basis_change = W.T
        
        self.change_basis = 1
        
        # We add these internal variables to comply with the notations of the reference paper by S. Chaturantabut, C. Beattie and S. Gugercin (2016)
        
        self.V = self.X_approx_basis_change
        
        self.W = self.F_approx_basis_change
        
        self.compute_V = 1
        self.compute_W = 1
        
        return self.change_basis
        
    ## Method.
    #  @param self The object pointer.
     
    def POD_Reduction(self,relative_threshold=0.1,approximate_rank=5):
        """
        Apply the POD method for the model reduction of the Port-Hamiltonian system. This offers:
        "SVD" : the classical full SVD 
        "R_SVD": the economic SVD based on Chan's algorithm
        "Randomized_SVD": approximate SVD based on randomized linear algebra. 
        
        The "Randomized_SVD" method has to be used when the problem leads to a tall and skinny matrix, i.e., 
        m >> n. 
        """
        assert relative_threshold >= 0, "The relative threshold value should be positive"
        assert relative_threshold <= 1, "The relative threshold value should be less than one"
        assert approximate_rank > 0, "The approximate rank should be strictly positive"

        if self.algorithm == 'SVD':       
           assert self.Factorize_Truncate_SVD(relative_threshold) == 1, "Failure in the SVD factorization of the two snapshot matrices"
        elif self.algorithm == 'R_SVD':
           assert self.Factorize_Truncate_R_SVD(relative_threshold) == 1, "Failure in the R-SVD factorization of the two snapshot matrices"   
        elif self.algorithm == 'Randomized_SVD':
           assert self.Randomized_SVD(approximate_rank) == 1, "Failure in the Randomized SVD factorization of the two snapshot matrices" 
        elif self.algorithm == 'Cholesky_QR':
           assert self.Factorize_Truncate_Cholesky_QR(relative_threshold) == 1, "Failure in the Cholesky QR factorization of the two snapshot matrices"
           
        assert self.Factorize_SVD_Projected_Matrix() == 1, "Failure in the SVD factorization of the projected matrix"
        
        assert self.Change_Basis() == 1, "Failure in the change of basis for the two approximation matrices"
        
        self.pod = 1
        
        return self.pod       
        
    
    ## Method.
    #  @param self The object pointer.
       
    def Apply_Approximation_X(self,input_variable):
        """
        Apply the approximation of X to a given vector/matrix of reduced size
        """
        assert self.X_approx_basis_change.shape[1] == input_variable.shape[0], "Wrong dimension in the number of rows of the input"
        assert self.factorize_and_truncate == 1, "Factorization and truncation of the snapshots to be performed before"
        assert self.compute_projection == 1, "QR factorization of the projection matrix to be performed before"          
        assert self.change_basis == 1, "Change of basis has been already performed" 
        
        return np.dot(self.X_approx_basis_change,input_variable)
  
    ## Method.
    #  @param self The object pointer.
       
    def Apply_Approximation_F(self,input_variable):
        """
        Apply the approximation of F to a given vector/matrix of reduced size
        """
        assert self.F_approx_basis_change.shape[1] == input_variable.shape[0], "Wrong dimension in the number of rows of the input"
        assert self.factorize_and_truncate == 1, "Factorization and truncation of the snapshots to be performed before"  
        assert self.compute_projection == 1, "QR factorization of the projection matrix to be performed before"          
        assert self.change_basis == 1, "Change of basis has been already performed"         
          
        return np.dot(self.F_approx_basis_change,input_variable)
        
    ## Method.
    #  @param self The object pointer.
   
    def Apply_Approximation_X_Transpose(self,input_variable):
        """
        Apply the transpose of the approximation of X to a given vector/matrix 
        """
        assert self.X_approx_basis_change.shape[0] == input_variable.shape[0], "Wrong dimension in the number of rows of the input"
        assert self.factorize_and_truncate == 1, "Factorization and truncation of the snapshots to be performed before" 
        assert self.compute_projection == 1, "QR factorization of the projection matrix to be performed before"         
        assert self.change_basis == 1, "Change of basis has been already performed"                 
        
        return np.dot(self.X_approx_basis_change.T,input_variable)
        
        
    ## Method.
    #  @param self The object pointer.
     
    def Apply_Approximation_F_Transpose(self,input_variable):
        """
        Apply the transpose of the approximation of F to a given vector/matrix
        """
        assert self.F_approx_basis_change.shape[0] == input_variable.shape[0], "Wrong dimension in the number of rows of the input"
        assert self.factorize_and_truncate == 1, "Factorization and truncation of the snapshots to be performed before" 
        assert self.compute_projection == 1, "QR factorization of the projection matrix to be performed before"          
        assert self.change_basis == 1, "Change of basis has been already performed"                
        
        return np.dot(self.F_approx_basis_change.T,input_variable)
        
    ## Method.
    #  @param self The object pointer.
        
    def IRKA_Order_Interpolation_Sets(self,sigma,Direction):
        """
        Order the complex sets according to the imaginary part (ascending order). Organize both 
        the interpolation points (sigma) and the tangential directions.  
        We assume that the two sets are closed under conjugation. 
        The case of real-valued interpolation points is postponed [NOT CHECKED]. 
        """
        assert sigma.shape[0] == Direction.shape[1], " The interpolation set and the tangential directions are not consistent"
                
        self.set_ordered = 0
        
        self.r    = sigma.shape[0] 
        indices   = np.argsort(sigma.imag)
        sigma     = np.take(sigma, indices)
        Direction = np.take(Direction,indices,axis=1)
        
        self.set_ordered = 1
        
        return (sigma, Direction)
        
    ## Method.
    #  @param self The object pointer.
       
    def IRKA_Compute_V(self,sigma,Direction,B,J,Q,R):
        """
        Obtain the matrix of the tentative V subspace for the IRKA method. Note that V is constructed as a REAL subspace here.
        This construction assumes a special ordering in the interpolation point and tangential direction sets. 
        """
        assert self.set_ordered == 1, "Please apply the Order_Complex_Sets method before"
        
        self.IRKA_V =  np.zeros(shape=(self.m,self.r))
        
        # Compute the RHS matrix [TO BE IMPROVED, J-Q in dot ?]
        
        Rhs      = np.dot(B,Direction)
        Block    = np.dot(J,Q)-np.dot(R,Q)
        
        # Solution of the self.r/2 linear systems
        
        column = 0
        
        for loop in range(self.r/2):
            
            # Build the matrix
            
            Matrix = sigma[loop]*np.eye(self.m) - Block
            
            # Solve the corresponding linear system (the solution is complex valued)
            
            z = linalg.solve(Matrix,Rhs[:,loop])
            
            # Store both real and imaginary parts into the V set
            
            self.IRKA_V[:,column] = z.real
            column = column + 1
            self.IRKA_V[:,column] = z.imag
            column = column + 1
            
        # Checks
            
        assert column == self.r, "Something wrong in IRKA_Compute_V"   
        
        self.compute_IRKA_V = 1
            
        return self.compute_IRKA_V
  
    ## Method.
    #  @param self The object pointer.

            
    def IRKA_Compute_W(self,Q):
        """
        Obtain the real-valued matrix of the tentative W subspace for the IRKA method.  
        """
        assert self.compute_IRKA_V == 1, "Please apply the IRKA_Compute_V method before" 
        
        self.IRKA_W =  np.zeros(shape=(self.m,self.r))         
        
        Block = np.dot(Q,self.IRKA_V)
        M     = np.dot(self.IRKA_V.T,Block) 

        Work  = linalg.solve(M,Block.T) 
        
        self.IRKA_W = Work.T
         
        self.compute_IRKA_W = 1
            
        return self.compute_IRKA_W
  
    ## Method.
    #  @param self The object pointer.

      
    def IRKA_Determine_Interpolation_Sets(self,B,J,Q,R):
        """
        Determine the new interpolation sets given tentative V and W matrices. 
        """
        assert self.compute_IRKA_V == 1, "Please apply the IRKA_Compute_V method before"
        assert self.compute_IRKA_W == 1, "Please apply the IRKA_Compute_W method before"
        
        # Compute the reduced matrices

        Br = np.dot(self.IRKA_W.T,B)
        Jr = np.dot(self.IRKA_W.T,np.dot(J,self.IRKA_W))
        Qr = np.dot(self.IRKA_V.T,np.dot(Q,self.IRKA_V))
        Rr = np.dot(self.IRKA_W.T,np.dot(R,self.IRKA_W))

        # Solve the eigenvalue problem 

        Ar = np.dot(Jr,Qr)-np.dot(Rr,Qr)
        
        eigenvalues, Left_Eig, Right_Eig = linalg.eig(Ar, left=True, right=True)

        # Set and order the complex-valued sets 
        
        sigma     = - eigenvalues[:]
        
        # The next relation has to be CHECKED (mismatch between eigenvalues and eigenvectors ?)
        
        Direction = np.dot(Br.T,Left_Eig)
        
        # Update the interpolation point and the tangential direction sets      
        
        sigma, Direction = self.IRKA_Order_Interpolation_Sets(sigma,Direction)
        
        return (sigma, Direction)
 
    ## Method.
    #  @param self The object pointer.

       
    def IRKA_Reduction_Iteration(self,sigma,Direction,B,J,Q,R):
        """
        Driver for the IRKA algorithm for the model reduction
        """
        
        assert self.IRKA_Order_Interpolation_Sets(sigma, Direction) == 1, "Failure in IRKA_Order_Interpolation_Sets"
        
        assert self.IRKA_Compute_V(sigma,Direction,B,J,Q,R) == 1, "Failure in IRKA_Compute_V"
        
        assert self.IRKA_Compute_W(Q) == 1, "Failure in IRKA_Compute_W"
        
        (sigma, Direction) = self.IRKA_Determine_Interpolation_Sets(B,J,Q,R)
       
        return (sigma, Direction)
  
    ## Method.
    #  @param self The object pointer.

      
    def IRKA_Evaluate_Transfer_Function(self,interpolation_point,B,J,Q,R):
        """
        Evaluate the transfer function G(s) at a single interpolation point in the complex plane.
        """
        
        assert self.compute_IRKA_V == 1, "Please apply the IRKA_Compute_V method before"
        assert self.compute_IRKA_W == 1, "Please apply the IRKA_Compute_W method before"
        
        G =  np.zeros(shape=(self.m,B.shape[1]))
        
        # Compute the RHS matrix [TO BE IMPROVED, J-Q in dot ?]
        
        Block  = np.dot(J,Q)-np.dot(R,Q)
        
        # Build the matrix
            
        Matrix = interpolation_point*np.eye(self.m) - Block
            
        # Solve the corresponding linear system (the solution is complex valued)
            
        Z  = linalg.solve(Matrix,B)
        
        # Store the result in  G
        
        G = np.dot(B.T,np.dot(Q,Z))
    
        return G
        
    ## Method.
    #  @param self The object pointer.

        
    def IRKA_Evaluate_Reduced_Transfer_Function(self,interpolation_point,B,J,Q,R):
        """
        Evaluate the reduced transfer function G(s) at a single interpolation point in the complex plane.
        """
        
        assert self.compute_IRKA_V == 1, "Please apply the IRKA_Compute_V method before"
        assert self.compute_IRKA_W == 1, "Please apply the IRKA_Compute_W method before"
            
        Gr =  np.zeros(shape=(self.m,B.shape[1]))
        
        # Compute the reduced matrices

        Br = np.dot(self.IRKA_W.T,B)
        Jr = np.dot(self.IRKA_W.T,np.dot(J,self.IRKA_W))
        Qr = np.dot(self.IRKA_V.T,np.dot(Q,self.IRKA_V))
        Rr = np.dot(self.IRKA_W.T,np.dot(R,self.IRKA_W))
        
        # Compute the RHS matrix [TO BE IMPROVED, J-Q in dot ?]
        
        Block  = np.dot(Jr,Qr)-np.dot(Rr,Qr)
        
        # Build the matrix
            
        Matrix = interpolation_point*np.eye(self.m) - Block
            
        # Solve the corresponding linear system (the solution is complex valued)
            
        Z  = linalg.solve(Matrix,Br)
        
        # Store the result in  G
        
        Gr = np.dot(Br.T,np.dot(Qr,Z))
    
        return Gr

    ## Method.
    #  @param self The object pointer.

    def Determine_Reduced_Problem(self,B,J,Q,R,M):
        """
        Determine the matrices of the reduced problem given V and W matrices obtained by a given model reduction technique.
        Here we have assumed that a linearized port-Hamiltonian is available with a quadratic Hamiltonian function. 
        [TO BE DISCUSSED for the general setting]
        """
        
        assert self.compute_V == 1, "Approximation subspaces have not been determined first"
        assert self.compute_W == 1, "Approximation subspaces have not been determined first"
        
        assert B.shape[0] == self.W.shape[0], "The number of rows of B does not match with the number of rows of W"        
        assert J.shape[1] == self.W.shape[0], "The number of columns of J does not match with the number of rows of W" 
        assert J.shape[0] == self.W.shape[0], "The number of rows of J does not match with the number of columns of W" 
        assert Q.shape[1] == self.V.shape[0], "The number of columns of Q does not match with the number of rows of V" 
        assert Q.shape[0] == self.V.shape[0], "The number of rows of Q does not match with the number of columns of V" 
        assert R.shape[1] == self.W.shape[0], "The number of columns of R does not match with the number of rows of W" 
        assert R.shape[0] == self.W.shape[0], "The number of rows of R does not match with the number of columns of W"  
        assert M.shape[1] == self.V.shape[0], "The number of columns of M does not match with the number of rows of V" 
        assert M.shape[0] == self.W.shape[0], "The number of rows of M does not match with the number of rows of W"     
           
        # Compute the matrices of the reduced problem

        #Br = np.dot(self.W.T,B)
        #Jr = np.dot(self.W.T,np.dot(J,self.W))
        #if self.weighted == 1:
        #    Qr = np.dot((self.Weight@self.V).T,np.dot(Q,self.V))
        #else:
        #    Qr = np.dot(self.V.T,np.dot(Q,self.V))
        #Rr = np.dot(self.W.T,np.dot(R,self.W))
        #Mr = np.dot(self.W.T,np.dot(M,self.V))
  
        Br = self.W.T@B
        Jr = self.W.T@(J@self.W)
        if self.weighted == 1:
            Qr = (self.Weight@self.V).T@(Q@self.V)
        else:
            Qr = self.V.T@(Q@self.V)
        Rr = self.W.T@(R@self.W)
        Mr = self.W.T@(M@self.V)
      
        return (Br, Jr, Qr, Rr, Mr)
 
    ## Method.
    #  @param self The object pointer.
        
        
if __name__ == '__main__':
       
    # Set the parameters for the test matrices
    
    m, n      = 9, 8
    threshold = 0.00001
    approximate_rank = 20
    
    # Define the two snapshot matrices
    
    #X = np.random.randn(m, n)
    #F = np.random.randn(m, n)
    
    X = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Kirchhoff/MatricesKirchh4Red/e_snap.npy')
    F = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Kirchhoff/MatricesKirchh4Red/e_snap.npy')   
    M = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Kirchhoff/MatricesKirchh4Red/M.npy')
  
    #X = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Gaussian/First/Energy.npy')
    #F = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Gaussian/First/Coenergy.npy')    
    #M = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Gaussian/First/M.npy')

    plt.spy(M)
    plt.show()

    MU, Ms, MVh = linalg.svd(M)
    print(Ms)
    print(Ms[0]/Ms[-1])
    np.savetxt('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Kirchhoff/MatricesKirchh4Red/singular_value_mass_matrix.txt',Ms)
 
    m, n             = X.shape[:]
    
    # Define the model reduction object
    
    MR = Model_Reduction(X,F)
    MR.Specify_Weight_Matrix(M)
    
    # Specify the method to be used  
    
    MR.Specify_POD_Method('SVD')
      
    # Call the POD method on the object
        
    assert MR.POD_Reduction(threshold,approximate_rank) == 1
        
    print(MR.threshold_F, MR.threshold_X, MR.r, MR.weighted)      
        
    Q = np.eye(m)
    B = np.eye(m)
    J = np.eye(m)
    R = np.eye(m)
    J = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Kirchhoff/MatricesKirchh4Red/J.npy')
    
    #J = np.load('/Users/vasseur/Anr/Python/Demo/Python_3.x/Store/Gaussian/First/J.npy')
    
    (Br, Jr, Qr, Rr, Mr) = MR.Determine_Reduced_Problem(B,J,Q,R,M)
    
    print(linalg.norm(MR.W.T @ M @ MR.V,1))
    
    print(MR.Ps)
    
    print(MR.Ps[0]/MR.Ps[-1])
        
    #v, w  = linalg.eig(MR.PU.T @ Jr @ MR.PVh.T, np.diag(MR.Ps))
    v, w  = linalg.eig(Jr, Mr)

    print(v)
    
    fig, (ax_eig) = plt.subplots(1,1)
    ax_eig.plot(np.real(v), np.imag(v), 'o')
    ax_eig.set(title='Spectrum of JQ with respect to the mass matrix', ylabel='Imaginary part',xlabel='Real part')
    ax_eig.grid(True)
    ax_eig.legend()   
    plt.show()
    
    # Useful checks
    
    #print(np.dot(MR.F_approx_basis_change.T,MR.X_approx_basis_change))
    
    #x = np.random.randn(MR.X_approx_basis_change.shape[1],1)
    #print(x)    
    #print(MR.Apply_Approximation_F_Transpose(MR.Apply_Approximation_X(x)))
    
    #y = np.random.randn(MR.F_approx_basis_change.shape[1],1)
    #print(y)    
    #print(MR.Apply_Approximation_X_Transpose(MR.Apply_Approximation_F(y)))
   
    #print(linalg.norm(np.dot(MR.W.T,MR.W)-np.eye(MR.r)))
    #print(linalg.norm(np.dot(MR.W.T,M@MR.V)-np.eye(MR.r)))
    #print(linalg.norm(np.dot(MR.W.T,MR.W)-np.eye(MR.r)))
    #print(linalg.norm(Qr-Qr.T))
    #print(linalg.norm(M.T@Q-Q.T@M))
    #print(linalg.norm(MR.V.T@(M.T@Q-Q.T@M)@MR.V))
    #print(linalg.norm(Qr-(M@MR.V).T@(Q@MR.V)))
    #print(Qr-Qr.T)