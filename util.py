# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:07:45 2017

@author: vasseur
"""

import numpy as np
from scipy import linalg, sparse 
from scipy.sparse import issparse, block_diag, bmat, isspmatrix_csr
import logging
import unittest

## Method.
#  @param A Matrix given as a numpy array
#  @param atol [float]
#  @param rtol [float]

def nullspace(A, atol=1.e-13, rtol=0.):
    """
    Compute an approximate basis Z (with orthonormal columns) for the nullspace of A.
    
    Notes
    -----
    The algorithm is based on the complete SVD. Another implementation is required to handle the case of 
    a large-scale matrix. 
    The current implementation is available in both real and complex arithmetics.
    
    Type-FREE method.
    
    Parameters
    ----------
    A : array_like, shape (m, n)
        matrix being analyzed
        
    atol : float
           absolute tolerance (i.e machine precision)
        
    rtol:  float   
           relative tolerance (with respect to the leading singular value)
        
    Returns
    -------
    Z : ndarray, shape (n, r)
        matrix the columns of which form an approximate nullspace basis of A i.e.
        \| True_Nullspace_of_A - Z \|_2 <= max(atol, rtol * S[0])
        with S[0] the leading singular value of A.
        
    Examples
    --------
    >>> import numpy as np
    >>> from util import nullspace,blockdiag
    >>> (m,k) = (10,4)
    >>> A = np.random.rand(m,m)
    >>> B = blockdiag(A,np.zeros(shape=(k,k)))
    >>> Z = nullspace(B)
    >>> print(Z), print(np.dot(B,Z))    
    """
  
    # Check the shape of A (required due to the SVD in numpy)
    A        = np.atleast_2d(A)
    
    # Perform the SVD of A 
    U, S, VH = np.linalg.svd(A.todense() if issparse(A) else A)
    
    # Set the tolerance to be used
    tol      = max(atol, rtol * S[0])
    
    # Truncate and deduce the nullspace 
    nnz      = (S >= tol).sum()
    Z        = VH[nnz:].conj().T
    
    return Z
 
## Method.
#  @param A Matrix given as a numpy array
#  @param B Matrix given as a numpy array
 
def blockdiag(A,B):
    """
    Define a block diagonal matrix C with A and B as diagonal blocks such as 
    C = [A   0]
        [0   B],
    where A and B are supposed to be rectangular blocks.  
    
    Notes
    -----
    Type-FREE method if all blocks have same type.

    Parameters
    ----------
    A, B: array_like
        
    Returns
    -------
    C : ndarray, of appropriate shape 
    C = [A   0]
        [0   B]
    C is returned as a ndarray (sparse or dense depending on the types of A and B).
    
    Examples
    --------
    >>> import numpy as np
    >>> from util import blockdiag
    >>> (m,n) = (3,2)
    >>> A = np.random.rand(m,m)
    >>> B = np.random.rand(n,n)
    >>> print(blockdiag(A,B))
    >>> A = scipy.sparse.coo_matrix([[1,2],[3,4]])
    >>> B = scipy.sparse.coo_matrix([[5,6],[7,8]])
    >>> C = blockdiag(A,B)
    >>> print(C)
    >>> print(C.toarray())
    
    """ 
    
    # Retrieve the shapes of the two matrices    
    
    nra = A.shape[0]
    nca = A.shape[1]
    nrb = B.shape[0]
    ncb = B.shape[1]
    
    # Check if the two matrices are sparse
    
    issparse_bool = issparse(A) and issparse(B)
      
    if (issparse_bool):
        C = block_diag((A,B))    
    else:    
        # Create the corresponding matrix as a numpy array
        C   = np.zeros(shape=(nra+nrb,nca+ncb))
        # Build C as a collection of blocks
        C[0:nra,0:nca]             = A
        C[nra:nra+nrb,nca:nca+ncb] = B

    
    return C
 
## Method.    
#  @param A Matrix given as a numpy array
#  @param B Matrix given as a numpy array
#  @param C Matrix given as a numpy array
#  @param D Matrix given as a numpy array   
    
def blockstructured(A,B,C,D):
    """
    Define a block structured matrix E such as 
    E = [A   C]
        [D   B],
    where A, B, C and D are supposed to be rectangular blocks of appropriate shapes.

    Notes
    -----
    Type-FREE method if all blocks have same type.

    Parameters
    ----------
    A, B, C, D: array_like
        
    Returns
    -------
    E : ndarray, of appropriate shape  
    E = [A   C]
        [D   B]
    E is returned as a dense or sparse matrix depending on the properties of the subblocks.
    
    Examples
    --------
    >>> import numpy as np
    >>> from util import blockstructured, blockdiag
    >>> import scipy
    >>> (m,n) = (3,2)
    >>> A = np.random.rand(m,m)
    >>> B = np.random.rand(n,n)
    >>> print(blockstructured(A,B,np.zeros(shape=(m,n)),np.zeros(shape=(n,m)))-blockdiag(A,B))
    >>> A = scipy.sparse.coo_matrix([[1,2],[3,4]])
    >>> B = scipy.sparse.coo_matrix([[5,6],[7,8]])
    >>> C = blockstructured(A,B,A,B)
    >>> print(C)
    >>> print(C.toarray())
    """ 
        
    # Retrieve the shapes of the matrices    
    
    nra = A.shape[0]
    nca = A.shape[1]
    nrb = B.shape[0]
    ncb = B.shape[1]
    nrc = C.shape[0]
    ncc = C.shape[1]
    nrd = D.shape[0]
    ncd = D.shape[1]
 
    assert nra == nrc, "The first  and third matrices do not have the same number of rows."
    assert nrb == nrd, "The second and fourth matrices do not have the same number of rows." 

    assert nca == ncd, "The first and fourth matrices do not have the same number of columns."
    assert ncb == ncc, "The second and third matrices do not have the same number of columns."

    # Check if the matrices are sparse
    
    issparse_bool = issparse(A) and issparse(B) and issparse(C) and issparse(D)
    
    # Build the resulting 2 by 2 block matrix
    
    if (issparse_bool):
        E = bmat([[A,C],[D,B]])    
    else:    
        # Create the corresponding matrix as a numpy array
        E   = np.zeros(shape=(nra+nrd,nca+ncc))
        # A first approach to build E in the dense case
        E[0:nra,0:nca]             = A
        E[nra:nra+nrd,nca:nca+ncb] = B
        E[0:nra,nca:nca+ncc]       = C
        E[nra:nra+nrd,0:nca]       = D

    return E    

## Method.
#  @param A Matrix given as a numpy array
    
def cholesky_qr(A):
    """
    Performs the Cholesky QR factorization (in two steps) of A i.e. A = Q R. 
    
    Notes
    ----- 
    This algorithm is recommanded when the matrix A is tall and skinny.
    The algorithm is known as Cholesky QR 2 due to the two steps performed for numerical stability.
    The current implementation is currently available in real arithmetic only.
    
    Type-DEPENDENT method.
    
    Parameters
    ----------
    A : array_like, shape (m, n)
        matrix being decomposed
        
    Returns
    -------
    Q : ndarray, shape (m, n)
        orthogonal factor of the decomposition
        
    R : ndarray, shape (n, n)
        upper triangular factor of the decomposition
        
    References
    ----------
    .. [CQR2] 
    """
    
    # First compute the Cholesky factor of A'*A
    R_first_step = linalg.cholesky(np.dot(A.T,A))
    
    # Deduce Q by solving a lower triangular system
    W = linalg.solve_triangular(R_first_step,A.T,trans='T')
    Q = W.T
    
    # Apply the same sequence of operations to Q
    # First compute the Cholesky factor of Q'*Q
    R_second_step = linalg.cholesky(np.dot(Q.T,Q))
 
    # Deduce Q by solving a lower triangular system
    W = linalg.solve_triangular(R_second_step,Q.T,trans='T')
    Q = W.T  
    
    return (Q,np.dot(R_second_step,R_first_step))


class Test_Cholesky_QR(unittest.TestCase):

    def test_dense(self):

        logging.info('Running Test_Cholesky_QR.test_dense...')
        logging.info('\n')
        
        err_approx = []
        err_orthog = []
        
        tolerance = 1.e-10
        
        # Perform a sequence of tests
        
        for (m, n) in [(40000, 200), (4000, 100), (200, 10), (100, 20), (40, 20)]:
            A    = np.random.rand(m,n)
            logging.info('Factorization of a (%d, %d) random dense matrix',m,n)
            Q, R = cholesky_qr(A) 
            # Compute and check numerical errors
            logging.info('Computing errors')
            err = linalg.norm(A-np.dot(Q,R))
            self.assertTrue(err < tolerance)
            err_approx.append(err)
            err = linalg.norm(np.eye(n)-np.dot(Q.T,Q))
            err_orthog.append(err)
            self.assertTrue(err < tolerance)
        
        # Print final informations related to errors
        # which should be close to machine precision
        
        logging.info('Printing errors...')
        logging.info('err_approx = \n%s', np.asarray(err_approx))
        logging.info('err_orthog = \n%s', np.asarray(err_orthog))
        
        logging.info('\n')
        logging.info('End of Test_Cholesky_QR.test_dense...')
    
## Method.
#  @param A Matrix given as a csr type sparse matrix
    
def sparse_extract_subblock_csr(A,row_b,row_e,col_b,col_e):
    """
    Extract a subblock of A made of contiguous rows and columns. 
    Each row of A is indexed by an integer in the interval    [row_b,row_e]
    Each column of A is indexed by an integer in the interval [col_b,col_e]
    A is assumed to be in csr format.
    
    Notes
    ----- 
    The subblock is a sparse matrix of csr format of smaller size.
    
    Type-DEPENDENT method.
    
    Parameters
    ----------
    A : sparse matrix (csr type), shape (m, n)
        matrix being decomposed
        
    Returns
    -------
    B: subblock in sparse csr format.
        
    References
    ----------
    See related topic at 
    https://stackoverflow.com/questions/7609108/slicing-sparse-scipy-matrix
    concerning performance.
    """
    
    # Check inputs
    
    (m,n) = A.shape[:]
    
    assert 0<= row_b <= m , "Wrong indices for the row slicing of A"
    assert 0<= row_e <= m , "Wrong indices for the row slicing of A"
    assert 0<= col_b <= n , "Wrong indices for the column slicing of A"
    assert 0<= col_e <= n , "Wrong indices for the column slicing of A"
    assert row_e >= row_b, "Wrong indices for the row slicing of A"
    assert col_e >= col_b, "Wrong indices for the column slicing of A"
    assert issparse(A) == True, "A is not of sparse format"
    assert isspmatrix_csr(A) == True, "A is not of sparse csr format"
    
    # Create the lists related to the rows and columns
    
    list_r = [i for i in range(row_b,row_e)]
    
    list_c = [i for i in range(col_b,col_e)]
    
    # Convert A into a linked list matrix for easy slicing
    
    W = A.tolil()
    
    # Slice the W matrix and obtain the linked list matrix B
    
    B = W[list_r][:, list_c]
    
    # Convert to a csr matrix
        
    return B.tocsr()

## Method.
#  @param A Matrix given as a csr type sparse matrix
    
def sparse_insert_subblock_csr(A,row_b,row_e,col_b,col_e,B,do_transpose=False):
    """
    Insert a subblock A made of contiguous rows and columns into B. 
    Each row of B is indexed by an integer in the interval    [row_b,row_e]
    Each column of B is indexed by an integer in the interval [col_b,col_e]
    A is assumed to be in csr format.
    
    Notes
    ----- 
    The subblock is a sparse matrix of csr format of smaller size.
    The do_transpose variable is a boolean. If True, the transpose of A is inserted in B as:
        
        B[col_b:col_e,row_b:row_e] = transpose_of_A
    
    Type-DEPENDENT method.
    
    Parameters
    ----------
    A : sparse matrix (csr type), shape (m, n)
        matrix being decomposed
        
    Returns
    -------
    Matrix in sparse csr format.
        
    References
    ----------
    See https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    
    # Check inputs
    
    (m,n) = B.shape[:]
    
    assert 0<= row_b <= m , "Wrong indices for the row slicing of B"
    assert 0<= row_e <= m , "Wrong indices for the row slicing of B"
    assert 0<= col_b <= n , "Wrong indices for the column slicing of B"
    assert 0<= col_e <= n , "Wrong indices for the column slicing of B"
    assert row_e >= row_b, "Wrong indices for the row slicing of B"
    assert col_e >= col_b, "Wrong indices for the column slicing of B"
    assert issparse(A) == True, "A is not of sparse format"
    assert isspmatrix_csr(A) == True, "A is not of sparse csr format"
    assert issparse(B) == True, "B is not of sparse format"
    assert isspmatrix_csr(B) == True, "B is not of sparse csr format"
    
    
    # Convert A into a linked list matrix for easy slicing
    
    W = A.tolil()
    
    # Slice the B matrix and obtain the linked list matrix B
    
    X = B.tolil()
      
    if (do_transpose):   
        X[col_b:col_e,row_b:row_e] = W.transpose() 
    else:
        X[row_b:row_e,col_b:col_e] = W
        
    
    # Convert back to a csr matrix
    
    print(X.toarray())     
    
    return X.tocsr()

    
if __name__ == '__main__':
    
    import scipy
    
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()
    
    #A = scipy.sparse.coo_matrix([[1,2],[3,4]])
    #B = scipy.sparse.coo_matrix([[5,6],[7,8]])
    #C = blockstructured(A,B,A,B)
    #print(C)
    #print(C.toarray())
    #m = 4
    #A = np.random.rand(m,m)
    #C = blockstructured(A,A,A,A)
    #print(C)
    
    