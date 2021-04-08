from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N  = X.shape[0]
    C = W.shape[1]

    for i in range(N):
      score = W.T @ X[i,:]
      pdf = np.e ** score # Un-normalized pdf
      pdf /= pdf.sum() # Normalized pdf
      loss_i = -np.log(pdf[y[i]]) # Loss due to ith training example
      dW_i = np.zeros_like(W) # Gradient due to ith training example
      for j in range(C):
        if j == y[i]:
          dW_i[:,j] = (pdf[j] - 1) * X[i,:]
        else:
          dW_i[:,j] = pdf[j] * X[i,:]

      loss += loss_i    
      dW += dW_i

    loss /= N
    dW /= N

    loss += reg * np.sum(W * W) # Regularization for loss
    dW += 2 * reg * W # Regularization for gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N  = X.shape[0]

    score = X @ W
    pdf = np.e ** score
    pdf = (pdf.T / np.sum(pdf, axis=1)).T

    p_true = pdf[np.arange(N), y]
    loss = np.sum( -np.log(p_true) )
    
    pdf[np.arange(N), y] -= 1 # Amending pdf for dW calculation
    dW = X.T @ pdf

    loss /= N
    dW /= N

    loss += reg * np.sum(W * W) # Regularization for loss
    dW += 2 * reg * W # Regularization for gradient



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
