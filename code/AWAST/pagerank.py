import numpy as np
from scipy.sparse import csc_matrix

def pageRank(G, s = .85, maxerr = .0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    n = G.shape[0]

    # transform G into markov matrix A
    A = csc_matrix(G,dtype=np.float)
    print(A)
    rsums = np.array(A.sum(1))[:,0]
    print(rsums)
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums==0

    print(A)

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0,n):
            # inlinks of state i
            Ai = np.array(A[:,i].todense())[:,0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )

    # return normalized pagerank
    return r/float(sum(r))

def pageRankUndirected(G, d = 0.85, epsilon = 0.0001):
    A = G.copy()
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                A[i, j] = float(1) / A[i, j]
    r = np.ones(n)
    for i in range(n):
        A[:,i] /= np.sum(A[:,i])
    r0 = np.zeros(n)
    while np.sum(np.abs(r - r0)) > epsilon:
        r0 = r.copy()
        r = (1-d) / float(n) + d / float(n) * np.dot(A, r)
    return r / float(sum(r))



