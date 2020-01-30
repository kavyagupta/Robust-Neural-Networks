import numpy as np
import time
import pickle
def sparse_matrix(w):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if (np.absolute(i - j) >= np.ceil(w.shape[1] / 2)) :
                w[i][j] = 0
    return w


def Constraint(w, Y, A, B, nit, rho):
    gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
    for i in range(nit):
        w_new = w - (np.transpose(A) @ Y @ np.transpose(B))
     #   w_new *= np.greater_equal(w_new, 0) # ensure non-negative weights
        w_new = sparse_matrix(w_new)
        print(w_new)
        print('L cst = ', np.linalg.norm(A @ w_new @ B, ord=2))
        T = A @ w_new @ B
        [u,s,v] = np.linalg.svd(T)
        criterion = np.linalg.norm(w_new - w, ord='fro')
        constraint = np.linalg.norm(s[s > rho] - rho, ord=2)
        print( 'iteration:', i+1, 'criterion: ', criterion, 'constraint: ', constraint)
        Yt = Y + gam * T
        [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
        s1 = np.clip(s1, 0, rho)
        Y = Yt - gam * np.dot(u1 * s1, v1)
        if (criterion < 30 and constraint < 0.001):
            return w_new
    return w_new

p = 7
n = 19
m = 17
q = 28

gen_new_set = True 
if (gen_new_set):
    w = np.random.random((m, n))
    A = np.random.random((p, n))
    B = np.random.random((m, q))
    Y0 = np.zeros([p,q])
    gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
    nit = 10000
    rho = 1

#    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#        pickle.dump([w, A, B, Y0, nit, rho,gam], f)
#else:
#    with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
#        w, A, B, Y0, nit, rho,gam = pickle.load(f)
#print('Lipschitz constant before: ', np.linalg.norm(A @ w @ B, ord=2))
tic = time.time()
w_new = Constraint(np.transpose(w), Y0, A, B, nit, rho)
toc = time.time()

#print(toc-tic)
print('Lipschitz constant after: ', np.linalg.norm(A @ w_new @ B, ord=2))
#print(w_new)


