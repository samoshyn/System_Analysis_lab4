import numpy as np
from system_solve import *
from tabulate import tabulate as tb
from copy import deepcopy
from solve import Solve

def eval_chebyt(n, x):
    t0 =np.poly1d([1])
    t1 =np.poly1d([1,0])
    if n == 0:
        t = t0
    elif n == 1:
        t = t1
    else:
        for i in range(1,n):
            t = np.poly1d([2,0])*t1 -t0
            t0 = t1
            t1 = t
    return t(x)

def eval_sh_chebyt(n, x):
    t0 =np.poly1d([1])
    t1 =np.poly1d([1,0])
    if n == 0:
        t = t0
    elif n == 1:
        t = t1
    else:
        for i in range(1,n):
            t = np.poly1d([2,0])*t1 -t0
            t0 = t1
            t1 = t
            # print(t)
    return t(np.poly1d([2,-1]))(x)

def eval_sh_chebyu(n, x):
    t0 =np.poly1d([1])
    t1 =np.poly1d([2,0])
    if n == 0:
        t = t0
    elif n == 1:
        t = t1
    else:
        for i in range(1,n):
            t = np.poly1d([2,0])*t1 -t0
            t0 = t1
            t1 = t
    return t(np.poly1d([2,-1]))(x)



class SolveExpTh(Solve):

    def built_B(self):
        def B_average():
            """
            Vector B as average of max and min in Y. B[i] =max Y[i,:]
            :return:
            """
            b = np.tile((self.Y.max(axis=1) + self.Y.min(axis=1)) / 2, (1, self.dim[3]))
            return b

        def B_scaled():
            """
            Vector B  = Y
            :return:
            """
            return deepcopy(self.Y)

        if self.weights == 'Середнє':
            self.B = B_average()
        elif self.weights == 'Нормоване':
            self.B = B_scaled()
        else:
            exit('B not defined')
        self.B_log = np.log(self.B + 1 + self.OFFSET)

    def built_A(self):
        """
        built matrix A on shifted polynomials Chebysheva
        :param self.p:mas of deg for vector X1,X2,X3 i.e.
        :param self.X: it is matrix that has vectors X1 - X3 for example
        :return: matrix A as ndarray
        """

        def mA():
            """
            :param X: [X1, X2, X3]
            :param p: [p1,p2,p3]
            :return: m = m1*p1+m2*p2+...
            """
            m = 0
            for i in range(len(self.X)):
                m += self.X[i].shape[1] * (self.deg[i] + 1)
            return m

        def coordinate(v, deg):
            """
            :param v: vector
            :param deg: chebyshev degree polynom
            :return:column with chebyshev value of coordiate vector
            """
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            """
            :param vec: it is X that consist of X11, X12, ... vectors
            :param p: max degree for chebyshev polynom
            :return: part of matrix A for vector X1
            """
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        # k = mA()
        A = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.deg[i])
            A = np.append(A, vec, 1)
        self.A = np.matrix(A)
        

    


    def lamb(self):
        lamb = np.ndarray(shape = (self.A.shape[1],0), dtype = float)
        for i in range(self.dim[3]):
            if self.splitted_lambdas:
                boundary_1 = self.deg[0] * self.dim[0]
                boundary_2 = self.deg[1] * self.dim[1] + boundary_1
                lamb1 = self._minimize_equation(self.A[:, :boundary_1], self.B[:, i])
                lamb2 = self._minimize_equation(self.A[:, boundary_1:boundary_2], self.B[:, i])
                lamb3 = self._minimize_equation(self.A[:, boundary_2:], self.B[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.A, self.B[:, i]), axis=1)
        self.Lamb = np.matrix(lamb) #Lamb in full events


    def psi(self):
        def built_psi(lamb):
            '''
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            '''
            psi = np.ndarray(shape=(self.n, self.mX), dtype = float)
            q = 0 #iterator in lamb and A
            l = 0 #iterator in columns psi
            for k in range(len(self.X)): # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):# choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                            psi[i,l] = self.A[i,q:q+self.deg[k]]*lamb[q:q+self.deg[k], 0]
                    q+=self.deg[k]
                    l+=1
            return np.matrix(psi)

        self.Psi = list() #as list because psi[i] is matrix(not vector)
        for i in range(self.dim[3]):
            self.Psi.append(built_psi(self.Lamb[:,i]))

    

    def built_a(self):
        self.a = np.ndarray(shape=(self.mX,0), dtype=float)
        for i in range(self.dim[3]):
            a1 = self._minimize_equation(self.Psi[i][:, :self.dim_integral[0]], self.Y[:, i]+self.OFFSET)
            a2 = self._minimize_equation(self.Psi[i][:, self.dim_integral[0]:self.dim_integral[1]], self.Y[:, i]+self.OFFSET)
            a3 = self._minimize_equation(self.Psi[i][:, self.dim_integral[1]:], self.Y[:, i]+self.OFFSET)
            # temp = self._minimize_equation(self.Psi[i], self.Y[:, i])
            # self.a = np.append(self.a, temp, axis=1)
            self.a = np.append(self.a, np.vstack((a1, a2, a3)),axis = 1)


    def built_F1i(self, psi, a):
        """
        not use; it used in next function
        :param psi: matrix psi (only one
        :param a: vector with shape = (6,1)
        :param dim_integral:  = [3,4,6]//fibonacci of deg
        :return: matrix of (three) components with F1 F2 and F3
        """
        m = len(self.X)  # m  = 3
        F1i = np.ndarray(shape=(self.n, m), dtype=float)
        k = 0  # point of beginning column to multiply
        for j in range(m):  # 0 - 2
            for i in range(self.n):  # 0 - 49
                try:
                    F1i[i, j] = psi[i, k:self.dim_integral[j]] * a[k:self.dim_integral[j], 0]
                except:
                    F1i[i, j] = 0
            k = self.dim_integral[j]
        return np.matrix(F1i)

    
    def built_Fi(self):
        self.Fi = []
        for i in range(self.dim[3]):
            self.Fi.append(self.built_F1i(self.Psi[i],self.a[:,i]))

    def built_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.dim[3]):
            #self.c = np.append(self.c, self._minimize_equation(self.Fi[i].T*self.Fi[i], self.Fi[i].T*self.Y[:,i],self.eps + self.OFFSET),\
            #                   axis=1)
            self.c = np.append(self.c, conjugate_gradient_method_v2(self.Fi[i].T*self.Fi[i], self.Fi[i].T*self.Y[:,i],self.eps),\
                              axis=1)
        

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                try:
                    F[i, j] = self.Fi[j][i,:]*self.c[:,j]
                except:
                    F[i, j] = 0
        self.F = np.matrix(F) - self.OFFSET  # F = exp(sum(c*tanh(Fi))) - 1
        self.norm_error = []
        for i in range(self.Y.shape[1]):
            self.norm_error.append(np.linalg.norm(self.Y[:, i] - self.F[:, i], np.inf))


    def aggregate(self, values, coeffs):
        return np.exp(np.dot(np.tanh(values), coeffs)) - 1

    def calculate_value(self, X):
        def calculate_polynomials(value, deg_lim):  # deg_lim is not reached
            return np.array([self.poly_f(deg, value) for deg in range(deg_lim)]).T

        X = np.array(X)
        X = (X - self.minX) / (self.maxX - self.minX)
        for i in range(len(X)):
            if np.isnan(X[i]):
                X[i] = 1
        X = np.split(X, self.dim_integral[:2])
        phi = np.array([calculate_polynomials(vector, self.deg[i]) for i, vector in enumerate(X)])
        psi = list()
        shift = 0
        for i in range(3):
            for j in range(self.dim[i]):
                psi.append(self.aggregate(phi[i][j], self.Lamb.A[shift: shift + self.deg[i]]))
                shift += self.deg[i]
        psi = np.array(psi).T
        big_phi = list()
        for i in range(3):
            try:
                rand_a = [self.aggregate(psi[k, (self.dim_integral[i - 1] if i > 0 else 0): self.dim_integral[i]],
                                           self.a.A[(self.dim_integral[i - 1] if i > 0 else 0):
                                           self.dim_integral[i], k]) for k in range(self.dim[3])]
            except:
                rand_a = [0 for k in range(self.dim[3])]
            big_phi.append(rand_a)
        big_phi = np.array(big_phi).T
        try:
            result = np.array([self.aggregate(big_phi[k], self.c.A[:, k]) for k in range(self.dim[3])])
        except:
            result = 0
        result = result * (self.maxY - self.minY) + self.minY
        return result


    def show(self):
        text = []
        text.append('\nError normalised (Y - F)')
        text.append(tb([self.norm_error]))

        text.append('\nError (Y_ - F_))')
        text.append(tb([self.error]))

        text.append('Input data: X')
        text.append(tb(np.array(self.datas[:, :self.dim_integral[2]])))

        text.append('\nInput data: Y')
        text.append(tb(np.array(self.datas[:, self.dim_integral[2]:self.dim_integral[3]])))

        text.append('\nX normalised:')
        text.append(tb(np.array(self.data[:, :self.dim_integral[2]])))

        text.append('\nY normalised:')
        text.append(tb(np.array(self.data[:, self.dim_integral[2]:self.dim_integral[3]])))

        text.append('\nmatrix B:')
        text.append(tb(np.array(self.B)))

        # text.append('\nmatrix A:')
        # text.append(tb(np.array(self.A)))

        text.append('\nmatrix Lambda:')
        text.append(tb(np.array(self.Lamb)))

        for j in range(len(self.Psi)):
            s = '\nmatrix Psi%i:' % (j + 1)
            text.append(s)
            text.append(tb(np.array(self.Psi[j])))

        text.append('\nmatrix a:')
        text.append(tb(self.a.tolist()))

        for j in range(len(self.Fi)):
            s = '\nmatrix F%i:' % (j + 1)
            text.append(s)
            text.append(tb(np.array(self.Fi[j])))

        text.append('\nmatrix c:')
        text.append(tb(np.array(self.c)))

        text.append('\nY rebuilt normalized :')
        text.append(tb(np.array(self.F)))

        text.append('\nY rebuilt :')
        text.append(tb(self.F_.tolist()))
        return '\n'.join(text)

