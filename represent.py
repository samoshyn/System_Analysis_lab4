import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as pnm
from os import name as os_name
from solve import Solve
import basis as b_gen
from show_polynomial import _Polynom

class PolynomialBuilder(object):
    def __init__(self, solution):
        assert isinstance(solution, Solve)
        self._solution = solution
        max_degree = max(solution.deg) - 1
        if solution.poly_type == 'Зміщені поліноми Чебишева':
            self.symbol = 'T'
            self.basis = b_gen.basis_sh_chebyshev(max_degree)
        elif solution.poly_type == 'Поліноми Чебишева':
            self.symbol = 'K'
            self.basis = b_gen.basis_chebyshev(max_degree)
        elif solution.poly_type == 'Зміщені полномі Чебишева 2 роду':
            self.symbol = 'U'
            self.basis = b_gen.basis_sh_chebyshev_2_shrinked(max_degree)
        assert self.symbol
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).getA1() for X in solution.X_]
        self.maxX = [X.max(axis=0).getA1() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).getA1()
        self.maxY = solution.Y_.max(axis=0).getA1()

    def _form_lamb_lists(self):
        """
        Generates specific basis coefficients for Psi functions
        """
        self.lamb = list()
        for i in range(self._solution.Y.shape[1]):  # `i` is an index for Y
            lamb_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                lamb_i_j = list()
                for k in range(self._solution.dim[j]):  # `k` is an index for vector component
                    lamb_i_jk = self._solution.Lamb[shift:shift + self._solution.deg[j], i].getA1()
                    shift += self._solution.deg[j]
                    lamb_i_j.append(lamb_i_jk)
                lamb_i.append(lamb_i_j)
            self.lamb.append(lamb_i)

    def _transform_to_standard(self, coeffs):
        """
        Transforms special polynomial to standard
        :param coeffs: coefficients of special polynomial
        :return: coefficients of standard polynomial
        """
        std_coeffs = np.zeros(coeffs.shape)
        for index in range(coeffs.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            std_coeffs += coeffs[index] * cp
        return std_coeffs

    def _print_psi_i_jk(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            strings.append('{0:.6f}\*{symbol}{deg}(x{1}{2})'.format(self.lamb[i][j][k][n], j + 1, k + 1,
                                                                   symbol=self.symbol, deg=n))
        return ' + '.join(strings)

    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            shift = sum(self._solution.dim[:j]) + k
            for n in range(len(self.lamb[i][j][k])):
                strings.append('{0:.6f}\*{symbol}{deg}(x{1}{2})'.format(self.a[i][shift] * self.lamb[i][j][k][n],
                                                                       j + 1, k + 1, symbol=self.symbol, deg=n))
        return ' + '.join(strings)

    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lamb[i][j][k])):
                    strings.append('{0:.6f}\*{symbol}{deg}(x{1}{2})'.format(self.c[i][j] * self.a[i][shift] *
                                                                           self.lamb[i][j][k][n],
                                                                           j + 1, k + 1, symbol=self.symbol, deg=n))
        return ' + '.join(strings)

    def _print_F_i_transformed_denormed(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                raw_coeffs = self._transform_to_standard(self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                    #print(current_poly)
                    #print(add_poly)
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                constant += current_poly[0]
                current_poly[0] = 0
                #print(current_poly)
                #print(i, j, k,"kek")
                current_poly = np.poly1d(current_poly.coeffs, variable='(x{0}{1})'.format(j + 1, k + 1))
                strings.append(str(_Polynom(current_poly, '(x{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return ' +\n'.join(strings)

    def _print_F_i_transformed(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                current_poly = np.poly1d(self._transform_to_standard(self.c[i][j] * self.a[i][shift] *
                                                                     self.psi[i][j][k])[::-1],
                                         variable='(x{0}{1})'.format(j + 1, k + 1))
                constant += current_poly[0]
                current_poly[0] = 0
                strings.append(str(_Polynom(current_poly, '(x{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return ' +\n'.join(strings)

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        self._form_lamb_lists()
        psi_strings = ['**(Psi{1}{2})[{0}]**={result}\n'.format(i + 1, j + 1, k + 1, result=self._print_psi_i_jk(i, j, k))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = ['**(Phi{1})[{0}]**={result}\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = ['**(F{0})**={result}\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(self._solution.Y.shape[1])]
        
        return '\n'.join(psi_strings + phi_strings)

    def plot_graphs(self):
        fig, axes = plt.subplots(2, self._solution.Y.shape[1])
        if self._solution.Y.shape[1] == 1:
            axes[0] = [axes[0]]
            axes[1] = [axes[1]]
        for index in range(self._solution.Y.shape[1]):
            ax = axes[0][index]  # real and estimated graphs
            norm_ax = axes[1][index]  # abs residual graph
            ax.set_xticks(np.arange(0, self._solution.n + 1, 5))
            ax.plot(np.arange(1, self._solution.n + 1), self._solution.Y_[:, index],
                    'r-', label='$Y_{0}$'.format(index + 1))
            ax.plot(np.arange(1, self._solution.n + 1), self._solution.F_[:, index],
                    'b-', label='$F_{0}$'.format(index + 1))
            ax.legend(loc='upper right', fontsize=16)
            ax.set_title('Coordinate {0}'.format(index + 1))
            ax.grid()

            norm_ax.set_xticks(np.arange(0, self._solution.n + 1, 5))
            norm_ax.plot(np.arange(1, self._solution.n + 1),
                         abs(self._solution.Y_[:, index] - self._solution.F_[:, index]), 'g-')
            norm_ax.set_title('Residual {0}'.format(index + 1))
            norm_ax.grid()

        manager = plt.get_current_fig_manager()
        manager.set_window_title('Graph')
        if os_name == 'posix':
            fig.show()
        else:
            plt.show()

    def compare_vals(self, name, real, predicted, reconstructed=None):
        fig = plt.figure()
        axes = plt.axes()
        r = np.arange(len(real))
        axes.set_title(name)
        axes.set_xlim(0, len(real))
        axes.grid()
        axes.plot(r, predicted, label='predicted')
        if reconstructed != None:
            axes.plot(r, reconstructed, label='reconstructed')
        axes.plot(r, real, label='real')
        axes.legend(loc='upper right', fontsize=16)
        if os_name == 'posix':
            fig.show()
        else:
            plt.show()

    def plot_graphs_with_prediction(self, steps):
        XF, YF = self._solution.build_predicted(steps)
        for i, x in enumerate(self._solution.X_):
            for j, xc in enumerate(x.T):
                self.compare_vals('X{}{}'.format(i + 1, j + 1), xc.getA1(), XF[i][j])
        for i in range(self._solution.dim[3]):
            self.compare_vals('Y{}'.format(i + 1), self._solution.Y_[:, i].getA1(), YF[:, i],
                              self._solution.F_[:, i].getA1())

class PolynomialBuilderMult(object):
    def __init__(self, solution):
        assert isinstance(solution, Solve)
        self._solution = solution
        max_degree = max(solution.deg) - 1
        if solution.poly_type == 'Зміщені поліноми Чебишева':
            self.symbol = 'T'
            self.basis = b_gen.basis_sh_chebyshev(max_degree)
        elif solution.poly_type == 'Поліноми Чебишева':
            self.symbol = 'K'
            self.basis = b_gen.basis_chebyshev(max_degree)
        elif solution.poly_type == 'Зміщені полномі Чебишева 2 роду':
            self.symbol = 'U'
            self.basis = b_gen.basis_sh_chebyshev_2_shrinked(max_degree)
        assert self.symbol
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).getA1() for X in solution.X_]
        self.maxX = [X.max(axis=0).getA1() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).getA1()
        self.maxY = solution.Y_.max(axis=0).getA1()

    def _form_lamb_lists(self):
        """
        Generates specific basis coefficients for Psi functions
        """
        self.lamb = list()
        for i in range(self._solution.Y.shape[1]):  # `i` is an index for Y
            lamb_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                lamb_i_j = list()
                for k in range(self._solution.dim[j]):  # `k` is an index for vector component
                    lamb_i_jk = self._solution.Lamb[shift:shift + self._solution.deg[j], i].getA1()
                    shift += self._solution.deg[j]
                    lamb_i_j.append(lamb_i_jk)
                lamb_i.append(lamb_i_j)
            self.lamb.append(lamb_i)

    def _print_psi_i_jk(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            strings.append('(1 + {symbol}{deg}(x{1}{2}))^({0:.6f})'.format(self.lamb[i][j][k][n], j + 1, k + 1,
                                                                           symbol=self.symbol, deg=n))
        return ' * '.join(strings)

    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            shift = sum(self._solution.dim[:j]) + k
            for n in range(len(self.lamb[i][j][k])):
                strings.append('(1 + {symbol}{deg}(x{1}{2}))^({0:.6f})'.format(self.a[i][shift] * self.lamb[i][j][k][n],
                                                                               j + 1, k + 1, symbol=self.symbol, deg=n))
        return ' * '.join(strings)

    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lamb[i][j][k])):
                    strings.append('(1 + {symbol}{deg}(x{1}{2}))^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] *
                                                                                   self.lamb[i][j][k][n],
                                                                                   j + 1, k + 1, symbol=self.symbol,
                                                                                   deg=n))
        return ' * '.join(strings)

    def _print_F_i_transformed(self, i):
        """
        Returns string of F function in regular polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        power_sum = 0
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                power_sum += self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][0]
                for n in range(1, len(self.lamb[i][j][k])):
                    summands = ['{0}(x{1}{2})^{deg}'.format(self.basis[n].coef[index], j + 1, k + 1, deg=index)
                                for index in range(1, len(self.basis[n].coef)) if self.basis[n].coef[index] != 0]
                    if self.basis[n].coef[0] != -1:
                        summands.insert(0, str(1 + self.basis[n].coef[0]))
                    strings.append('({repr})^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                                               j + 1, k + 1, repr=' + '.join(summands)))
        strings.insert(0, str((1 + self.basis[0].coef[0]) ** (power_sum)))
        return ' * '.join(strings)

    def _print_F_i_transformed_recovered(self, i):
        """
        Returns string of recovered F function in regular polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        power_sum = 0
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = pnm([- self.minX[j][k] / diff, 1 / diff])
                power_sum += self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][0]
                for n in range(1, len(self.lamb[i][j][k])):
                    res_polynomial = self.basis[n](mult_poly) + 1
                    coeffs = res_polynomial.coef
                    summands = ['{0}(x{1}{2})^{deg}'.format(coeffs[index], j + 1, k + 1, deg=index)
                                for index in range(1, len(coeffs))]
                    summands.insert(0, str(coeffs[0]))
                    strings.append('({repr})^({0:.6f})'.format(self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n],
                                                               j + 1, k + 1, repr=' + '.join(summands)))
        strings.insert(0, str((self.maxY[i] - self.minY[i]) * (1 + self.basis[0].coef[0]) ** (power_sum)))
        return ' * '.join(strings) + ' + ' + str((2 * self.minY[i] - self.maxY[i]))

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        if self.symbol == 'CC':
            return ''
        self._form_lamb_lists()
        psi_strings = ['**(Psi{1}{2})[{0}]**={result} - 1\n'.format(i + 1, j + 1, k + 1,
                                                                 result=self._print_psi_i_jk(i, j, k))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = ['**(Phi{1})[{0}]**={result} - 1\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = ['F^{0} in special basis:\n{result} - 1\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(self._solution.Y.shape[1])]
        return '\n'.join(psi_strings + phi_strings)


class PolynomialBuilderExpTh(PolynomialBuilderMult):
    def _print_psi_i_jk(self, i, j, k, mode=0):
        """
        Returns string of Psi function
        mode = 0 -  in special polynomial form
        mode = 1 -  in regular polynomial form
        mode = 2 -  in regular polynomial form with restored X
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.lamb[i][j][k])):
            inner = 'stub'
            if mode == 0:
                inner = '{symbol}{deg}(x{0}{1})'.format(j + 1, k + 1, symbol=self.symbol, deg=n)
            elif mode == 1:
                inner = str(self.basis[n].coef[0])
                if n > 0:
                    inner += ' + ' + ' + '.join('({coef})(x{0}{1})^{deg}'.format(j + 1, k + 1, coef=coef, deg=index)
                                                for index, coef in enumerate(self.basis[n].coef) if index > 0)
            elif mode == 2:
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = pnm([- self.minX[j][k] / diff, 1 / diff])
                cur_poly = self.basis[n](mult_poly)
                inner = str(cur_poly.coef[0])
                if n > 0:
                    inner += ' + ' + ' + '.join('({coef})(x{0}{1})^{deg}'.format(j + 1, k + 1, coef=coef, deg=index)
                                                for index, coef in enumerate(cur_poly.coef) if index > 0)
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.lamb[i][j][k][n], inner=inner))
        return ' * '.join(strings) + ' - 1'

    def _print_phi_i_j(self, i, j, mode=0):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.lamb[i][j])):
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.a[i][sum(self._solution.dim[:j]) + k],
                                                               inner=self._print_psi_i_jk(i, j, k, mode)))
        return ' * '.join(strings) + ' - 1'

    def _print_F_i(self, i, mode=0):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            strings.append('exp({0:.6f}*tanh({inner}))'.format(self.c[i][j], inner=self._print_phi_i_j(i, j, mode)))
        if mode == 2:
            strings.insert(0, str(self.maxY[i] - self.minY[i]))
            return ' * '.join(strings) + ' + (' + str((2 * self.minY[i] - self.maxY[i])) + ')'
        else:
            return ' * '.join(strings) + ' - 1'

    def get_results(self):
        """
        Generates results based on given solution
        :return: Results string
        """
        self._form_lamb_lists()
        psi_strings = ['**(Psi{1}{2})[{0}]**={result}\n'.format(i + 1, j + 1, k + 1,
                                                             result=self._print_psi_i_jk(i, j, k))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.dim[j])]
        phi_strings = ['**(Phi{1})[{0}]**={result}\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = ['F^{0} in special basis:\n{result}\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(self._solution.Y.shape[1])]
        
        return '\n'.join(psi_strings + phi_strings)
