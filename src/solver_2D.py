import re
from multiprocessing import Pool
import matplotlib.pyplot as plt
import meshio
import glob
import numpy as np
import timeit
import time
from skfem import *
from skfem.helpers import dot, grad, mul, prod, eye
from skfem.io import from_meshio
from skfem.io.meshio import to_file
from skfem.visuals.matplotlib import plot
import os

class TwoDsolver:
    def __init__ (self,path):
        self.name = '2D'
        self.vtu_mesh = glob.glob(f'{path}/*.vtu')[0]
        self.data_vm = sorted(glob.glob(f'{path}/vm_*d*/*.txt'))
        self._read_mesh()
        self._basis_fun()
        self._tensor_conductivity()
        self._matrix_A()
        self.electrode1 = 9640
        self.electrode2 = 9880


    def _read_mesh(self):
        start = timeit.default_timer()
        self.mesh = meshio.read(self.vtu_mesh)
        self.mesh.cell_data_to_sets('material')

        self.m = from_meshio(self.mesh)
        end = timeit.default_timer()
        print(fr"time read {self.vtu_mesh.split('/')[-1]}: {end - start}")

    def _basis_fun(self):
        self.element_type = ElementTriP1()
        self.basis = Basis(self.m,self.element_type)
        self.basis0 = self.basis.with_element((ElementTriP0()))


    def _tensor_conductivity(self):
        self.sigma_i = self.basis0.zeros() + 0.174
        self.sigma_e = self.basis0.zeros() + 0.625
        self.sigma_i[self.basis0.get_dofs(elements='set-material-1')] = 0
        self.sigma_e[self.basis0.get_dofs(elements='set-material-1')] = 0.7

    def _matrix_A(self):
        self.A = asm(self.laplace, self.basis, sigma_i=self.basis0.interpolate(self.sigma_i),
                     sigma_e=self.basis0.interpolate(self.sigma_e))
    @BilinearForm
    def laplace(phi, psi, options):
        return dot((options['sigma_i'] + options['sigma_e']) * grad(phi), grad(psi))

    @LinearForm
    def rhs(psi, options):
        return dot(options['sigma_i'] * grad(options['v']), grad(psi))

    @BilinearForm
    def laplace_A_residual(phi, psi, options):
        return dot(options['sigma_i'] * grad(phi), grad(psi))

    def _solver(self,A,b):
        return solve(A,b)

    def full_solver(self):
        start = timeit.default_timer()

        res_full = []

        # preconditioners = build_pc_ilu(A, drop_tol=1e-3,)
        # preconditioners = None
        j = 0
        for vm_txt in self.data_vm:
            # print(vm_txt)
            vm = np.loadtxt(vm_txt)
            b = asm(self.rhs, self.basis, v=self.basis.interpolate(vm), sigma_i=self.basis0.interpolate(self.sigma_i))
            # x = solve(A, b, solver=solver_iter_pcg(M=preconditioners))
            x = self._solver(self.A,b)

            '''
            saved vtu file
            '''
            # to_file(self.m ,f'visualization/data_potential/{j}.vtu',point_data={'solution':x})
            j += 1
            res_full.append(x[9640] - x[9880])
        end = timeit.default_timer()
        np.save('visualization/npy_file/full_method_2d.npy', res_full)
        print(f"time full method 2D:{end - start}")
        # time = list(range(len(res_full)))
        # plt.plot(time, res_full)
        # plt.show()


    def FL_method(self):
        start = timeit.default_timer()

        b1 = self.basis.zeros()
        b2 = self.basis.zeros()
        # точки электродов для экг
        b1[self.electrode1] = -1
        b2[self.electrode2] = +1
        b = b1 + b2
        x = self._solver(self.A, b)
        res_FL = []
        A_residual = asm(self.laplace_A_residual, self.basis, sigma_i=self.basis0.interpolate(self.sigma_i))
        Ax = (x * A_residual)

        for vm_txt in self.data_vm:
            # print(vm_txt)
            vm = np.loadtxt(vm_txt)
            ECG_value = dot(Ax, vm)
            # ECG_value = dot(x, A_residual * vm)
            res_FL.append(ECG_value)
        np.save(fr'visualization/npy_file/FL_method_{self.name}.npy', res_FL)
        end = timeit.default_timer()
        print(f"time FL method {self.name}:{end - start}")
        time = list(range(len(res_FL)))
        plt.plot(time, res_FL)
        plt.show()


    def discrete_method(self):
        start = timeit.default_timer()
        res_discrete_method = []
        self.A[50,50] += 1

        e1 = self.basis.zeros()
        e2 = self.basis.zeros()
        b = e1 + e2
        b[self.electrode1] = 1
        x1 = self._solver(self.A, b)
        b[self.electrode1] = 0
        b[self.electrode2] = 1
        x2 = self._solver(self.A, b)
        B = asm(self.laplace_A_residual, self.basis, sigma_i=self.basis0.interpolate(self.sigma_i))
        for vm_txt in self.data_vm:
            vm = np.loadtxt(vm_txt)
            b = B * vm
            res_discrete_method.append(dot(x2,b)-dot(x1,b))
        self.A[50,50] -= 1

        np.save(f'visualization/npy_file/discrete_method_{self.name}.npy', res_discrete_method)
        end = timeit.default_timer()
        print(f"time discrete_method  {self.name}:{end - start}")
        time = list(range(len(res_discrete_method)))
        plt.plot(time, res_discrete_method)
        plt.show()

class TwoDsolverFibra(TwoDsolver):
    def __init__(self,path):
        super().__init__(path)
        self.name = '2D_Fibra'



    # def _basis_fun(self):
        # self.element_type = ElementTriP1()
        # self.basis = Basis(self.m,self.element_type)
        # self.basis0 = self.basis.with_element((ElementTriP0()))


    def _tensor_conductivity(self):
        lambda_i_f = 0.174
        lambda_i_l = 0.019
        lambda_e_f = 0.625
        lambda_e_l = 0.236

        # self.sigma_i = self.basis0.zeros() + 0.174
        # self.sigma_e = self.basis0.zeros() + 0.625
        # self.sigma_i[self.basis0.get_dofs(elements='set-material-1')] = 0
        # self.sigma_e[self.basis0.get_dofs(elements='set-material-1')] = 0.7
        self.sigma_i = np.zeros((self.basis0.nelems, 2, 2))
        self.sigma_e = np.zeros((self.basis0.nelems, 2, 2))
        self.basis2x2 = self.basis.with_element(ElementVector(ElementVector(ElementTriP0())))
        self.sigma_i[self.basis0.get_dofs(elements='set-material-1'), 0, 0] = 0.
        self.sigma_i[self.basis0.get_dofs(elements='set-material-1'), 1, 1] = 0.
        self.sigma_i[self.basis0.get_dofs(elements='set-material-2'), 0, 0] = 0.174
        self.sigma_i[self.basis0.get_dofs(elements='set-material-2'), 1, 1] = 0.019

        self.sigma_e[self.basis0.get_dofs(elements='set-material-1'), 0, 0] = 0.7
        self.sigma_e[self.basis0.get_dofs(elements='set-material-1'), 1, 1] = 0.7
        self.sigma_e[self.basis0.get_dofs(elements='set-material-2'), 0, 0] = 0.625
        self.sigma_e[self.basis0.get_dofs(elements='set-material-2'), 1, 1] = 0.236

        # for i in self.basis0.get_dofs(elements='set-material-2').flatten():
        #     self.sigma_i[i] = self._sigma(lambda_i_f, lambda_i_l, self.fiber[i])
        #     self.sigma_e[i] = self._sigma(lambda_e_f, lambda_e_l, self.fiber[i])
        #
        # for i in self.basis0.get_dofs(elements='set-material-1').flatten():
        #     self.sigma_i[i] = self._sigma(0, 0, self.fiber[i])
        #     self.sigma_e[i] = self._sigma(0.7, 0.7, self.fiber[i])

    def _matrix_A(self):
        self.A = asm(self.laplace, self.basis, sigma_i=self.basis2x2.interpolate(self.sigma_i.flatten()),
                     sigma_e=self.basis2x2.interpolate(self.sigma_e.flatten()))

    @BilinearForm
    def laplace(u, v, options):
        gradu = grad(u)
        gradv = grad(v)
        sigma_i = options['sigma_i']
        sigma_e = options['sigma_e']
        sgradu = mul(sigma_i + sigma_e, gradu)
        scal = dot(sgradu, gradv)
        return scal

    @LinearForm
    def rhs(v, optionals):
        return -dot(mul(optionals['sigma_i'], grad(optionals['v'])), grad(v))


    def  discrete_method(self):
        #TODO add solver
        print(f'RESULTS of  discrete_method_{self.name}')

    def FL_method(self):
        #TODO add solver
        print(f'RESULTS  of FL_method_{self.name}')




if __name__ == '__main__':
    t = TwoDsolver('mesh/2D')
    # t.full_solver()
    t.FL_method()
    t.discrete_method()
    d = TwoDsolverFibra('mesh/2D')
    d.FL_method()
