from multiprocessing import Pool
import matplotlib.pyplot as plt
import meshio
import glob
import numpy as np
import timeit
from skfem import *
from skfem.helpers import dot, grad, mul, prod, eye
from skfem.io import from_meshio
from solver_2D import TwoDsolver,TwoDsolverFibra

class ThreeDsolver(TwoDsolver):
    def __init__(self,path):
        super().__init__(path)
        self.name = '3D'
        # self.data_vm = sorted(glob.glob(f'{path}/vm_{self.name[0]}d*/*.txt'))
        # self.electrode1 = 40672
        # self.electrode2 = 43180
        '''
        if real model 
        '''
        self.electrode1 = 83715
        self.electrode2 = 83781


    def _basis_fun(self):
        self.element_type = ElementTetP1()
        self.basis = Basis(self.m,self.element_type)
        self.basis0 = self.basis.with_element((ElementTetP0()))

    def _solver(self,A,b):
        # preconditioners = build_pc_ilu(A, drop_tol=1e-3)
        preconditioners = None
        return  solve(A, b, solver=solver_iter_pcg(M=preconditioners))







class ThreeDsolverFibra(ThreeDsolver):
    def __init__(self,path):
        super().__init__(path)
        self.name = '3D_fibra'

    def _read_mesh(self):
        super()._read_mesh()

        self.fiber = self.mesh.cell_data_dict["fiber"]["tetra"]

    def _basis_fun(self):
        super()._basis_fun()
        self.basis3x3 = self.basis.with_element(ElementVector(ElementVector(ElementTetP0())))


    def _sigma(self, lambda_f, lambda_l, f):
        tensor_mult = prod(f, f)
        return lambda_f * tensor_mult + lambda_l * (eye(1, 3) - tensor_mult)

    def _tensor_conductivity(self):
        lambda_i_f = 0.174
        lambda_i_l = 0.019
        lambda_e_f = 0.625
        lambda_e_l = 0.236
        self.sigma_i = np.zeros((self.basis0.nelems, 3, 3))
        self.sigma_e = np.zeros((self.basis0.nelems, 3, 3))
        for i in self.basis0.get_dofs(elements='set-material-2').flatten():
            self.sigma_i[i] = self._sigma(lambda_i_f, lambda_i_l, self.fiber[i])
            self.sigma_e[i] = self._sigma(lambda_e_f, lambda_e_l, self.fiber[i])

        for i in self.basis0.get_dofs(elements='set-material-1').flatten():
            self.sigma_i[i] = self._sigma(0, 0, self.fiber[i])
            self.sigma_e[i] = self._sigma(0.7, 0.7, self.fiber[i])


    @BilinearForm
    def laplace(phi, psi, optionals):
        gradu = grad(phi)
        gradv = grad(psi)
        sigma_i = optionals['sigma_i']
        sigma_e = optionals['sigma_e']
        sgradu = mul(sigma_i + sigma_e, gradu)
        scal = dot(sgradu, gradv)
        return scal

    @LinearForm
    def rhs(psi, optionals):
        return -dot(mul(optionals['sigma_i'], grad(optionals['v'])), grad(psi))
    @BilinearForm
    def laplace_A_residual(phi, psi, options):
        return dot(mul(options['sigma_i'], grad(phi)), grad(psi))

    def _matrix_A(self):
       self.A = asm(self.laplace, self.basis,
                sigma_i=self.basis3x3.interpolate(self.sigma_i.flatten()),
                sigma_e=self.basis3x3.interpolate(self.sigma_e.flatten()))



    def temp_FL(self, el):
        print(s)
        vm = np.loadtxt(el)
        return dot(self. Ax, vm)

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
        A_residual = asm(self.laplace_A_residual, self.basis, sigma_i = self.basis3x3.interpolate(self.sigma_i.flatten()))
        # print('ss')
        self.Ax = (x * A_residual)
        #
        # with Pool(4) as pool:
        #     res_FL = pool.map(self.temp_FL, [vm_txt for vm_txt in self.data_vm])

        for vm_txt in self.data_vm:
            # print(vm_txt)
            vm = np.loadtxt(vm_txt)
            ECG_value = dot(self. Ax, vm)
            res_FL.append(ECG_value)
        np.save(fr'visualization/npy_file/FL_method_{self.name}.npy', res_FL)
        end = timeit.default_timer()
        print(f"time FL method {self.name}:{end - start}")
        time = list(range(len(res_FL)))
        plt.plot(time, res_FL)
        plt.show()

    def temp_DM(self,vm_txt):
            vm = np.loadtxt(vm_txt)
            b = self.B * vm
            return dot(self.x2,b)-dot(self.x1,b)
    def discrete_method(self):
        start = timeit.default_timer()
        res_discrete_method = []
        self.A[50,50] += 1

        e1 = self.basis.zeros()
        e2 = self.basis.zeros()
        b = e1 + e2
        b[self.electrode1] = 1
        self.x1 = self._solver(self.A, b)
        b[self.electrode1] = 0
        b[self.electrode2] = 1
        self.x2 = self._solver(self.A, b)
        self.B = asm(self.laplace_A_residual, self.basis, sigma_i=self.basis3x3.interpolate(self.sigma_i.flatten()))

        with Pool(4) as pool:
            res_discrete_method = pool.map(self.temp_DM, [vm_txt for vm_txt in self.data_vm])

        #
        # for vm_txt in self.data_vm:
        #     vm = np.loadtxt(vm_txt)
        #     b = B * vm
        #     res_discrete_method.append(dot(x2,b)-dot(x1,b))

        self.A[50,50] -= 1

        np.save(f'visualization/npy_file/discrete_method_{self.name}.npy', res_discrete_method)
        end = timeit.default_timer()
        print(f"time discrete_method  {self.name}:{end - start}")
        time = list(range(len(res_discrete_method)))
        plt.plot(time, res_discrete_method)
        plt.show()





# if __name__ == "__main__":
# t = ThreeDsolver('mesh/3D')
# # print(t.vtu_mesh)
# t.FL_method()
# t.discrete_method()
# print(ThreeDsolverFibra.mro())
s = ThreeDsolverFibra('mesh/real_model')
s.FL_method()
# s.discrete_method()

