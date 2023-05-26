import matplotlib.pyplot as plt
import meshio
import glob
import numpy as np
import timeit
from skfem import *
from skfem.helpers import dot, grad, mul, prod, eye
from skfem.io import from_meshio


from skfem.io.meshio import to_file
from skfem.visuals.matplotlib import plot

mesh = meshio.read("mesh_3d.vtu")
# print(dir(mesh))
mesh.cell_data_to_sets('material')
fiber = mesh.cell_data_dict["fiber"]["tetra"]
m = from_meshio(mesh)

e = ElementTetP1()
basis = Basis(m, ElementTetP1())
basis0 = basis.with_element(ElementTetP0())
basis3x3 = basis.with_element(ElementVector(ElementVector(ElementTetP0())))

@BilinearForm
def laplace(u, v, _):
    gradu = grad(u)
    gradv = grad(v)
    sigma_i = _['sigma_i']
    sigma_e = _['sigma_e']
    sgradu = mul(sigma_i + sigma_e, gradu)
    scal = dot(sgradu, gradv)
    return scal

@LinearForm
def rhs(psi, _):
    return -dot(mul(_['sigma_i'], grad(_['v'])), grad(psi))

def sigma(lambda_f, lambda_l, f:np.array):
    tensor_mult = prod(f, f)
    return lambda_f * tensor_mult + lambda_l * (eye(1, 3) - tensor_mult)

lambda_i_f = 0.174
lambda_i_l = 0.019
lambda_e_f = 0.625
lambda_e_l = 0.236

sigma_i = np.zeros((basis0.nelems, 3, 3))
sigma_e = np.zeros((basis0.nelems, 3, 3))

for i in basis0.get_dofs(elements='set-material-2').flatten():
    sigma_i[i] = sigma(lambda_i_f, lambda_i_l, fiber[i])
    sigma_e[i] = sigma(lambda_e_f, lambda_e_l, fiber[i])

for i in basis0.get_dofs(elements='set-material-1').flatten():
    sigma_i[i] = sigma(0, 0, fiber[i])
    sigma_e[i] = sigma(0.7, 0.7, fiber[i])

A = asm(laplace, basis,
        sigma_i=basis3x3.interpolate(sigma_i.flatten()),
        sigma_e=basis3x3.interpolate(sigma_e.flatten()))
e1 = basis.zeros()
e2 = basis.zeros()

e1[40672] = -1
e2[43180] = +1
b = e1 + e2


def solve_residual_equation():

    # e1 = basis.zeros()
    # e2 = basis.zeros()
    #
    # e1[40672] = -1
    # e2[43180] = +1

    A = asm(laplace, basis, sigma_i=basis3x3.interpolate(sigma_i.flatten()), sigma_e=basis3x3.interpolate(sigma_e.flatten()))

    preconditioners = None

    b = e1 + e2
    # x = solve(A, b)
    x = solve(A, b, solver=solver_iter_pcg(M=preconditioners))

    # ECG_value = dot(x, A_residual * vm)
    # print(ECG_value)
    return x

def ECG(x:np.array):
    @BilinearForm
    def laplace_A_residual(u, v, _):
        return dot(mul(_['sigma_i'], grad(u)), grad(v))

    A_residual = asm(laplace_A_residual, basis, sigma_i=basis3x3.interpolate(sigma_i.flatten()))
    values = []
    data = glob.glob('vm_3d_600/*.txt')
    data = sorted(data)

    j = 0
    for i in data:
        print(i)
        vm = np.loadtxt(i)
        ECG_value = dot(x, A_residual * vm)
        values.append(ECG_value)
        #
        # b = asm(rhs, basis,
        #         v=basis.interpolate(vm),
        #         sigma_i=basis3x3.interpolate(sigma_i.flatten()))

        # x = solve(A, b)
        # visualize(j,x)
        # to_file(m,f'{10}_3d.vtu',point_data={'solution':x})
        # detta.append(x[9640] - x[9880])
        # j += 1
    # np.save('3d_fibra.npy',values)
    return ECG_plot(values)


def ECG_plot(delta):
    time = list(range(len(delta)))
    plt.plot(time, delta)
    plt.title('FL метод с тензором проводимости')
    plt.ylabel("Напряжение [мВ]")
    plt.xlabel("время [мс]")
    plt.legend()
    # plt.savefig('ECG_1.png')
    plt.show()

if __name__ == '__main__':
    # visualize().show()
    # y = []
    start = timeit.default_timer()
    ECG(solve_residual_equation())
    # ECG_plot(solver(y))
    end = timeit.default_timer()
    print(f"time:{end - start}")