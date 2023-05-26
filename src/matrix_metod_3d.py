from multiprocessing import Pool
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


def sigma(lambda_f, lambda_l, f: np.array):
    tensor_mult = prod(f, f)
    return lambda_f * tensor_mult + lambda_l * (eye(1, 3) - tensor_mult)


lambda_i_f = 0.174
lambda_i_l = 0.019
lambda_e_f = 0.625
lambda_e_l = 0.236

sigma_i = np.zeros((basis0.nelems, 3, 3))
sigma_e = np.zeros((basis0.nelems, 3, 3))

for i in basis0.get_dofs(elements='set-material-2').flatten():
    sigma_i[i] = sigma(lambda_i_f, lambda_i_f, fiber[i])
    sigma_e[i] = sigma(lambda_e_f, lambda_e_l, fiber[i])

for i in basis0.get_dofs(elements='set-material-1').flatten():
    sigma_i[i] = sigma(0, 0, fiber[i])
    sigma_e[i] = sigma(0.7, 0.7, fiber[i])


@BilinearForm
def laplace_2(phi, psi, _):
    return -dot(mul(_['sigma_i'], grad(phi)), grad(psi))


B = asm(laplace_2, basis, sigma_i=basis3x3.interpolate(sigma_i.flatten()))

A = asm(laplace, basis,
        sigma_i=basis3x3.interpolate(sigma_i.flatten()),
        sigma_e=basis3x3.interpolate(sigma_e.flatten()))

n = A.shape[0]
A[50, 50] += 1
e_v1 = np.zeros(n)
e_v1[40672] = 1
e_v2 = np.zeros(n)
e_v1[43180] = 1
preconditioners = None

d_v1 = solve(A, e_v1, solver=solver_iter_pcg(M=preconditioners))
d_v2 = solve(A, e_v2, solver=solver_iter_pcg(M=preconditioners))
# d_v1 = solve(A, e_v1)
# d_v2 = solve(A, e_v2)



def temp(el):
    vm = np.loadtxt(el)
    b = B * vm
    return dot(d_v2, b) - dot(d_v1, b)


def ECG():
    # data = glob.glob('vm_biv_fixed_600/vm_0001.txt')
    # data = sorted(data)
    values = []
    vm = np.loadtxt('vm_biv_3d_600/vm_0001.txt')
    b = B * vm
    values.append(dot(d_v2, b) - dot(d_v1, b))
    # with Pool(16) as pool:
    # values = pool.starmap(temp, [(el) for el in data])
    np.save('matrix_metod_3d.npy', values)


def ECG_plot(delta):
    time = list(range(len(delta)))
    plt.plot(time, delta)
    plt.title('Матричный метод на реальной модели')
    plt.ylabel("Напряжение [мВ]")
    plt.xlabel("Время [мс]")
    plt.savefig('ECG_biv.png')
    plt.show()


if __name__ == '__main__':
    # visualize().show()
    # y = []
    start = timeit.default_timer()
    ECG()
    # ECG_plot(solver(y))
    end = timeit.default_timer()
    print(f"time:{end - start}")