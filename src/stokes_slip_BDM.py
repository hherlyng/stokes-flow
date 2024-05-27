import ufl

import numpy   as np
import dolfinx as dfx

from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc, apply_lifting
from ufl import div, dot, inner, sym, grad, sqrt

# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n)
Tangent = lambda v, n: v - n*dot(v, n)

# Symmetric gradient
eps = lambda u: sym(grad(u))


def assemble_system(a_cpp: dfx.fem.form, L_cpp: dfx.fem.form, bcs: list[dfx.fem.dirichletbc]):
    A = assemble_matrix(a_cpp, bcs)
    A.assemble()

    b = assemble_vector(L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs=bcs)

    return A, b

# Stabilization terms for the variational form
def Stabilization(mesh: dfx.mesh.Mesh,
                u: ufl.TrialFunction,
                v: ufl.TestFunction,
                mu: dfx.fem.Constant,
                penalty: dfx.fem.Constant,
                consistent: bool=True):

    '''Displacement/Flux Stabilization from Krauss et al paper'''
    n, hA = ufl.FacetNormal(mesh), ufl.avg(ufl.CellDiameter(mesh))

    D  = lambda v: sym(grad(v)) # the symmetric gradient
    dS = Measure('dS', domain=mesh)

    if consistent:
        return (-inner(Avg(2*mu*D(u), n), Jump(Tangent(v, n)))*dS
                -inner(Avg(2*mu*D(v), n), Jump(Tangent(u, n)))*dS
                + 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS)

    # For preconditioning
    return 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS

def get_system(msh: dfx.mesh.Mesh, penalty_val: float, direct: bool):

    t = 0
    T = 1
    dt = 0.02
    num_timesteps = int(T / dt)
    k  = 1
    cell = msh.basix_cell()
    Velm = element('BDM', cell, k)
    Qelm = element('DG', cell, k-1)
    Welm = mixed_element([Velm, Qelm])
    W = dfx.fem.functionspace(msh, Welm)
    V, _ = W.sub(0).collapse()


    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Exact expressions and normals
    dx = ufl.Measure("dx", msh)
    tdim = msh.topology.dim
    num_facets = msh.topology.index_map(2).size_global
    bdry_facets = np.full(num_facets, 1, dtype=np.int32)
    right_bdry_facets = dfx.mesh.locate_entities_boundary(msh, tdim-1, lambda x: np.isclose(x[0], msh.geometry.x[:, 0].max()))
    bdry_facets[right_bdry_facets] = 2
    meshtags = dfx.mesh.meshtags(msh, tdim-1, np.arange(num_facets, dtype=np.int32), bdry_facets)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=meshtags)
    n  = ufl.FacetNormal(msh)
        
    # Bilinear form
    u_ = dfx.fem.Function(V)
    deltaT  = dfx.fem.Constant(msh, dfx.default_scalar_type(dt))
    penalty = dfx.fem.Constant(msh, dfx.default_scalar_type(penalty_val))
    mu  = dfx.fem.Constant(msh, dfx.default_scalar_type(1))
    rho = dfx.fem.Constant(msh, dfx.default_scalar_type(1))

    a  = rho/deltaT*dot(u, v) * dx
    a += 2*mu*inner(eps(u), eps(v)) * dx - p * div(v) * dx - q * div(u) * dx
    a += Stabilization(msh, u, v, mu, penalty=penalty)

    # Linear form
    f = dfx.fem.Function(V)
    class Acceleration():
        def __init__(self, A, omega):
            self.t = 0
            self.A = A
            self.omega = omega

        def __call__(self, x):
            return np.stack((np.ones (x.shape[1]) * self.A * np.sin(self.omega*self.t),
                             np.zeros(x.shape[1]),
                             np.zeros(x.shape[1])))
    f_expr = Acceleration(A=1, omega=np.pi)
    f.interpolate(f_expr)
    tau_0 = 1.15e-3*ufl.as_vector((1, 0, 1)) # Tangential stress on curved boundaries
    tangent_traction = lambda n: Tangent(tau_0, n)
    L  = rho/deltaT*dot(u_, v) * dx
    L += inner(rho*f, v) * dx
    L += inner(Tangent(v, n), tangent_traction(n)) * ds(1)

    # Impose impermeability strongly
    u_bc = dfx.fem.Function(V)
    msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    bdry_facets = dfx.mesh.exterior_facet_indices(msh.topology)
    dofs = dfx.fem.locate_dofs_topological((W.sub(0), V), msh.topology.dim-1, bdry_facets)
    bc   = dfx.fem.dirichletbc(u_bc, dofs, W.sub(0))

    bcs = [bc]

    # Assemble system
    a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
    A, b = assemble_system(a_cpp, L_cpp, bcs)

    # Create nullspace vector
    ns_vec = A.createVecLeft()
    Q, Q_dofs = W.sub(1).collapse()
    ns_vec.array[Q_dofs] = 1.0
    ns_vec.normalize()
    nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=msh.comm)
    assert(nullspace.test(A))
    A.setNullSpace(nullspace) if direct else A.setNearNullSpace(nullspace)
    nullspace.remove(b)

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    wh = dfx.fem.Function(W)
    uh_out = dfx.fem.Function(dfx.fem.FunctionSpace(mesh, ufl.VectorElement("DG", mesh.ufl_cell(), 1)))
    vtx_u = dfx.io.VTXWriter(comm=mesh.comm, filename="acceleration_velocity_cylinder.bp", output=[uh_out], engine="BP4")
    vtx_u.write(0)

    for _ in range(num_timesteps):
        t += dt
        print(f"Time t = {t}")
        f_expr.t = t
        f.interpolate(f_expr)
        A, b = assemble_system(a_cpp, L_cpp, bcs)
        A.setNullSpace(nullspace) if direct else A.setNearNullSpace(nullspace)
        nullspace.remove(b)

        ksp.solve(b, wh.vector)
        niters = ksp.getIterationNumber()
        rnorm  = ksp.getResidualNorm()
        wh.x.scatter_forward()
        uh, ph = wh.split()
        u_.x.array[:] = wh.sub(0).collapse().x.array.copy()
        
        print(ph.x.array.max())
        uh_out.interpolate(uh)
        vtx_u.write(t)

    vtx_u.close()

    return A, b, W


if __name__ == '__main__':

    import tabulate

    mu_value = 1.0
    penalty_value = 20.0
    direct = True

    history = []
    headers = ('hmin', 'dimW', '|eu|_1', '|eu|_div', '|div u|_0', '|ep|_0', 'niters', '|r|')

    for i in [0]:
        
        filename = f'../geometries/cylinder_{i}.xdmf'
        with dfx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
            mesh = xdmf.read_mesh()
        mesh.geometry.x[:] *= 1e3
        mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
        A, b, W = get_system(mesh, penalty_value, direct)
        
        wh = dfx.fem.Function(W)

        # Setup solver
        if direct:
            ksp = PETSc.KSP().create(MPI.COMM_WORLD)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
        
        else:
            ksp = PETSc.KSP().create(mesh.comm)
            B = 1
            ksp.setOperators(A, B)

            opts = PETSc.Options()
            opts.setValue('ksp_type', 'minres')
            opts.setValue('ksp_rtol', 1E-9)                
            #opts.setValue('ksp_view', None)
            #opts.setValue('ksp_monitor_true_residual', None)                
            #opts.setValue('ksp_converged_reason', None)
            opts.setValue('fieldsplit_0_ksp_type', 'preonly')
            opts.setValue('fieldsplit_0_pc_type', 'lu')
            opts.setValue('fieldsplit_1_ksp_type', 'preonly')
            opts.setValue('fieldsplit_1_pc_type', 'lu')           

            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
            _, V_dofs = W.sub(0).collapse()
            _, Q_dofs = W.sub(1).collapse()
            is_V = PETSc.IS().createGeneral(V_dofs)
            is_Q = PETSc.IS().createGeneral(Q_dofs)

            pc.setFieldSplitIS(('0', is_V), ('1', is_Q))
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) 

            ksp.setUp()
            pc.setFromOptions()
            ksp.setFromOptions()

        ksp.solve(b, wh.vector)
        niters = ksp.getIterationNumber()
        rnorm  = ksp.getResidualNorm()
        uh, ph = wh.split()

        # Post-process pressure
        dx  = ufl.Measure('dx', domain=mesh)
        vol = mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * dx)), op=MPI.SUM)
        mean_ph = mesh.comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(ph * dx)), op=MPI.SUM)
        ph.x.array[:] -= mean_ph
   

        div_uh = sqrt(abs(dfx.fem.assemble_scalar(dfx.fem.form(div(uh)**2*dx))))
        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_global
        cells = np.arange(num_cells, dtype=np.int32)
        h = dfx.cpp.mesh.h(mesh._cpp_object, tdim, cells)
        history.append((h.min(), W.dofmap.index_map.size_global, div_uh, niters, rnorm))
        print(tabulate.tabulate(history, headers=headers))

    
    ph_out = dfx.fem.Function(dfx.fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1)))
    ph_out.interpolate(ph)
    with dfx.io.XDMFFile(mesh.comm, "acceleration_pressure_cylinder.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(ph_out)

    