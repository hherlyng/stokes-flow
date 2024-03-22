import ufl
import time
import multiphenicsx.fem
import multiphenicsx.fem.petsc

import numpy         as np
import dolfinx       as dfx

from ufl      import grad, div, dot, inner, sym
from mpi4py   import MPI
from petsc4py import PETSc

# Mesh tags
SLIP   = 1
INLET  = 2
OUTLET = 3
STRESS = 4

# Use PETSc printer to enable flush in parallel
print = PETSc.Sys.Print

# Orthogonal projection operator, action of (1 - n x n)
Tangent = lambda v, n: v - dot(v, n)*n

# Symmetric gradient
eps = lambda u: sym(grad(u))


class FlowSolver:
    """ Solver for the Stokes equations using Taylor-Hood finite elements with
        a Lagrange multiplier for the normal stresses to impose tangential stresses.

    """

    def __init__(self, mesh: dfx.mesh.Mesh,
                       ft: dfx.mesh.MeshTags,
                       write_output: bool = False,
                       post_process: bool = False) -> None:
        """ Constructor.

        Parameters
        ----------
        mesh : dfx.mesh.Mesh
            Computational mesh.

        ft : dfx.mesh.MeshTags
            Facet tags used for boundary integrals.

        """

        self.mesh = mesh
        self.ft   = ft
        self.write_output = write_output
        self.pp  = post_process
        

    def SetBilinearForm(self, u    : ufl.TrialFunction, v   : ufl.TestFunction,
                              p    : ufl.TrialFunction, q   : ufl.TestFunction,
                              zeta : ufl.TrialFunction, eta : ufl.TestFunction):

        """ Create and set the bilinear form for the Stokes equations variational problem
            with slip boundary conditions (BCs).

            BCs:
                - Boundary with tag SLIP has full slip BC; u.n = 0 and tangential stresses = 0.
                - Boundary with tag STRESS has the slip BC; u.n = 0 with specified tangential stress.
                - Boundary with tag INLET has normal traction set to a given value, no tangential flow.
                - Boundary with tag OUTLET has normal traction set to zero, no tangential flow.
        
        """

        dx = self.dx # Cell integral measure
        ds = self.ds # Boundary facet integral measure
        n  = self.n  # Facet normal vector

        # Initialize form blocks
        a00 = a01 = a02 = 0
        a10 = 0
        a20 = 0

        a00 = 2*self.mu * inner(eps(u), eps(v)) * dx # Viscous momentum diffusion term
        a01 = - p * div(v) * dx # Pressure term
        a10 = - q * div(u) * dx # Continuity equation

        # Multiplier only on slip boundary
        ds_slip = ds(SLIP) + ds(STRESS) # The part of the boundary where we impose u.n=0
        a02 = - zeta * dot(v, n) * ds_slip # Multiplier trial function term
        a20 = - eta * dot(u, n) * ds_slip  # Multiplier test function term

        # Pressure boundary integrals to ensure no tangential flow
        a00 += - self.mu * inner(dot(grad(u).T, n), v) * (ds(INLET) + ds(OUTLET))        

        a = [[a00, a01, a02],
            [a10, None, None],
            [a20, None, None]]
        
        self.a_cpp = dfx.fem.form(a)


    def SetLinearForm(self, v: ufl.TestFunction, q: ufl.TestFunction, eta: ufl.TestFunction):
        """ Create and set the linear form for the Stokes equations variational problem
            with slip boundary conditions (BCs).

            BCs:
                - Boundary with tag SLIP has full slip bc; u.n = 0 and tangential stresses = 0.
                - Boundary with tag STRESS has the slip BC; u.n = 0 with specified tangential stress.
                - Boundary with tag INLET has normal traction set to a given value, no tangential flow.
                - Boundary with tag OUTLET has normal traction set to zero, no tangential flow.
        
        """

        dx = self.dx # Cell integral measure
        ds = self.ds # Boundary facet integral measure

        # Define stuff used in the variational form
        zero = dfx.fem.Constant(self.mesh, PETSc.ScalarType(0)) # Zero value function used to assemble zero blocks
        stress_vector = 2*ufl.as_vector((1, 1, 1))
        tangential_stress  = Tangent(stress_vector, self.n)
        traction_bc_inlet  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(1.0))
        traction_bc_outlet = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))

        # Set the normal traction on the INLET boundary
        # Not including the OUTLET boundary integral means we set it to zero
        L0  = dot(traction_bc_inlet*self.n, v) * ds(INLET) + dot(traction_bc_outlet*self.n, v) * ds(OUTLET)

        # Impose a tangential stress on the STRESS boundary
        L0 -= dot(tangential_stress, Tangent(v, self.n)) * ds(STRESS) 
        
        # Assemble zero blocks for the pressure and the multiplier
        L1 = inner(zero, q) * dx # Zero RHS for pressure test eq.
        L2 = inner(zero, eta) * ds # Zero RHS for multiplier test eq.

        L = [L0, L1, L2]

        self.L_cpp = dfx.fem.form(L)

    def solve_linear_system(self):
        """ Solve the assembled linear system that represents the Stokes equations. """

        tic = time.perf_counter()
        # Solve and ghost update solution vector
        self.solver.solve(self.b, self.x)
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        print(f"Solved linear system in {time.perf_counter()-tic:.2f} seconds.")

        # Store solutions
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(self.x, [self.V.dofmap, self.Q.dofmap, self.Z.dofmap], self.restriction) as u_p_wrapper:
            for u_p_wrapper_local, component in zip(u_p_wrapper, (self.u_h, self.p_h, self.z_h)):
                with component.vector.localForm() as component_local:
                    component_local[:] = u_p_wrapper_local
        
        if self.write_output:
            # Update output functions
            self.u_out.interpolate(self.u_h)
            self.p_out.interpolate(self.p_h)
            self.z_out.interpolate(self.z_h)
                
    def setup(self):
        """ Perform setup of finite elements and output files. """
        
        print("#----------- Performing setup -----------#")
        tic = time.perf_counter()

        # Mesh related stuff
        self.n  = ufl.FacetNormal(self.mesh) # Facet normal vector
        self.dx = ufl.Measure("dx", domain=self.mesh) # Cell integral measure
        self.ds = ufl.Measure("ds", self.mesh, subdomain_data=self.ft) # Facet integral measure

        # Fluid parameters
        nu  = dfx.fem.Constant(self.mesh, PETSc.ScalarType(1e-6))  # Kinematic viscosity [m^2 / s]
        rho = dfx.fem.Constant(self.mesh, PETSc.ScalarType(1e3))  # Density [kg / m^3]
        self.mu = dfx.fem.Constant(self.mesh, PETSc.ScalarType(nu.value * rho.value)) # Dynamic viscosity [kg / (m * s)]
        
        # Elements and functions spaces
        P2_vec = ufl.VectorElement("Lagrange", self.mesh.ufl_cell(), 2) # Quadratic Lagrange vector elements
        P2     = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 2) # Quadratic Lagrange elements
        P1     = ufl.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1) # Linear Lagrange elements
        self.V = V = dfx.fem.FunctionSpace(self.mesh, P2_vec)  # Velocity function space 
        self.Q = Q = dfx.fem.FunctionSpace(self.mesh, P1)  # Pressure function space 
        self.Z = Z = dfx.fem.FunctionSpace(self.mesh, P2)  # Multiplier space 

        #--------- Create multiphenicsx dofmap restrictions ---------#
        # Interior dofs for velocity and pressure, this is the volume of the mesh
        dofs_V = np.arange(0, V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
        dofs_Q = np.arange(0, Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts)
        
        # Get the dofs of the boundary where the Lagrange multiplier lives
        if type(SLIP)==int:
            # There is only one slip boundary facet tag
            gamma_facets = ft.find(SLIP) 
        elif type(SLIP)==tuple:
            # There are multipled slip boundary facet tags
            gamma_facets = np.concatenate(([ft.find(SLIP[i]) for i in range(len(SLIP))]))
        
        gamma_facets = np.concatenate((gamma_facets, ft.find(STRESS)))

        dofs_gamma   = dfx.fem.locate_dofs_topological(Z, self.mesh.topology.dim-1, gamma_facets)

        # Define restricitons
        V_restr = multiphenicsx.fem.DofMapRestriction(V.dofmap, dofs_V)
        Q_restr = multiphenicsx.fem.DofMapRestriction(Q.dofmap, dofs_Q)
        Z_restr = multiphenicsx.fem.DofMapRestriction(Z.dofmap, dofs_gamma)

        self.restriction = [V_restr, Q_restr, Z_restr]

        # Print size of index maps
        size_local  = V.dofmap.index_map.size_local  + Q.dofmap.index_map.size_local  + Z.dofmap.index_map.size_local
        size_global = V.dofmap.index_map.size_global + Q.dofmap.index_map.size_global + Z.dofmap.index_map.size_global
        num_ghosts  = V.dofmap.index_map.num_ghosts  + Q.dofmap.index_map.num_ghosts  + Z.dofmap.index_map.num_ghosts
        
        print(f"MPI rank: {self.mesh.comm.rank}", comm=MPI.COMM_SELF)
        print(f"Size of local index map: {size_local}", comm=MPI.COMM_SELF)
        print(f"Size of global index map: {size_global}", comm=MPI.COMM_SELF)
        print(f"Number of ghost nodes: {num_ghosts}", comm=MPI.COMM_SELF)

        # Trial and test functions
        (u, p, zeta) = (ufl.TrialFunction(V), ufl.TrialFunction(Q), ufl.TrialFunction(Z))
        (v, q, eta ) = (ufl.TestFunction (V), ufl.TestFunction (Q), ufl.TestFunction (Z))

        # Set up variational form
        self.SetBilinearForm(u, v, p, q, zeta, eta)
        self.SetLinearForm(v, q, eta)

        tic_assembly = time.perf_counter()
        # Assemble linear system matrix and right-hand side vector
        self.A = multiphenicsx.fem.petsc.assemble_matrix_block(self.a_cpp, bcs=[], restriction=(self.restriction, self.restriction))
        self.A.assemble()
        self.b = multiphenicsx.fem.petsc.assemble_vector_block(self.L_cpp, self.a_cpp, bcs=[], restriction=self.restriction)
        print(f"\nAssembly of linear system in {time.perf_counter()-tic_assembly:.2f} seconds.")     
        
        # Create solution vector
        self.x = multiphenicsx.fem.petsc.create_vector_block(self.L_cpp, restriction=self.restriction)

        #----------- Solver setup -----------#
        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)

        # Configure a direct solver
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.getPC().setFactorSolverType("mumps")

        #----------- Initialize output -----------#
        # Create functions for storing solutions and functions for writing output
        self.u_h   = dfx.fem.Function(V)
        self.u_out = dfx.fem.Function(V)
        self.p_h   = dfx.fem.Function(Q)
        self.p_out = dfx.fem.Function(Q)
        self.z_h   = dfx.fem.Function(Z)
        self.z_out = dfx.fem.Function(Z)

        self.u_out.name = "velocity"
        self.p_out.name = "pressure"
        self.z_out.name = "sigma"

        if self.write_output:
            # Create output file strings and output files
            self.out_dir = '../output/flow/test/'
            u_out_str = self.out_dir + 'velocity.bp'
            p_out_str = self.out_dir + 'pressure.bp'
            z_out_str = self.out_dir + 'zeta.bp'
            
            self.vtx_u = dfx.io.VTXWriter(self.mesh.comm, u_out_str, [self.u_out], engine="BP4")
            self.vtx_p = dfx.io.VTXWriter(self.mesh.comm, p_out_str, [self.p_out], engine="BP4")
            self.vtx_z = dfx.io.VTXWriter(self.mesh.comm, z_out_str, [self.z_out], engine="BP4")

        toc = time.perf_counter()
        print(f"Setup in {toc-tic:.2f} seconds.")

    def run(self):
        
        print("\n#----------- Solving -----------#")
        # Solve the Stokes equations
        self.solve_linear_system()

        # Write output
        if self.write_output:
            self.vtx_u.write(0)
            self.vtx_p.write(0)
            self.vtx_z.write(0)

        # Post process
        if self.pp: self.post_process()
        
    def post_process(self):
        comm = self.mesh.comm # MPI communicator

        # Calculate maximum value of the velocity magnitude
        u1 = self.u_h.sub(0).collapse().x.array
        u2 = self.u_h.sub(1).collapse().x.array
        u3 = self.u_h.sub(2).collapse().x.array
        u_h_mag = np.sqrt(u1**2 + u2**2 + u3**2)
        u_h_mag_max = u_h_mag.max()

        self.u_h_mag_max = comm.allreduce(u_h_mag_max, op=MPI.MAX)

        # Calculate mean pressure and subtract it from the calculated pressure 
        # to get a pressure with zero mean
        vol = comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1 * self.dx)), op=MPI.SUM)
        mean_p_h = comm.allreduce(1/vol * dfx.fem.assemble_scalar(dfx.fem.form(self.p_h * self.dx)), op=MPI.SUM)
        self.p_h.x.array[:] -= mean_p_h
        
        # Get maximum pressure
        p_h_max = self.p_h.vector.norm(norm_type=PETSc.NormType.NORM_INFINITY)
        self.p_h_max = comm.allreduce(p_h_max, op=MPI.MAX)

        # Calculate velocity and pressure L2 norms
        self.u_L2_norm_max = self.calc_L2_norm(u_h=self.u_h)
        self.p_L2_norm_max = self.calc_L2_norm(u_h=self.p_h)

        # Calculate divergence
        div_u_ufl = ufl.div(self.u_h)
        div_u_expr = dfx.fem.Expression(div_u_ufl, self.p_h.function_space.element.interpolation_points())
        div_u = dfx.fem.Function(self.p_h.function_space)
        div_u.interpolate(div_u_expr)

        if comm.rank == 0:
            print("\n#----------- Post process -----------#")
            print(f"Max velocity: {self.u_h_mag_max}")
            print(f"Max pressure: {self.p_h_max}")
            print(f"Max absolute value of divergence: {np.abs(div_u.x.array).max()}")
            print(f"L2 norm velocity: {self.u_L2_norm_max:.2e}")
    
    def calc_L2_norm(self, u_h: dfx.fem.Function) -> np.float_:

        """ Calculates the L2 norm (scaled with the mesh volume) of a finite element function u. """

        vol = dfx.fem.assemble_scalar(dfx.fem.form(1 * self.dx))
        vol = self.mesh.comm.allreduce(vol, op=MPI.SUM)
        u_L2 = dfx.fem.form(inner(u_h, u_h) * self.dx)
        u_L2_norm_local = 1/vol * dfx.fem.assemble_scalar(u_L2)
        u_L2_norm_global = self.mesh.comm.allreduce(u_L2_norm_local, op=MPI.SUM)

        return np.sqrt(u_L2_norm_global)

def create_cube_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
    """ Create a unit cube mesh with associated facet meshtags.

    Parameters
    ----------
    N_cells : int
        Number of computational cells in each coordinate direction.

    Returns
    -------
    tuple(mesh, ft) : tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags))
        The mesh and the facet tags (ft).
    """
    mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, N_cells, N_cells, N_cells,
                                        cell_type = dfx.mesh.CellType.tetrahedron,
                                        ghost_mode = dfx.mesh.GhostMode.shared_facet)

    def left(x):
        return np.isclose(x[0], 0.0)
    
    def right(x):
        return np.isclose(x[0], 1.0)

    def front(x):
        return np.isclose(x[1], 0.0)

    def back(x):
        return np.isclose(x[1], 1.0)

    def bottom(x):
        return np.isclose(x[2], 0.0)

    def top(x):
        return np.isclose(x[2], 1.0)

    # Facet tags
    bc_facet_indices, bc_facet_markers = [], []
    fdim = mesh.topology.dim - 1

    inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
    bc_facet_indices.append(inlet_BC_facets)
    bc_facet_markers.append(np.full_like(inlet_BC_facets, INLET))

    outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
    bc_facet_indices.append(outlet_BC_facets)
    bc_facet_markers.append(np.full_like(outlet_BC_facets, OUTLET))

    front_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, front)
    bc_facet_indices.append(front_BC_facets)
    bc_facet_markers.append(np.full_like(front_BC_facets, SLIP))

    back_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, back)
    bc_facet_indices.append(back_BC_facets)
    bc_facet_markers.append(np.full_like(back_BC_facets, SLIP))

    bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
    bc_facet_indices.append(bottom_BC_facets)
    bc_facet_markers.append(np.full_like(bottom_BC_facets, SLIP))

    top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
    bc_facet_indices.append(top_BC_facets)
    bc_facet_markers.append(np.full_like(top_BC_facets, STRESS))

    bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
    bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

    sorted_facets = np.argsort(bc_facet_indices)

    facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

    return mesh, facet_tags

if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator

    # Model options
    write_output = True
    post_process = True
    mesh_filename = "mesh.xdmf"
    
    # with dfx.io.XDMFFile(comm, mesh_filename, "r") as xdmf:
    #     mesh = xdmf.read_mesh()
    #     ft   = xdmf.read_meshtags(mesh, name="ft")
    mesh, ft = create_cube_mesh_with_tags(N_cells=8)
    mesh.geometry.x[:, 0] *= 5

    print(f"Total # of cells in mesh: {mesh.topology.index_map(3).size_global}")
    
    #-----Create flow solver object-----#
    solver = FlowSolver(mesh=mesh, ft=ft, write_output=write_output, post_process=post_process)
                     
    #-----Perform setup and run flow simulation-----#
    solver.setup()
    solver.run()

    import pyvista as pv
    pl = pv.Plotter()
    topology, cell_types, x = dfx.plot.vtk_mesh(solver.V)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    # from IPython import embed;embed()
    grid.point_data["u_vec"] = np.reshape(solver.u_h.x.array[:]*1e-8, (solver.V.dofmap.index_map.size_global, 3))
    grid.set_active_vectors("u_vec")
    
    # Set up plot window
    #glyphs = grid.glyph(scale="u_vec", orient=True)
    #pl.add_mesh(glyphs)
    #pl.view_xy(True)
    grid.arrows.plot()
    #pl.show()
