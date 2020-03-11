using ImplicitDomainQuadrature, LinearAlgebra, SparseArrays
using BenchmarkTools, PyPlot
include("make_mesh.jl")
include("map_values.jl")


function velocity(∇u::Vector)
    return -speed/(dot(∇u,∇u))*∇u
end

function quadratic_initial_condition(x,y)
    return (x-center[1])^2 + (y-center[2])^2 - initial_radius_squared
end

function quadratic_maximum_velocity(sol,mesh::Mesh,precomputed_basis)
    mid_el_i = floor(Int,mesh.elmts_per_side/2)
    elID = cartesian_to_linear_index(mid_el_i,mid_el_i,mesh.elmts_per_side)
    element_sol = sol[mesh.connectivity[:,elID]]
    ∇u_min = precomputed_basis.grads[end]'*element_sol
    return velocity(∇u_min)
end

function quadratic_initial_condition(mesh::Mesh)
    nodes_per_side = length(mesh.xrange)
    nodes_in_mesh = nodes_per_side^2
    vals = zeros(nodes_in_mesh)
    count = 1
    for x in mesh.xrange
        for y in mesh.xrange
            vals[count] = quadratic_initial_condition(x,y)
            count += 1
        end
    end
    return vals
end

function streamline_stiffness_matrix(element_sol,precomputed_basis::PrecomputedBasis,quad,element_size)
    NF = precomputed_basis.NF
    stiffness_matrix = zeros(NF,NF)
    for idx in 1:length(quad)
        p,w = quad[idx]
        grads = precomputed_basis.grads[idx]
        vals = precomputed_basis.vals[idx]
        det_jac = precomputed_basis.determinants[idx]

        ∇u = grads'*element_sol
        direction = normalize(velocity(∇u))
        stiffness_matrix += (grads*direction)*vals'*det_jac*w
    end
    return 0.5*element_size*stiffness_matrix
end

function element_advection_matrix(element_sol,precomputed_basis::PrecomputedBasis,quad,dt)
    NF = precomputed_basis.NF
    advection_matrix = zeros(NF,NF)
    for idx in 1:length(quad)
        p,w = quad[idx]
        grads = precomputed_basis.grads[idx]
        vals = precomputed_basis.vals[idx]
        det_jac = precomputed_basis.determinants[idx]

        ∇u = grads'*element_sol
        v = velocity(∇u)

        advection_matrix += vals*(grads*v)'*det_jac*w
    end
    return -dt*advection_matrix
end

function streamline_diffusion_matrix(element_sol,precomputed_basis::PrecomputedBasis,quad,element_size,dt)
    NF = precomputed_basis.NF
    diffusion_matrix = zeros(NF,NF)
    for idx in 1:length(quad)
        p,w = quad[idx]
        grads = precomputed_basis.grads[idx]
        det_jac = precomputed_basis.determinants[idx]

        ∇u = grads'*element_sol
        v = velocity(∇u)
        dir = normalize(v)
        diffusion_matrix += (grads*dir)*(grads*v)'*det_jac*w
    end
    return -0.5*dt*element_size*diffusion_matrix
end

function supg_evaluate_rhs(coeffs,mass,stiffness,advection,diffusion)
    return (mass+stiffness+advection+diffusion)*coeffs
end

function standard_evaluate_rhs(coeffs,mass,advection)
    return (mass+advection)*coeffs
end

function supg_assemble(sol,mesh,mass_matrix,precomputed_basis,quad,time_step_size)
    matrix_row_indices = Int[]
    matrix_col_indices = Int[]
    matrix_vals = Float64[]

    vec_mass_matrix = vec(mass_matrix)

    rhs_indices = Int[]
    rhs_vals = Float64[]

    for ej in 1:mesh.elmts_per_side
        for ei in 1:mesh.elmts_per_side
            elID = cartesian_to_linear_index(ei,ej,mesh.elmts_per_side)
            element_node_indices = mesh.connectivity[1:precomputed_basis.NF,elID]
            element_sol = sol[element_node_indices]

            stiffness_matrix = streamline_stiffness_matrix(element_sol,precomputed_basis,quad,mesh.element_size)
            advection_matrix = element_advection_matrix(element_sol,precomputed_basis,quad,time_step_size)
            diffusion_matrix = streamline_diffusion_matrix(element_sol,precomputed_basis,quad,mesh.element_size,time_step_size)

            rhs = supg_evaluate_rhs(element_sol,mass_matrix,stiffness_matrix,advection_matrix,diffusion_matrix)
            append!(rhs_indices,element_node_indices)
            append!(rhs_vals,rhs)

            row_indices = repeat(element_node_indices,outer=(precomputed_basis.NF,1))
            col_indices = repeat(element_node_indices,inner=(precomputed_basis.NF,1))
            vals = vec_mass_matrix + vec(stiffness_matrix)
            append!(matrix_row_indices,row_indices)
            append!(matrix_col_indices,col_indices)
            append!(matrix_vals,vals)
        end
    end
    num_dof = (length(mesh.xrange))^2

    rhs = Array(sparsevec(rhs_indices,rhs_vals,num_dof))
    matrix = sparse(matrix_row_indices,matrix_col_indices,matrix_vals,num_dof,num_dof)
    return matrix,rhs
end

function run_steps_SUPG(sol0,mesh,quad_order,time_step_size,num_time_steps)

    poly_order = mesh.poly_order
    basis = TensorProductBasis(2,poly_order)
    quad = TensorProductQuadratureRule(2,quad_order)
    map = InterpolatingPolynomial(2,basis)
    coords = nodal_coords_of_elmt(mesh,1,1)
    update!(map,coords)
    precomputed_basis = PrecomputedBasis(basis,quad,map)

    mass_matrix = element_mass_matrix(precomputed_basis,quad)

    sol = copy(sol0)
    for t in 1:num_time_steps
        matrix,rhs = supg_assemble(sol,mesh,mass_matrix,precomputed_basis,quad,time_step_size)
        sol = matrix\rhs
    end
    return sol
end

function quadratic_initial_condition_error(sol,mesh,quad_order,final_time)
    poly_order = mesh.poly_order
    basis = TensorProductBasis(2,poly_order)
    quad = TensorProductQuadratureRule(2,quad_order)
    map = InterpolatingPolynomial(2,basis)
    coords = nodal_coords_of_elmt(mesh,1,1)
    update!(map,coords)
    precomputed_basis = PrecomputedBasis(basis,quad,map)
    err = 0.0
    for ej in 1:mesh.elmts_per_side
        for ei in 1:mesh.elmts_per_side
            elID = cartesian_to_linear_index(ei,ej,mesh.elmts_per_side)
            element_node_indices = mesh.connectivity[:,elID]
            element_sol = sol[element_node_indices]
            coords = nodal_coords_of_elmt(mesh,ei,ej)
            update!(map,coords)

            for idx in 1:length(quad)
                p,w = quad[idx]
                spatial = map(p)
                vals = precomputed_basis.vals[idx]
                det_jac = precomputed_basis.determinants[idx]

                analytical_solution = quadratic_initial_condition(spatial[1],spatial[2]) + speed*final_time
                numerical_solution = element_sol'*vals
                err += (analytical_solution - numerical_solution)*det_jac*w
            end
        end
    end
    return err
end

function standard_assemble(sol,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
    matrix_row_indices = Int[]
    matrix_col_indices = Int[]
    matrix_vals = Float64[]

    vec_mass_matrix = vec(mass_matrix)

    rhs_indices = Int[]
    rhs_vals = Float64[]

    for ej in 1:mesh.elmts_per_side
        for ei in 1:mesh.elmts_per_side
            elID = cartesian_to_linear_index(ei,ej,mesh.elmts_per_side)
            coords = nodal_coords_of_elmt(mesh,ei,ej)
            update!(map,coords)
            # stiffness_matrix = streamline_stiffness_matrix(precomputed_basis,map,quad,mesh.element_size)
            advection_matrix = element_advection_matrix(precomputed_basis,map,quad,time_step_size)
            # diffusion_matrix = streamline_diffusion_matrix(precomputed_basis,map,quad,mesh.element_size,time_step_size)
            element_node_indices = mesh.connectivity[1:precomputed_basis.NF,elID]
            element_node_vals = sol[element_node_indices]
            rhs = standard_evaluate_rhs(element_node_vals,mass_matrix,advection_matrix)
            append!(rhs_indices,element_node_indices)
            append!(rhs_vals,rhs)

            row_indices = repeat(element_node_indices,outer=(precomputed_basis.NF,1))
            col_indices = repeat(element_node_indices,inner=(precomputed_basis.NF,1))
            vals = vec_mass_matrix
            append!(matrix_row_indices,row_indices)
            append!(matrix_col_indices,col_indices)
            append!(matrix_vals,vals)
        end
    end
    num_dof = (length(mesh.xrange))^2

    rhs = Array(sparsevec(rhs_indices,rhs_vals,num_dof))
    matrix = sparse(matrix_row_indices,matrix_col_indices,matrix_vals,num_dof,num_dof)
    return matrix,rhs
end

function plot_contourf(rho,xrange;cmap="inferno",filename="",crange=range(minimum(rho),stop=maximum(rho),length=10))
    N = length(xrange)
    xxs = reshape([x for x in xrange for y in xrange], N, N)
    yys = reshape([y for x in xrange for y in xrange], N, N)
    fig, ax = PyPlot.subplots()
    cbar = ax.contourf(xxs,yys,reshape(rho,N,N),crange,cmap=cmap)
    fig.colorbar(cbar)
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

function make_grid(xrange)
    N = length(xrange)
    xxs = reshape([x for x in xrange for y in xrange], N, N)
    yys = reshape([y for x in xrange for y in xrange], N, N)
    return xxs,yys
end

function plot_contour(rho,xrange;filename="",crange=range(minimum(rho),stop=maximum(rho),length=10))
    N = length(xrange)
    xxs, yys = make_grid(xrange)
    fig, ax = PyPlot.subplots()
    cbar = ax.contour(xxs,yys,reshape(rho,N,N),crange)
    fig.colorbar(cbar)
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

function error_convergence(element_sizes,poly_order,quad_order,error_quad_order,final_time;CFL=0.5)
    errors = zeros(length(element_sizes))
    for (idx,element_size) in enumerate(element_sizes)
        mesh = generate_mesh(width,poly_order,element_size)
        time_step_size = abs(CFL*element_size/speed)
        num_time_steps = final_time/time_step_size
        @assert isinteger(num_time_steps)
        num_time_steps = round(Int,num_time_steps)

        sol0 = quadratic_initial_condition(mesh)
        sol = run_steps_SUPG(sol0,mesh,quad_order,time_step_size,num_time_steps)
        err = quadratic_initial_condition_error(sol,mesh,error_quad_order,final_time)
        errors[idx] = err
    end
    return errors
end

const width = 1.0
const poly_order = 1
const quad_order = 4
const initial_radius = 0.3
const initial_radius_squared = initial_radius^2
const center = [0.5,0.5]
const speed = -0.01
const alpha = 0.5
element_size = 0.25
final_time = 10.0
time_step_size = abs(alpha*element_size/speed)
num_time_steps = ceil(Int,final_time/time_step_size)


mesh = generate_mesh(width,poly_order,element_size)
sol0 = quadratic_initial_condition(mesh)

sol = run_steps_SUPG(sol0,mesh,quad_order,time_step_size,num_time_steps)
err = quadratic_initial_condition_error(sol,mesh,4,final_time)

# errors = error_convergence([0.5,0.25,0.1,0.05,0.01],1,4,4,5.0)
# analytical_solution = sol0 .+ speed*final_time
fig = plot_contour(sol,mesh.xrange)
