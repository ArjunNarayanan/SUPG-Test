using ImplicitDomainQuadrature, LinearAlgebra, SparseArrays
using BenchmarkTools, PyPlot
include("make_mesh.jl")
include("map_values.jl")

function velocity(x::AbstractVector)
    r = x - center
    return speed*r/norm(r)
end

function initial_distance(center,radius,mesh::Mesh)
    nodes_per_side = length(mesh.xrange)
    nodes_in_mesh = nodes_per_side^2
    distance = zeros(nodes_in_mesh)
    count = 1
    for x in mesh.xrange
        for y in mesh.xrange
            distance[count] = sqrt((x-center[1])^2 + (y-center[2])^2) - radius
            count += 1
        end
    end
    return distance
end

function streamline_stiffness_matrix(precomputed_basis::PrecomputedBasis,map,quad,element_size)
    NF = precomputed_basis.NF
    stiffness_matrix = zeros(NF,NF)
    for idx in 1:length(quad)
        p,w = quad[idx]
        spatial = map(p)
        direction = normalize(velocity(spatial))
        grads = precomputed_basis.grads[idx]
        vals = precomputed_basis.vals[idx]
        det_jac = precomputed_basis.determinants[idx]
        stiffness_matrix += (grads*direction)*vals'*det_jac*w
    end
    return 0.5*element_size*stiffness_matrix
end

function element_advection_matrix(precomputed_basis::PrecomputedBasis,map,quad,dt)
    NF = precomputed_basis.NF
    advection_matrix = zeros(NF,NF)
    for idx in 1:length(quad)
        p,w = quad[idx]
        spatial = map(p)
        v = velocity(spatial)
        grads = precomputed_basis.grads[idx]
        vals = precomputed_basis.vals[idx]
        det_jac = precomputed_basis.determinants[idx]
        advection_matrix += vals*(grads*v)'*det_jac*w
    end
    return -dt*advection_matrix
end

function streamline_diffusion_matrix(precomputed_basis::PrecomputedBasis,map,quad,element_size,dt)
    NF = precomputed_basis.NF
    diffusion_matrix = zeros(NF,NF)
    for idx in 1:length(quad)
        p,w = quad[idx]
        spatial = map(p)
        v = velocity(spatial)
        dir = normalize(v)
        grads = precomputed_basis.grads[idx]
        det_jac = precomputed_basis.determinants[idx]
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

function supg_assemble(sol,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
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
            stiffness_matrix = streamline_stiffness_matrix(precomputed_basis,map,quad,mesh.element_size)
            advection_matrix = element_advection_matrix(precomputed_basis,map,quad,time_step_size)
            diffusion_matrix = streamline_diffusion_matrix(precomputed_basis,map,quad,mesh.element_size,time_step_size)
            element_node_indices = mesh.connectivity[1:precomputed_basis.NF,elID]
            element_node_vals = sol[element_node_indices]
            rhs = supg_evaluate_rhs(element_node_vals,mass_matrix,stiffness_matrix,advection_matrix,diffusion_matrix)
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

function plot_contourf(rho,xrange;cmap="inferno",filename="",crange=range(-initial_radius,stop=1.0,length=10))
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

function plot_contour(rho,xrange;filename="",crange=range(-initial_radius,stop=1.0,length=10))
    N = length(xrange)
    xxs = reshape([x for x in xrange for y in xrange], N, N)
    yys = reshape([y for x in xrange for y in xrange], N, N)
    fig, ax = PyPlot.subplots()
    cbar = ax.contour(xxs,yys,reshape(rho,N,N),crange)
    fig.colorbar(cbar)
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

const width = 1.0
const poly_order = 2
const quad_order = 4
const initial_radius = 0.3
const center = [0.5,0.5]
const speed = 0.02
const alpha = 0.5
element_size = 0.1
time_step_size = alpha*element_size/speed

mesh = generate_mesh(width,poly_order,element_size)
distance = initial_distance(center,initial_radius,mesh)
# fig = plot_contour(distance,mesh.xrange)

basis = TensorProductBasis(2,poly_order)
interpolated_distance = InterpolatingPolynomial(1,basis)
quad = TensorProductQuadratureRule(2,quad_order)

map = InterpolatingPolynomial(2,basis)
coords = nodal_coords_of_elmt(mesh,1,1)
update!(map,coords)
precomputed_basis = PrecomputedBasis(basis,quad,map)
mass_matrix = element_mass_matrix(precomputed_basis,quad)

# matrix,rhs = supg_assemble(distance,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
# distance2 = matrix\rhs
# matrix,rhs = supg_assemble(distance2,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
# distance3 = matrix\rhs
# matrix,rhs = supg_assemble(distance3,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
# distance4 = matrix\rhs
# matrix,rhs = supg_assemble(distance4,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
# distance5 = matrix\rhs

matrix,rhs = standard_assemble(distance,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
distance2 = matrix\rhs
matrix,rhs = standard_assemble(distance2,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
distance3 = matrix\rhs
matrix,rhs = standard_assemble(distance3,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
distance4 = matrix\rhs
matrix,rhs = standard_assemble(distance4,mesh,mass_matrix,precomputed_basis,map,quad,time_step_size)
distance5 = matrix\rhs

plot_contour(distance5,mesh.xrange,crange=[-0.2,0.0,0.2])
