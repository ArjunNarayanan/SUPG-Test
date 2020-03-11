function cartesian_to_linear_index(i,j,num_per_side)
    return (j-1)*num_per_side + i
end

function element_start_node_index(Ei,Ej,poly_order,nodes_per_side)
    return poly_order*(Ej-1)*nodes_per_side + poly_order*(Ei -1) + 1
end

function node_indices_in_element(Ei,Ej,nodes_per_elmt_edge,nodes_per_side)
    nodes_in_element = nodes_per_elmt_edge^2
    indices = zeros(Int,nodes_in_element)
    start = element_start_node_index(Ei,Ej,nodes_per_elmt_edge-1,nodes_per_side)
    count = 1
    for col in 1:nodes_per_elmt_edge
        indices[count:count+nodes_per_elmt_edge-1] = start:(start+nodes_per_elmt_edge-1)
        start += nodes_per_side
        count += nodes_per_elmt_edge
    end
    return indices
end

function node_indices_in_element!(indices,Ei,Ej,nodes_per_elmt_edge,nodes_per_side)
    nodes_in_element = nodes_per_elmt_edge^2
    start = element_start_node_index(Ei,Ej,nodes_per_elmt_edge-1,nodes_per_side)
    count = 1
    for col in 1:nodes_per_elmt_edge
        indices[count:count+nodes_per_elmt_edge-1] = start:(start+nodes_per_elmt_edge-1)
        start += nodes_per_side
        count += nodes_per_elmt_edge
    end
end

function element_connectivity(elmts_per_side,nodes_per_elmt_edge)
    nodes_per_side = (nodes_per_elmt_edge-1)*elmts_per_side+1
    number_of_elmts = elmts_per_side^2
    nodes_per_elmt = nodes_per_elmt_edge^2
    indices = zeros(Int,nodes_per_elmt)
    connectivity = zeros(Int,nodes_per_elmt,number_of_elmts)
    idx = 1
    for Ej in 1:elmts_per_side
        for Ei in 1:elmts_per_side
            node_indices_in_element!(indices,Ei,Ej,nodes_per_elmt_edge,nodes_per_side)
            connectivity[1:nodes_per_elmt,idx] = indices
            idx += 1
        end
    end
    return connectivity
end

struct Mesh
    xrange::AbstractArray
    connectivity::Matrix{Int64}
    nodes_per_elmt_edge::Int64
    elmts_per_side::Int64
    element_size::Float64
    poly_order::Int64
end

function generate_mesh(width,poly_order,element_size)
    @assert isinteger(width/element_size)
    elmts_per_side = round(Int,width/element_size)
    nodes_per_elmt_edge = poly_order+1
    nodes_per_side = poly_order*elmts_per_side+1

    xrange = range(0.0,stop=width,length=nodes_per_side)
    connectivity = element_connectivity(elmts_per_side,nodes_per_elmt_edge)
    return Mesh(xrange,connectivity,nodes_per_elmt_edge,elmts_per_side,element_size,poly_order)
end

function nodal_coords_of_elmt(mesh::Mesh,Ei,Ej)
    poly_order = mesh.nodes_per_elmt_edge-1
    nodes_per_elmt = mesh.nodes_per_elmt_edge^2
    ni = cartesian_to_linear_index(1,Ei,poly_order)
    nj = cartesian_to_linear_index(1,Ej,poly_order)

    coords = zeros(2,nodes_per_elmt)
    count = 1
    for j in nj:(nj+mesh.nodes_per_elmt_edge-1)
        for i in ni:(ni+mesh.nodes_per_elmt_edge-1)
            coords[1,count] = mesh.xrange[j]
            coords[2,count] = mesh.xrange[i]
            count += 1
        end
    end
    return coords
end

function nodal_coords_of_elmt!(coords::Matrix,mesh::Mesh,Ei,Ej)
    poly_order = mesh.nodes_per_elmt_edge-1
    nodes_per_elmt = mesh.nodes_per_elmt_edge^2
    ni = cartesian_to_linear_index(1,Ei,poly_order)
    nj = cartesian_to_linear_index(1,Ej,poly_order)

    count = 1
    for j in nj:(nj+mesh.nodes_per_elmt_edge-1)
        for i in ni:(ni+mesh.nodes_per_elmt_edge-1)
            coords[1,count] = mesh.xrange[j]
            coords[2,count] = mesh.xrange[i]
            count += 1
        end
    end
end
