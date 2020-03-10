function map_gradient(grad::AbstractVector,jacobian::AbstractMatrix)
    return (jacobian')\grad
end

function map_gradient(grad::AbstractMatrix,jacobian::AbstractMatrix)
    mapped_grad = zeros(size(grad))
    jac = lu(jacobian')
    for i in 1:size(grad)[1]
        mapped_grad[i,:] = jac\grad[i,:]
    end
    return mapped_grad
end

struct PrecomputedBasis
    vals
    grads
    jacobians
    determinants
    NF
    NQ
    function PrecomputedBasis(basis::TensorProductBasis{D,T,NF},quad::TensorProductQuadratureRule{D,R,NQ,S},map) where {D,T,NF,NQ,R,S}
        vals = [basis(p) for (p,w) in quad]
        jacobians = [gradient(map,p) for (p,w) in quad]
        determinants = [det(j) for j in jacobians]
        reference_grad = [gradient(basis,p) for (p,w) in quad]
        grads = [map_gradient(reference_grad[i],jacobians[i]) for i in 1:length(reference_grad)]
        return new(vals,grads,jacobians,determinants,NF,NQ)
    end
end

function element_mass_matrix(precomputed_basis::PrecomputedBasis,quad)
    NF = precomputed_basis.NF
    mass = zeros(NF,NF)
    for (idx,wt) in enumerate(quad.weights)
        v = precomputed_basis.vals[idx]
        j = det(precomputed_basis.jacobians[idx])
        mass += v*v'*j*wt
    end
    return mass
end
