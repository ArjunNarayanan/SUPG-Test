using PyPlot
include("map_values.jl")

function gaussian_initial_condition(x,y)
    return exp(-((x-center[1])^2 + (y-center[2])^2))
end

function initial_condition(mesh::Mesh,initial_condition_function)
    nodes_per_side = length(mesh.xrange)
    nodes_in_mesh = nodes_per_side^2
    vals = zeros(nodes_in_mesh)
    count = 1
    for x in mesh.xrange
        for y in mesh.xrange
            vals[count] = initial_condition_function(x,y)
            count += 1
        end
    end
    return vals
end

function make_grid(xrange)
    N = length(xrange)
    xxs = reshape([x for x in xrange for y in xrange], N, N)
    yys = reshape([y for x in xrange for y in xrange], N, N)
    return xxs,yys
end

mesh = generate_mesh(1.0,1,0.01)
sol = initial_condition(mesh,gaussian_initial_condition)
sol = reshape(sol,length(mesh.xrange),length(mesh.xrange))

final_sol = sol .- 10.0

xxs, yys = make_grid(mesh.xrange)

using3D()
fig = PyPlot.figure()
ax = fig.add_subplot(1, 1, 1, projection = "3d")
ax.plot_surface(xxs, yys, sol)
fig.savefig("initial_condition.png")
