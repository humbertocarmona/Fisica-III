using GLMakie
using GeometryBasics

include("utils.jl")
include("fields.jl")
include("field_lines.jl")

# %%  --------------------------------------------------------------------------
"""
	Plot the magnetic field lines of a circular current loop.

	Arguments:
		N  - Number of field lines to plot
		I  - Current in the loop (A)
		R  - Radius of the loop (m)
"""
function field_lines_current_loops(;
	N = 3,
	I = 1,
	R = 1,
	z_pos = [0.0, 0.1],
	arrow_idx = nothing,
	start_points = nothing)
	if isa(I, Number)
		I = fill(I, length(z_pos))
	end
	if isa(R, Number)
		R = fill(R, length(z_pos))
	end

	if isnothing(start_points)
		start_points = uniform_disk(N;
			R = 0.9 * R[1],
			n_hat = [0, 0.0, 1.0],
			center = [0.0, 0.0, z_pos[1]])
	end

	# Define the field function for the current loops at positions z_pos
	current_loops_(r::Point3f) = current_loops(r::Point3f, I = I, R = R, z_pos = z_pos)

	# each element of field_lines_set is a field line (Vector{Point3f})
	field_lines_set, field_vectors = compute_field_lines(
		current_loops_,
		start_points;
		max_steps = 7000,
		step_size = 0.005f0,
		loop_tolerance = 0.05f0,
	)


	α = range(0, 2π, length = 360)
	loop_outline = Vector{Point3f}[]
	for (r, z) in zip(R, z_pos)
		loop = zeros(Point3f, length(α))
		for (j, a) in enumerate(α)
			loop[j] = Point3f(r * cos(a), r * sin(a), z)
		end
		push!(loop_outline, loop)
	end

	return field_lines_set, field_vectors, loop_outline
end

"""
	Plot the electric field lines of point charges.

	Arguments:
		N  - Number of field lines to plot
		q  - Charges of the point charges (C)
		pos  - Positions of the point charges (m)
		r  - Radius of the point charges (m)
"""
function field_lines_point_charges(;
	N::Int = 10,
	q::Union{Vector{Float64}, Number} = 1.0e-9,
	pos::Vector{Point3f} = [Point3f(0.0, 0.0, 0.0)],
	r = 0.1,
	start_points::Union{Vector{Point3f}, Nothing} = nothing)

	if isa(q, Number)
		q = fill(q, length(pos))
	else
		@assert length(q) == length(pos) "The number of charges and positions must be equal."
	end

	if isnothing(start_points)
		start_points = uniform_sphere(N = N, R = r, center = pos[1])
	end

	cmap = ColorSchemes.turbo
	color = [get(cmap, i) for i in range(0, stop = 1, length = length(q) + 1)]

	# Define the field function for the point charges at positions pos
	point_charges_(r::Point3f) = point_charges(r::Point3f; q = q, pos = pos)


	# each element of field_lines_set is a field line (Vector{Point3f})
	field_lines_set, field_vector_set =
		compute_field_lines(
			point_charges_,
			start_points;
			max_steps = 7000,
			step_size = 0.005f0,
			loop_tolerance = 0.05f0,
		)


	charges=[]
	for (p, qi) in zip(pos, q)
		# Plot the sphere surface
		# Reduce quality of sphere
		s = Tessellation(Sphere(Point3f(p), 0.1f0), 12)
		ps = coordinates(s)
		fs = faces(s)

		# Use a FaceView to with a new set of faces which refer to one color per face.
		# Each face must have the same length as the respective face in fs.
		# (Using the same face type guarantees this)
		FT = eltype(fs)
		N = length(fs)
		# cs = FaceView(rand(RGBf, N), [FT(i) for i in 1:N])
		qi > 0 ? c = RGBf(1, 0, 0) : c = RGBf(0, 0, 1)

		cs = FaceView(fill(c, N), [FT(i) for i in 1:N])

		# generate normals per face (this creates a FaceView as well)
		ns = face_normals(ps, fs)

		# Create mesh
		m = GeometryBasics.mesh(ps, fs, normal = ns, color = cs)
		push!(charges, m)
	end

	return field_lines_set, field_vector_set, charges
end

# %%  --------------------------------------------------------------------------
# # Example: Magnetic field lines of a circular current loop
z = range(-1.0, 1.0, length = 20) |> collect;
# z = [0.0];

start_points = [[0.9 * cos(a), 0.9 * sin(a), 0.0] for a in 0:2π/18:2π-2π/18];
start_points = vcat(start_points,
	[[0.7 * cos(a), 0.7 * sin(a), 0.0] for a in 0:2π/18:2π-2π/18]);
start_points = vcat(start_points,
	[[0.5 * cos(a), 0.5 * sin(a), 0.0] for a in 0:2π/18:2π-2π/18]);
start_points = vcat(start_points,
	[[0.3 * cos(a), 0.3 * sin(a), 0.0] for a in 0:2π/18:2π-2π/18]);

N = 19
xi = LinRange(-0.9, 0.9, N);
yi = zeros(N);
zi = zeros(N);
start_points = [Point3f(xi[i], yi[i], zi[i]) for i in 1:N];


lines_current_loops, B_field, loop_outline =
	field_lines_current_loops(N = 100, I = 1.0, R = 1.0, z_pos = z,
		start_points = start_points);
# %%  --------------------------------------------------------------------------
draw_arrows = true
arrow_idx = "z=0"
fig, ax = plot_field_lines(
	lines_current_loops, B_field;
	draw_arrows = draw_arrows,
	arrow_idx = arrow_idx
)

for loop in loop_outline
	lines!(ax, loop, color = :black, linewidth = 2.0)
end

limits!(ax, -1.5, 1.5, -1.5, 1.5, -2, 2)
display(fig)


# %%  --------------------------------------------------------------------------
# # Example: Electric field lines of point charges

pos = [
	Point3f(0.0, 0.0, -1.0),
	Point3f(0.0, 0.0, 1.0),
];
q = [1.0e-9, -1.0e-9];
arrow_idx = "z=0"

start_points_E = [
	Point3f(0.2 * cos(a), 0.0, 1 + 0.2 * sin(a)) for a in 0:2π/36:2π-2π/36
];

start_points_E = vcat(
	start_points_E,
	[
		Point3f(0.2 * cos(a), 0.0, -1 + 0.2 * sin(a)) for a in 0:2π/36:2π-2π/36
	],
);

lines_point_charges, E_field, charges = field_lines_point_charges(N = 20, q = q, pos = pos, r = 0.05,
	start_points = start_points_E);
# %%  --------------------------------------------------------------------------


fig, ax = plot_field_lines(lines_point_charges,E_field; draw_arrows = draw_arrows,
arrow_idx = arrow_idx)

for m in charges
	mesh!(ax, m)
end
limits!(ax, -1, 1, -1, 1, -2, 2)
display(fig)
