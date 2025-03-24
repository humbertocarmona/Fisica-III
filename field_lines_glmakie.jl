using GLMakie, LinearAlgebra
using QuadGK
using LinearAlgebra
using ColorSchemes
using GeometryBasics
using Random
using PhysicalConstants.CODATA2022: ε_0, μ_0

const K_electric = 1 / (4π * ε_0).val # Coulomb's constant
const μ0 = μ_0.val  # Permeability of free space

# %%  --------------------------------------------------------------------------

# Function to generate random points on a sphere
function uniform_sphere(;
	N::Int = 10,
	R::Float64 = 1.0,
	center::Vector{Float64} = [0.0, 0.0, 0.0])
	points = fill(zeros(3), N)
	Random.seed!(1)
	for i in 1:N
		ϕ = 2π * rand()  # Azimuth angle
		z = 2 * rand() - 1  # Cosine-sampled z
		sinθ = sqrt(1 - z^2)  # Compute sin(θ)

		x = R * cos(ϕ) * sinθ
		y = R * sin(ϕ) * sinθ
		z = R * z

		points[i] = [x, y, z] .+ center
	end
	return points
end

using LinearAlgebra

function uniform_disk(N::Int; 
		R::Float64=1.0, 
		n_hat::Vector{Float64}= [0.0, 0.0, 1.0],
		center::Vector{Float64}=[0.0, 0.0, 0.0], 
		seed::Int=1)

	Random.seed!(seed)

    # Step 1: Create orthonormal basis with n̂ as the new z-axis
    z_axis = normalize(n_hat)
    # Pick arbitrary vector not parallel to n̂
    tmp = abs(z_axis[1]) < 0.99 ? Float64[1.0, 0.0, 0.0] : Float64[0.0, 1.0, 0.0]
    x_axis = normalize(cross(tmp, z_axis))
    y_axis = cross(z_axis, x_axis)

    # Rotation matrix: columns are the new axes
    rot = hcat(x_axis, y_axis, z_axis)

    # Step 2: Generate points
    
	r = rand(N)
	θ = 2π * rand(N)
	x = R * sqrt.(r) .* cos.(θ)
	y = R * sqrt.(r) .* sin.(θ)
	z = zeros(N)

	points = [rot*[x[i], y[i], z[i]] for i in 1:N]
	points = [p .+ center for p in points]	
	return points
end


# Function to compute the curvature of a line at a given point
function curvature(line_points, i)
	n = length(line_points)
	if i < 2
		return 0.0
	end
	@assert n > i

	p1, p2, p3 = line_points[i-1], line_points[i], line_points[i+1]

	# Compute velocity vectors (first derivative)
	v1 = p2 .- p1
	v2 = p3 .- p2
	norm_v1 = norm(v1)
	norm_v2 = norm(v2)
	# Avoid division by zero
	if isapprox(norm_v1, 0.0, atol=0.01) || isapprox(norm_v2, 0.0, atol=0.01)
		return 0.0
	end
	# Compute tangent vectors
	T1 = v1 / norm(v1)
	T2 = v2 / norm(v2)

	# Compute the change in tangent vector (numerical derivative)
	dT = T2 - T1

	# Compute step size (arc length approximation)
	ds = norm(v2)

	# Compute curvature κ = |dT/ds|
	κ = norm(dT) / ds

	return κ
end

# Function to compute closed field lines for any vector field
function compute_field_lines(
	field_function::Function,
	start_points;
	max_steps = 360,
	step_size = 0.005, α_curvature = 1.0,
	loop_tolerance = 0.01,
	limits = (-1000, 1000, -1000, 1000, -1000, 1000))

	# Initialize the field lines
	field_lines_all = Vector{Vector{Float64}}[]
	for r0 in start_points
		trajectory_forward = Vector{Float64}[r0]
		trajectory_backward = Vector{Float64}[r0]

		r_fwd, r_bwd = r0, r0

		i = 1
		err = 1.0
		off_limits = false

		while (i < max_steps) && (err > loop_tolerance) && !off_limits
			Field_fwd = field_function(r_fwd...)
			Field_bwd = field_function(r_bwd...)

			Field_fwd_norm, Field_bwd_norm = norm(Field_fwd), norm(Field_bwd)

			if Field_fwd_norm ≈ 0 || Field_bwd_norm ≈ 0
				break  # Avoid singularities
			end
			Field_fwd_hat = Field_fwd / Field_fwd_norm
			Field_bwd_hat = Field_bwd / Field_bwd_norm

			# Adaptive step size: Reduce step size in high-curvature areas
			κ_fwd = curvature(trajectory_forward, i - 1)
			κ_bwd = curvature(trajectory_backward, i - 1)

			δl_fwd = step_size / (1 + α_curvature * κ_fwd)
			δl_bwd = step_size / (1 + α_curvature * κ_bwd)

			# Move along the field line in both forward and backward directions
			r_fwd_new = r_fwd + Field_fwd_hat * δl_fwd
			r_bwd_new = r_bwd - Field_bwd_hat * δl_bwd

			off_limits =
				r_fwd_new[3] > limits[6] || r_fwd_new[3] < limits[5] ||
				r_fwd_new[2] > limits[4] || r_fwd_new[2] < limits[3] ||
				r_fwd_new[1] > limits[2] || r_fwd_new[1] < limits[1]

			push!(trajectory_forward, r_fwd_new)
			push!(trajectory_backward, r_bwd_new)

			# Check if the trajectory returns close to the start point
			err_fwd = norm(r_fwd_new - r0)
			err_bwd = norm(r_bwd_new - r0)
			if i > 50
				err = min(err_fwd, err_bwd)
				if err < loop_tolerance
					push!(trajectory_forward, r0)
					push!(trajectory_backward, r0)
				end
				if i >= max_steps
					println("Max steps reached $i")
				end
			end

			r_fwd, r_bwd = r_fwd_new, r_bwd_new

			i += 1
		end

		# Merge forward and backward trajectories into a single closed loop
		full_trajectory = vcat(reverse(trajectory_backward), trajectory_forward)

		push!(field_lines_all, full_trajectory)
	end

	return field_lines_all
end

# Function to plot computed field lines in 3D using GLMakie
function plot_field_lines(
	field_lines;
	loop_outline = nothing,
	ax = nothing,
	draw_arrows = false,
	arrow_idx = nothing)
	ret = false
	if isnothing(ax)
		fig = Figure()
		ax = Axis3(fig[1, 1], xlabel = "x", ylabel = "y", zlabel = "z",
			perspectiveness = 0.6,  # Adjusts perspective strength
			azimuth = -π / 2, elevation = 0.0,  # Sets default viewing angles
			aspect = :data, # Keeps aspect ratio correct
		)
		ret = true
	end

	for line in field_lines
		# Ensure the field line is in proper matrix format (3 × N)
		line_matrix = hcat(line...)  # Converts Vector{Vector{Float64}} to 3 × N matrix
		vel = line_matrix[:, 2:end] - line_matrix[:, 1:end-1]
		vel = hcat(vel, vel[:, end])  # Repeat last column
		vel_norm = norm.(eachcol(vel))  # Compute norm of each column
		vel_norm = max.(vel_norm, 1e-8)  # Avoid division by zero

		vel_norm = (vel_norm .- minimum(vel_norm)) ./ (maximum(vel_norm) - minimum(vel_norm))

		colors = get(ColorSchemes.turbo, range(0, stop = 1, length = size(vel_norm, 1) + 1))

		lines!(ax, line_matrix[1, :], line_matrix[2, :], line_matrix[3, :],
			color = vel_norm,
			linewidth = 2.0, colormap = :turbo)

		if draw_arrows
			if isnothing(arrow_idx) || arrow_idx == "min z"
				vel_z = vel[3, :]
				idx = argmin(vel_z)
				idx = max(idx, 2)
				idx = min(idx, length(line) - 1)
			elseif arrow_idx == "end"
				idx = length(line) - 10
			elseif arrow_idx == "x=0"
				xpos = line_matrix[1, :]
				idx = findfirst(x -> isapprox(x, 0.0; atol = 0.01), xpos)
				idx = max(idx, 2)
				idx = min(idx, length(xpos) - 1)
			elseif arrow_idx == "y=0"
				ypos = line_matrix[2, :]
				idx = findfirst(y -> isapprox(y, 0.0; atol = 0.01), ypos)
				idx = max(idx, 2)
				idx = min(idx, length(ypos) - 1)
			elseif arrow_idx == "z=0"
				zpos = line_matrix[3, :]
				idx = findfirst(z -> isapprox(z, 0.0; atol = 0.01), zpos)
				if isnothing(idx)
					idx = 2
				end
				idx = max(idx, 2)
				idx = min(idx, length(zpos) - 1)
			elseif isa(arrow_idx, Int)
				idx = arrow_idx
			end

			arrows_indexes = [idx]
			ps = []
			vs = []
			for n in arrows_indexes
				# p1 = line[n-1]
				# p2 = line[n+1]
				p = line[n]
				# v = p2 - p1
				v = vel[:, n]
				# norm_v = norm(v)
				# if norm_v > 0
				push!(ps, Point3f(p))
				push!(vs, Point3f(v))
				# end
			end
			arrows!(ax, ps, vs,
				fxaa = true, # turn on anti-aliasing
				linewidth = 0.0,
				arrowsize = Vec3f(0.02, 0.02, 0.035),
				align = :center,
				color = colors[idx],
			)
		end
	end

	if loop_outline !== nothing
		for loop in loop_outline
			lines!(
				ax,
				loop[:, 1],
				loop[:, 2],
				loop[:, 3],
				color = :black,
			)
		end
	end

	if ret
		return fig, ax
	else
		return nothing
	end
end

# %%  --------------------------------------------------------------------------
# Helper functions for computing fields

"""
	Compute the electric field at (x, y, z) in Cartesian coordinates 
	due to a point charge at position p with charge q.

	Arguments:
		q  - Charge of the point charge (C)
		p  - Position of the point charge (m)
		x  - x-coordinate of observation point (m)
		y  - y-coordinate of observation point (m)
		z  - z-coordinate of observation point (m)

	Returns:
		(Ex, Ey, Ez) - Electric field components in Cartesian coordinates (N/C)
"""
function point_charge(x, y, z; q = 1.0, p = [0, 0, 1])
	r = [x, y, z] - p

	r_norm = norm(r)
	if r_norm == 0
		return [0.0, 0.0, 0.0]  # Avoid singularity
	end

	E_magnitude = q * K_electric / r_norm^3

	E = E_magnitude * r

	return E
end

"""
	Compute the electric field at (x, y, z) in Cartesian coordinates 
	due to a set of point charges.

	Arguments:
		q  - Charges of the point charges (C)
		pos  - Positions of the point charges (m)
		x  - x-coordinate of observation point (m)
		y  - y-coordinate of observation point (m)
		z  - z-coordinate of observation point (m)

	Returns:
		(Ex, Ey, Ez) - Electric field components in Cartesian coordinates (N/C)
"""
function point_charges(x, y, z;
	q = [-1.0, 1.0], pos = [[0, 0, -1.0], [0, 0, 1.0]])
	@assert length(q) == length(pos) "The number of charges and positions must be equal."

	# Position vector of the field point
	r = [x, y, z]
	E = [0.0, 0.0, 0.0]
	for (qi, pos_i) in zip(q, pos)
		E += point_charge(x, y, z; q = qi, p = pos_i)
	end

	return E
end

"""
	Compute the magnetic field at (x, y, z) in Cartesian coordinates 
	due to a circular current loop of radius R carrying current I.

	Arguments:
		I  - Current in the loop (A)
		R  - Radius of the loop (m)
		x  - x-coordinate of observation point (m)
		y  - y-coordinate of observation point (m)
		z  - z-coordinate of observation point (m)

	Returns:
		(Bx, By, Bz) - Magnetic field components in Cartesian coordinates (T)
"""
function current_loop(x, y, z; I = 1.0, R = 1.0, z_pos = 0.0)::Vector{Float64}
	function integrand(θ)
		# Loop element position
		x′, y′, z′ = R * cos(θ), R * sin(θ), z_pos

		dl = R * [-sin(θ), cos(θ), 0.0]  # dl vector

		# Observation point in Cartesian coordinates
		r = [x, y, z]
		r_src = [x′, y′, z′]

		# Compute r and |r|
		r_vec = r - r_src
		r_mag = norm(r_vec)

		# Compute Biot-Savart contribution: (dl × r̂) / r²
		dB = cross(dl, r_vec) / r_mag^3
		return dB
	end

	# Perform numerical integration over θ from 0 to 2π
	B = μ0 * I / (4π) * quadgk(integrand, 0, 2π, rtol = 1e-6)[1]

	return B
end

"""
	Compute the magnetic field at (x, y, z) in Cartesian coordinates 
	due to a set of current loops.

	Arguments:
		I  - Currents in the loops (A)
		R  - Radii of the loops (m)
		x  - x-coordinate of observation point (m)
		y  - y-coordinate of observation point (m)
		z  - z-coordinate of observation point (m)

	Returns:
		(Bx, By, Bz) - Magnetic field components in Cartesian coordinates (T)
"""
function current_loops(x, y, z; I = [1.0], R = [1.0], z_pos = [0.0])::Vector{Float64}
	@assert length(I) == length(R) == length(z_pos) "The number of currents, radii, and z-positions must be equal."

	B = [0.0, 0.0, 0.0]
	for (Ii, Ri, z_pos_i) in zip(I, R, z_pos)
		B += current_loop(x, y, z; I = Ii, R = Ri, z_pos = z_pos_i)
	end

	return B
end

# %%  --------------------------------------------------------------------------
"""
	Plot the magnetic field lines of a circular current loop.

	Arguments:
		N  - Number of field lines to plot
		I  - Current in the loop (A)
		R  - Radius of the loop (m)
"""
function plot_current_loop(;
	N = 3,
	I = 1,
	R = 1,
	z_pos = [0.0, 0.1],
	arrow_idx = nothing,
	start_points = nothing,
	draw_arrows = true)

	if isa(I, Number)
		I = fill(I, length(z_pos))
	end
	if isa(R, Number)
		R = fill(R, length(z_pos))
	end

	if isnothing(start_points)
		start_points = uniform_disk(N; 
		R = 0.9*R[1], 
		n_hat = [0, 0.0, 1.0],
		center = [0.0, 0.0, z_pos[1]])
	end



	current_loops_(x, y, z) = current_loops(x, y, z, I = I, R = R, z_pos = z_pos)


	field_lines = compute_field_lines(
		current_loops_,
		start_points;
		max_steps = 2500,
		step_size = 0.001,
		loop_tolerance = 0.01,
	)

	
	α = range(0, 2π, length = 360)
	loop_outline = []
	for (r, z) in zip(R, z_pos)
		loop = zeros(length(α), 3)
		for (j, a) in enumerate(α)
			loop[j, :] = [r * cos(a), r * sin(a), z]
		end
		push!(loop_outline, loop)
	end


	fig, ax = plot_field_lines(
		field_lines,
		loop_outline = loop_outline,
		draw_arrows = draw_arrows,
		arrow_idx = "z=0",
	)


	display(fig)

	return fig, ax
end

"""
	Plot the electric field lines of point charges.

	Arguments:
		N  - Number of field lines to plot
		q  - Charges of the point charges (C)
		pos  - Positions of the point charges (m)
		r  - Radius of the point charges (m)
"""
function plot_point_charges(;
	N = 10,
	q = [1.0],
	pos = [[0.0, 0.0, 0.0]],
	r = 0.1,
	arrow_idx = nothing,
	start_points = nothing)
	if isa(q, Number)
		q = fill(q, length(pos))
	else
		@assert length(q) == length(pos) "The number of charges and positions must be equal."
	end

	cmap = ColorSchemes.turbo
	color = [get(cmap, i) for i in range(0, stop = 1, length = length(q) + 1)]
	field_lines_set = Vector{Vector{Vector{Float64}}}[]

	println("Charges: $q")
	println("Positions: $pos")

	point_charges_(x, y, z) = point_charges(x, y, z; q = q, pos = pos)

	if isnothing(start_points)
		start_points = uniform_sphere(N = N, R = r, center = pos[1])
	end

	# Define the field function for the point charges at positions pos 
	f_lines =
		compute_field_lines(
			point_charges_,
			start_points;
			max_steps = 5000,
			step_size = 0.01,
		)
	push!(field_lines_set, f_lines)

	fig, ax = plot_field_lines(field_lines_set[1],
		draw_arrows = true,
		arrow_idx = arrow_idx)

	for (p, c) in zip(pos, [:blue, :red])
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
		cs = FaceView(fill(c, N), [FT(i) for i in 1:N])

		# generate normals per face (this creates a FaceView as well)
		ns = face_normals(ps, fs)

		# Create mesh
		m = GeometryBasics.mesh(ps, fs, normal = ns, color = cs)

		mesh!(ax, m)
	end
	display(fig)
	return fig, ax
end

# %%  --------------------------------------------------------------------------
# # Example: Magnetic field lines of a circular current loop
z = range(-1, 1, length = 11) |> collect
z = [0.0]

start_points = [[0.9 * cos(a), 0.9 * sin(a), 0.0] for a in 0:2π/36:2π-2π/36]
start_points = vcat(start_points, 
	[[0.8 * cos(a), 0.8 * sin(a), 0.0] for a in 0:2π/36:2π-2π/36])
start_points = vcat(start_points, 
	[[0.7 * cos(a), 0.7 * sin(a), 0.0] for a in 0:2π/18:2π-2π/18])

x= LinRange(-0.9, 0.9, 19)	
y = zeros(19)
z = zeros(19)
start_points = [Point3f(x[i], y[i], z[i]) for i in 1:19]
fig, ax = plot_current_loop(N = 100, I = 1.0, R = 1.0, z_pos = z, draw_arrows = false,
start_points=start_points)
# limits!(ax, -1.5, 1.5, -1.5, 1.5, -2, 2)

# %%  --------------------------------------------------------------------------
# # Example: Electric field lines of point charges
pos = [
	[0.0, 0.0, -1.0],
	[0.0, 0.0, 1.0],
];
q = [-1.0, 1.0];

start_points = Vector{Float64}[
	[0.2 * cos(a), 0.0, 1 + 0.2 * sin(a)] for a in 0:2π/36:2π-2π/36
]

start_points = vcat(
	start_points,
	Vector{Float64}[
		[0.2 * cos(a), 0.0, -1 + 0.2 * sin(a)] for a in 0:2π/36:2π-2π/36
	],
)

fig, ax = plot_point_charges(N = 20, q = q, pos = pos, r = 0.05, arrow_idx = "z=0",
	start_points = start_points)

# limits!(ax, -1, 1, -1, 1, -2, 2)
