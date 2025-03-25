using Random
using LinearAlgebra

# Function to generate random points on a sphere
function uniform_sphere(N::Int = 10;
	R::Float64 = 1.0,
	center::Point3f = Point3f(0.0, 0.0, 0.0),
	seed::Int = 1)::Vector{Point3f}
	Random.seed!(seed)
	points = zeros(Point3d, N)
	for i in 1:N
		ϕ = 2π * rand()  # Azimuth angle
		z = 2 * rand() - 1  # Cosine-sampled z
		sinθ = sqrt(1 - z^2)  # Compute sin(θ)

		x = R * cos(ϕ) * sinθ
		y = R * sin(ϕ) * sinθ
		z = R * z

		points[i] = Point3f(x, y, z) .+ center
	end
	return points
end

# Function to generate random points on a disk
function uniform_disk(N::Int;
	R::Float64 = 1.0,
	n_hat::Point3f = Point3f(0.0, 0.0, 1.0),
	center::Point3f = Point3f(0.0, 0.0, 0.0),
	seed::Int = 1)::Vector{Point3f}
	#
	Random.seed!(seed)

	# Create orthonormal basis with n̂ as the new z-axis
	z_axis = normalize(n_hat)
	# Pick arbitrary vector not parallel to n̂
	tmp = abs(z_axis[1]) < 0.99 ? Point3f(1.0, 0.0, 0.0) : Point3f(0.0, 1.0, 0.0)
	x_axis = normalize(cross(tmp, z_axis))
	y_axis = cross(z_axis, x_axis)

	# create rotation matrix: columns are the new axes
	rot = hcat(x_axis, y_axis, z_axis)

	# Generate points
	r = rand(N)
	θ = 2π * rand(N)
	x = R * sqrt.(r) .* cos.(θ)
	y = R * sqrt.(r) .* sin.(θ)
	z = zeros(N)
	points = [Point3f(x[i], y[i], z[i]) for i in 1:N]

	# Rotate points
	points = [rot * p for p in points]

	# Translate points
	points = [p .+ center for p in points]
	return points
end

# Function to compute the curvature of a line at a given point
function curvature(line_points, i)::Float32
	n = length(line_points)
	if i < 2
		return 0.0
	end
	@assert n > i "found  $n > $i"

	p1, p2, p3 = line_points[i-1], line_points[i], line_points[i+1]

	# Compute velocity vectors (first derivative)
	v1 = p2 .- p1
	v2 = p3 .- p2
	dv = v2 - v1
	ds = (v2 + v1) / 2
	ds2 = dot(ds, ds)
	if isapprox(ds2, 0.0, atol = 1e-4)
		return 0.0
	end

	κ = norm(dv) / ds2

	# norm_v1 = norm(v1)
	# # Avoid division by zero
	# norm_v2 = norm(v2)
	# if isapprox(norm_v1, 0.0, atol = 1e-4) || isapprox(norm_v2, 0.0, atol = 1e-4)
	# 	return 0.0
	# end
	# Compute tangent unitary vectors
	# v1_hat = v1 / norm_v1
	# v2_hat = v2 / norm_v2
	# # Compute the change in tangent vector (numerical derivative)
	# dT = v2_hat - v1_hat
	# # Compute step size (arc length approximation)
	# ds = norm_v2
	# Compute curvature κ = |dT/ds|
	# κ = norm(dT) / ds

	κ = min(κ, 100.0)
	return Float32(κ)
end