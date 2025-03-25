# Helper functions for computing fields
using PhysicalConstants.CODATA2022: ε_0, μ_0

const K_electric = 1 / (4π * ε_0).val # Coulomb's constant
const μ0 = μ_0.val  # Permeability of free space

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
function point_charge(x, y, z; q = 1.0e-9, p = Point3f(0, 0, 0))::Point3f
	r = Point3f(x, y, z)
	return point_charge(r; q = q, p = p)
end

function point_charge(r::Point3f; q = 1.0e-9, p = Point3f(0, 0, 0))::Point3
	rr = r - p

	r2 = dot(rr, rr)

	if isapprox(r2, 0.0, atol = 1e-2)
		return [0.0, 0.0, 0.0]  # Avoid singularity
	end
	r_norm = sqrt(r2)
	E_magnitude = q * K_electric / r_norm^3

	E = E_magnitude * rr

	return E
end

"""
	Compute the electric field at (x, y, z) or Point3f r in Cartesian coordinates 
	due to a set of point charges.

	Arguments:
		q  - Charges of the point charges (C)
		pos  - Positions of the point charges (m)
		either:
		x  - x-coordinate of observation point (m)
		y  - y-coordinate of observation point (m)
		z  - z-coordinate of observation point (m)
		or:
		r - Observation point in Cartesian coordinates (m)

	Returns:
		(Ex, Ey, Ez) - Electric field components in Cartesian coordinates (N/C)
"""
function point_charges(x, y, z;
	q = [-1.0e-9, 1.0e-9], pos = [Point3f(0, 0, -1.0), Point3f(0, 0, 1.0)])::Point3
	@assert length(q) == length(pos) "The number of charges and positions must be equal."

	r = Point3f(x, y, z)
	return point_charges(r; q = q, pos = pos)
end

function point_charges(r::Point3f;
	q = [-1.0e-9, 1.0e-9],
	pos = [Point3f(0, 0, -1.0), Point3f(0, 0, 1.0)])::Point3f
	@assert length(q) == length(pos) "The number of charges and positions must be equal."

	# Position vector of the field point
	E = Point3f(0.0, 0.0, 0.0)
	for (qi, pos_i) in zip(q, pos)
		E += point_charge(r; q = qi, p = pos_i)
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
function current_loop(x, y, z; I = 1.0, R = 1.0, z_pos = 0.0)::Point3f
	r = Point3f(x, y, z)
	return current_loop(r; I = I, R = R, z_pos = z_pos)
end

function current_loop(r::Point3f; I = 1.0, R = 1.0, z_pos = 0.0)::Point3f
	function integrand(θ)
		# Loop element position
		r_src = Point3f(R * cos(θ), R * sin(θ), z_pos)

		dl = Point3f(-sin(θ), cos(θ), 0.0) * R  # dl vector

		# Observation point in Cartesian coordinates

		# Compute r and |r|
		r_rel = r - r_src
		r_rel_mag = norm(r_rel)

		# Compute Biot-Savart contribution: (dl × r̂) / r²
		if isapprox(r_rel_mag, 0.0, atol = 1e-8)
			dB = Point3f(0.0, 0.0, 0.0)  # Avoid singularity
		else
			dB = cross(dl, r_rel) / r_rel_mag^3
		end
		return dB
	end

	# Perform numerical integration over θ from 0 to 2π
	B = μ0 * I / (4π) * quadgk(integrand, 0, 2π, rtol = 1e-6)[1]

	return B / 1e-7  # Convert from T to mili-Gauss
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
function current_loops(x, y, z; I = [1.0], R = [1.0], z_pos = [0.0])::Point3f
	@assert length(I) == length(R) == length(z_pos) "The number of currents, radii, and z-positions must be equal."

	r = Point3f(x, y, z)

	return current_loops(r; I = I, R = R, z_pos = z_pos)
end

function current_loops(r::Point3f; I = [1.0], R = [1.0], z_pos = [0.0])::Point3f
	@assert length(I) == length(R) == length(z_pos) "The number of currents, radii, and z-positions must be equal."

	B = Point3f(0.0, 0.0, 0.0)
	for (Ii, Ri, z_pos_i) in zip(I, R, z_pos)
		B += current_loop(r; I = Ii, R = Ri, z_pos = z_pos_i)
	end

	return B
end