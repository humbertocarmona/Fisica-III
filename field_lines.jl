using ColorSchemes
using LinearAlgebra
using Makie
using QuadGK

# Function to compute closed field lines for any vector field
function compute_field_lines(
	field_function::Function,
	start_points::Vector{Point3f};
	max_steps = 360,
	step_size::Float32 = 0.05f0,
	α_curvature::Float32 = 1.0f0,
	loop_tolerance::Float32 = 0.01f0,
)

	# Initialize the field lines
	field_lines_set = Vector{Point3f}[]
	field_vetor_set = Vector{Point3f}[]

	k = 1
	for r0 in start_points
		field_line_fwd = [r0, r0]
		field_line_bwd = [r0, r0]

		r_fwd, r_bwd = r0, r0
		F0 = field_function(r0)

		field_vector_fwd = [F0, F0]
		field_vector_bwd = [F0, F0]

		closed_loop = false
		touches_ends = false
		i = 1
		while (i < max_steps) && !closed_loop && !touches_ends
			Field_fwd = field_function(r_fwd)
			Field_bwd = field_function(r_bwd)

			Field_fwd_norm = norm(Field_fwd)
			Field_bwd_norm = norm(Field_bwd)

			zero_norms = [
				isapprox(Field_fwd_norm, 0.0, atol = 1e-10)
				isapprox(Field_bwd_norm, 0.0, atol = 1e-10)]

			if ~any(zero_norms)
				κ_fwd = curvature(field_line_fwd, i - 1)
				κ_bwd = curvature(field_line_bwd, i - 1)

				# println("$i $κ_fwd $κ_bwd, $low_curvature")

				Field_fwd_hat = Field_fwd / Field_fwd_norm
				Field_bwd_hat = Field_bwd / Field_bwd_norm

				# Adaptive step size: Reduce step size in high-curvature areas

				δl_fwd = step_size / (1 + α_curvature * κ_fwd)
				δl_bwd = step_size / (1 + α_curvature * κ_bwd)

				# Move along the field line in both forward and backward directions
				r_fwd_new = r_fwd + Field_fwd_hat * δl_fwd
				r_bwd_new = r_bwd - Field_bwd_hat * δl_bwd

				push!(field_line_fwd, r_fwd_new)
				push!(field_line_bwd, r_bwd_new)
				push!(field_vector_fwd, Field_fwd)
				push!(field_vector_bwd, Field_bwd)
				i += 1

				# Check if the trajectory returns close to the start point
				err_fwd = norm(r_fwd_new - r0)
				err_bwd = norm(r_bwd_new - r0)
				if i > 50
					err = min(err_fwd, err_bwd)
					closed_loop = err < loop_tolerance
					if closed_loop
						push!(field_line_fwd, r0)
						push!(field_line_bwd, r0)
						push!(field_vector_fwd, F0)
						push!(field_vector_bwd, F0)
						println("$k closed loop  at $i")
					end
					touches_ends = norm(r_fwd_new - r_bwd_new) < loop_tolerance
					if touches_ends
						push!(field_line_fwd, r_bwd_new)
						push!(field_line_bwd, r_fwd_new)
						push!(field_vector_fwd, F0)
						push!(field_vector_bwd, F0)
						println("$k touched end at $i")
					end

					if i >= max_steps
						println("$k reached max steps at $max_steps")
					end
				end
				r_fwd, r_bwd = r_fwd_new, r_bwd_new
			end
		end

		k = k + 1

		# Merge forward and backward trajectories into a single closed loop
		field_line = vcat(reverse(field_line_bwd), field_line_fwd)
		field_vectors = vcat(reverse(field_vector_bwd), field_vector_fwd)

		push!(field_lines_set, field_line)
		push!(field_vetor_set, field_vectors)
	end

	return field_lines_set, field_vetor_set
end

# Function to plot computed field lines in 3D using GLMakie
function plot_field_lines(
	field_lines::Vector{Vector{Point3f}},
	field_vectors::Vector{Vector{Point3f}};
	ax = nothing,
	draw_arrows = false,
	arrow_idx = nothing)

	#
	#
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

	i = 1
	for (line, F) in zip(field_lines, field_vectors)
		x_line = [p[1] for p in line]
		y_line = [p[2] for p in line]
		z_line = [p[3] for p in line]
		i += 1

		x_vel = x_line[2:end] - x_line[1:end-1]
		y_vel = y_line[2:end] - y_line[1:end-1]
		z_vel = z_line[2:end] - z_line[1:end-1]

		x_vel = vcat(x_vel, x_vel[end])  # Repeat last column
		y_vel = vcat(y_vel, y_vel[end])  # Repeat last column
		z_vel = vcat(z_vel, z_vel[end])  # Repeat last column

		F_norm = Float64[norm(f) for f in F]
		F_norm = log10.(F_norm .+ 1e-6)
		F_norm = (F_norm .- minimum(F_norm)) / (maximum(F_norm) - minimum(F_norm))
		colors = get(ColorSchemes.turbo, range(0, stop = 1, length = size(F_norm, 1)))

		lines!(ax, x_line, y_line, z_line,
			color = F_norm,
			linewidth = 2.0,
			colormap = colors)

		if draw_arrows
			arrows_indexes = []

			if isnothing(arrow_idx) || arrow_idx == "min vz"
				idx = argmin(z_vel)
				idx = max(idx, 2)
				idx = min(idx, length(line) - 1)
				push!(arrows_indexes, idx)
			elseif arrow_idx == "end"
				idx = length(line) - 10
				push!(arrows_indexes, idx)
			elseif arrow_idx == "start"
				idx = 10
				push!(arrows_indexes, idx)
			elseif arrow_idx == "x=0"
				xpos = line_matrix[1, :]
				idx = findall(x -> isapprox(x, 0.0; atol = 0.01), xpos)
				idx = max.(idx, 2)
				idx = min.(idx, length(xpos) - 1)
				for i in idx
					push!(arrows_indexes, i)
				end
			elseif arrow_idx == "y=0"
				ypos = line_matrix[2, :]
				idx = findall(y -> isapprox(y, 0.0; atol = 0.01), ypos)
				idx = max.(idx, 2)
				idx = min.(idx, length(ypos) - 1)
				for i in idx
					push!(arrows_indexes, i)
				end
			elseif arrow_idx == "z=0"
				idx = findfirst(z -> isapprox(abs(z), 0.0; atol = 0.01), z_line)
				for i in [idx]
					if ~isnothing(i)
						i > 200
						i = max.(i, 2)
						i = min.(i, length(z_line) - 1)
						push!(arrows_indexes, i)
					end
				end
				idx = findall(z -> isapprox(abs(z), 0.0; atol = 0.01), z_line)
				for i in idx
					if ~isnothing(i) && ~(i in arrows_indexes) && i > 200
						i = max.(i, 2)
						i = min.(i, length(z_line) - 1)
						push!(arrows_indexes, i)
					end
				end
			end

			if length(arrows_indexes) > 0
				ps = []
				vs = []
				for n in arrows_indexes
					# p = line[n]
					p = Point3f(x_line[n], y_line[n], 0.0f0)
					v = Point3f(x_vel[n], y_vel[n], z_vel[n])
					push!(ps, Point3f(p))
					push!(vs, Point3f(v))
				end
				arrows!(ax, ps, vs,
					fxaa = true, # turn on anti-aliasing
					linewidth = 0.0,
					arrowsize = Vec3f(0.015, 0.015, 0.03),
					align = :center,
					color = F_norm[arrows_indexes],
					colormap = colors,
				)
			end
		end
	end

	if ret
		return fig, ax
	else
		return nothing
	end
end