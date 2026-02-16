### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 23e74848-0b52-11f1-8a00-0b81e48381ea
begin
	include("../simpleNN/neuron.jl")
	using .SimpleNN
end

# ╔═╡ fdcd1512-42e3-4bef-9cb8-f86648c1665e
function total_loss(neuron, data)
	s = 0.0
	for (x,t) in data
		y = SimpleNN.predict(neuron, x)
		s += 0.5 * (y - t)^2
	end
	return s
end

# ╔═╡ c5462a38-4b3a-476d-97ea-ab6b3c661c78
function show_predictions(neuron, data)
	for (x, t) in data
		y = SimpleNN.predict(neuron, x)
		println("x = $x target = $t  prediction =$(round(y, digits=4))")
	end
end

# ╔═╡ a585cb3a-417e-4e70-9741-fab354687c1e
begin
	# init values
	neuron = Neuron([0.5, 0.5], -0.5)
	learning_rate = 0.01
end

# ╔═╡ d1ee9f78-83f0-4b96-a515-d671c84ea3f3
begin
	and_table = [
		([0.0, 0.0], 0.0),
		([0.0, 1.0], 0.0),
		([1.0, 0.0], 0.0),
		([1.0, 1.0], 1.0)]
	n10 = train(neuron, and_table, learning_rate, 10000)
	@show n10
	n11 = train(n10, and_table, learning_rate, 10000)
	@show n11
	n12 = train(n11, and_table, learning_rate, 10000)
	@show n12
	@show total_loss(n12, and_table)
end

# ╔═╡ b6e2eba7-8a5e-40c2-83d1-dd7b7c627fd8
begin
	xor_table = [
		([0.0, 0.0], 0.0),
		([0.0, 1.0], 1.0),
		([1.0, 0.0], 1.0),
		([1.0, 1.0], 0.0)]
	n20 = train(neuron, xor_table, learning_rate, 10000)
	@show n20
	n21 = train(n20, xor_table, learning_rate, 10000)
	@show n21
	n22 = train(n21, xor_table, learning_rate, 10000)
	@show n22
	@show total_loss(n22, xor_table)
end

# ╔═╡ 4b41e9cb-a712-4b51-8bf0-b52cf4d7dd7c
begin
	show_predictions(n12, and_table)
	println("---")
	show_predictions(n22, xor_table)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "71853c6197a6a7f222db0f1978c7cb232b87c5ee"

[deps]
"""

# ╔═╡ Cell order:
# ╠═23e74848-0b52-11f1-8a00-0b81e48381ea
# ╠═fdcd1512-42e3-4bef-9cb8-f86648c1665e
# ╠═c5462a38-4b3a-476d-97ea-ab6b3c661c78
# ╠═a585cb3a-417e-4e70-9741-fab354687c1e
# ╠═d1ee9f78-83f0-4b96-a515-d671c84ea3f3
# ╠═b6e2eba7-8a5e-40c2-83d1-dd7b7c627fd8
# ╠═4b41e9cb-a712-4b51-8bf0-b52cf4d7dd7c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
