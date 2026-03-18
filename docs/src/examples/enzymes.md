```@meta
EditURL = "enzymes.jl"
```

# Predicting enzymes classes

````@example enzymes
using HyperdimensionalComputing
using MLDatasets

data = TUDataset("ENZYMES")
````

Split dataset in training and testing datasets

````@example enzymes
split = 0.9
train = rand(data.num_graphs) .< split
test = .! train
````

Encoding node features

Node features
- AA length
- 3-bin Waals
- 3-bin Hydro.
- 3-bin Polarity
- 3-bin Polariz.
- 3d length
- Total Waals
- Total Hydro.
- Total Polarity
- Total Polariz.

1. Generate random hypervectors that represent the feature names / keys

````@example enzymes
AA_LENGTH = TernaryHV()
BIN_WAALS = TernaryHV()
BIN_HYDRO = TernaryHV()
BIN_POLARITY = TernaryHV()
BIN_POLARIZATION = TernaryHV()
LENGTH_3D = TernaryHV()
TOTAL_WAALS = TernaryHV()
TOTAL_HYDRO = TernaryHV()
TOTAL_POLARITY = TernaryHV()
TOTAL_POLARIZATION = TernaryHV()

feat_vals = vcat([unique(g.node_data.features) for g in data.graphs]...) |> unique
feat_hvs = Dict(v => TernaryHV() for v in feat_vals)
````

2. Encode graphs

````@example enzymes
function encode(G::MLDatasets.Graph)
	nodes_hvs = [feat_hvs[n] for n in G.node_data.features]
	return graph(nodes_hvs[G.edge_index[1]], nodes_hvs[G.edge_index[2]]) #.v .|> sign |> TernaryHV
end
````

3. Train centroid prototype hypervectors

````@example enzymes
graph_hvs = map(encode, data.graphs)

classes = data.graph_data.targets
classes_hvs = begin
	train_idx = findall(train)
	train_classes = classes[train_idx]
	prototypes_hvs = Vector{TernaryHV}(undef, length(unique(train_classes)))
	for class in train_classes
		class_idx = train_classes .== class
		prototypes_hvs[class] = bundle(graph_hvs[train_idx[class_idx]])
	end
	prototypes_hvs
end
@show similarity(classes_hvs...)

test_classes = data.graph_data.targets[test_idx]
acc = 0
for idx in test_idx
	class_sim = [similarity(graph_hvs[idx], hv) for hv in classes_hvs]
	acc += data.graph_data.targets[idx] == argmax(class_sim)
end
@show acc / length(test_classes)
````

4. Retraining

````@example enzymes
for idx in findall(train)
	ytrue = classes[idx]
	ypred = [similarity(graph_hvs[idx], hv) for hv in classes_hvs] |> argmax
	if ytrue != ypred
		classes_hvs[ypred] += graph_hvs[idx]
		classes_hvs[ytrue] += TernaryHV(graph_hvs[idx] * -1)
	end
end
@show similarity(classes_hvs...)

test_classes = data.graph_data.targets[test_idx]
acc = 0
for idx in test_idx
	class_sim = [similarity(graph_hvs[idx], hv) for hv in classes_hvs]
	acc += data.graph_data.targets[idx] == argmax(class_sim)
end
@show acc / length(test_classes)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

