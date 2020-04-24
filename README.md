# PPNP and APPNP

<p align="center">
<img src="https://raw.githubusercontent.com/klicperajo/ppnp/master/ppnp_model.svg?sanitize=true" width="600">
</p>

TensorFlow and PyTorch implementations of the model proposed in the paper:

**[Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://www.kdd.in.tum.de/ppnp)**   
by Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann   
Published at ICLR 2019.

## Run the code
The easiest way to get started is by looking at the notebook `simple_example_tensorflow.ipynb` or `simple_example_pytorch.ipynb`. The notebook `reproduce_results.ipynb` shows how to reproduce the results from the paper.

## Requirements
The repository uses these packages:

```
numpy
scipy
tensorflow>=1.6,<2.0
pytorch>=1.5
```

You can install all requirements via `pip install -r requirements.txt`.
However, in practice you will only need either TensorFlow or PyTorch, depending on which implementation you use.
If you use the `networkx_to_sparsegraph` method for importing other datasets you will additionally need NetworkX.

## Installation
To install the package, run `python setup.py install`.

## Datasets
In the `data` folder you can find several datasets. If you want to use other (external) datasets, you can e.g. use the `networkx_to_sparsegraph` method in `ppnp.data.io` for converting NetworkX graphs to our SparseGraph format.

The Cora-ML graph was extracted by Aleksandar Bojchevski, and Stephan Günnemann. *"Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018,   
while the raw data was originally published by Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore. *"Automating the construction of internet portals with machine learning."* Information Retrieval, 3(2):127–163, 2000.

The Citeseer graph was originally published by Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad.
*"Collective Classification in Network Data."* AI Magazine, 29(3):93–106, 2008.

The PubMed graph was originally published by Galileo Namata, Ben London, Lise Getoor, and Bert Huang. *"Query-driven Active Surveying for Collective Classification"*.  International Workshop on Mining and Learning with Graphs (MLG) 2012.

The Microsoft Academic graph was originally published by Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, Stephan Günnemann. *"Pitfalls of Graph Neural Network Evaluation"*. Relational Representation Learning Workshop (R2L), NeurIPS 2018.

## Contact
Please contact klicpera@in.tum.de in case you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{klicpera_predict_2019,
	title = {Predict then Propagate: Graph Neural Networks meet Personalized PageRank},
	author = {Klicpera, Johannes and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
	booktitle={International Conference on Learning Representations (ICLR)},
	year = {2019}
}
```
