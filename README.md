# CST Part II Project: Example Based Explanation in Machine Learning

Timothy Ye (Supervisors: Zohreh Shams, Umang Bhatt)

---

A project aiming to implement several approaches to example based explanation, for easy use with a TensorFlow 2 model.

*The project is still very much a work in progress~*

## Running

The `notebooks` folder will be updated with explained Jupyter notebook examples using this repository, and can act as a reference for how to use the library.

Currently, for both influence functions and RelatIF, we use the class `influence.influence_model.InfluenceModel`. An instance of this class represents a model in which we have upweighted a particular training point, and we can use it to retrieve influence values for given test points, as well as new model parameters which result for varying level of upweighting.

## Status

### Currently Implemented:

- Influence Functions (https://arxiv.org/abs/1703.04730)
- RelatIF (https://arxiv.org/abs/2003.11630)

### Future Goals:

- Representer Points (https://arxiv.org/abs/1811.09720)
- Fisher Kernels (https://arxiv.org/abs/1810.10118v1)
- Performance improvements.