Features
========

The features in MushroomRL are 1-D arrays computed applying a specified function
to a raw input, e.g. polynomial features of the state of an MDP.
MushroomRL supports three types of features:

* basis functions;
* tensor basis functions;
* tiles.

The tensor basis functions are a PyTorch implementation of the standard
basis functions. They are less straightforward than the standard ones, but they
are faster to compute as they can exploit parallel computing, e.g. GPU-acceleration
and multi-core systems.

All the types of features are exposed by a single factory method ``Features``
that builds the one requested by the user.

.. automodule:: mushroom_rl.features.features
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

The factory method returns a class that extends the abstract class
``FeatureImplementation``.

.. automodule:: mushroom_rl.features._implementations.features_implementation
    :members:
    :private-members:
    :inherited-members:
    :show-inheritance:

The documentation for every feature type can be found here:

.. toctree::

    features/mushroom_rl.features.basis
    features/mushroom_rl.features.tensors
    features/mushroom_rl.features.tiles
