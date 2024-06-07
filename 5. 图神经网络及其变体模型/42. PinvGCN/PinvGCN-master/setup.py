from distutils.core import setup

setup(name = 'pinvgcn',
    version = '1.0',
    description = 'Graph Convolutional Networks with fast Pseudoinverse filters',
    author = 'Dominik Alfke',
    author_email = 'dominik.alfke@mathematik.tu-chemnitz.de',
    url = 'https://github.com/dominikalfke/PinvGCN',
    packages = ['pinvgcn'],
    py_modules = ['pinvgcn.graphs', 'pinvgcn.hypergraphs', 'pinvgcn.pointclouds'])
