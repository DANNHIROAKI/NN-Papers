# UORO

This repository provides a torch implementation of the UORO algorithm
described in https://arxiv.org/abs/1702.05043.

## Using UORO implementation

### Prerequisites

Torch must be installed for the implementation to work. Installation
instructions are provided here: http://torch.ch/docs/getting-started.html.

### A simple working example

The following command line
```
th train.lua
```
trains a GRU model on the anbn dataset presented in the paper using 
the UORO algorithm.
The command line
```
th train.lua -h
```
provides a detailled list of the additional command line arguments
that you can provide.

## Going further

The UORO implementation provided in the file uoro.lua provides a
general implementation of UORO that you freely use in all cases
covered by the framework detailled in the paper.
