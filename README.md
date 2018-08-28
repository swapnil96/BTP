# BTP
BTP related materials: Code, links, patches etc. Each folder contains a Readme.md file which describes what that folder contains

## Contents
* **Movidius**: Contains materials and tools needed to install/use it

* **Torch-Openface**: Contains links and tutorials about Openface and also how to use torch

* **Hack**: Contains bug details and links to install torch2caffe, although the installed version didn't worked

# torch2Caffe
[This](https://github.com/facebook/fb-caffe-exts) tool provided by facebook to convert pretrained torch models to caffe

## Issues
* The support for the repo ended 3 years ago and if used today many dependency errors are found
* Also not all the layers are supported, changes in torch or in caffe has rendered this module to be useless
* Miscellaneous errors caused by other libraries due to versions not matching. **Boost, Zstd, Mstch, Glog, Gflags**

## Dependencies
* [fblualib](https://github.com/facebookarchive/fblualib) A collection of Lua / Torch utilities. **fb.python** module provided is a bridge between Lua and Python, allowing seamless integration between the two (enabling, for example, using SciPy with Lua tensors almost as efficiently as with native numpy arrays; data between Lua tensors and the corresponding Python counterpart numpy.ndarray objects is shared, not copied). Requires Torch

* [folly](https://github.com/facebook/folly) (acronymed loosely after Facebook Open Source Library) is a library of C++14 components designed with practicality and efficiency in mind. Folly contains a variety of core library components used extensively at Facebook

* [wangle](https://github.com/facebook/wangle) Wangle is a library that makes it easy to build protocols, application clients, and application servers. Required by folly library

* [fbthrift](https://github.com/facebook/fbthrift) Thrift is a serialization and RPC framework for service communication. Required by fblualib library. Thrift has a set of protocols for serialization that may be used in different languages to serialize the generated structures created from the code generator

* [thpp](https://github.com/facebook/thpp) TH++ is a C++ tensor library, implemented as a wrapper around the TH library (the low-level tensor library in Torch)

# Alternatives
Many repos/modules are there in the internet although most of them are not maintained and are deprecated

* [This repo](https://github.com/ysh329/deep-learning-model-convertor) provides a guide to search for other repos/modules/softwares which can convert models across frameworks

* [FaceNet paper](https://github.com/davidsandberg/facenet) This repo provides tensorflow model of the facenet paper, although again training will be required

* [th2caffe](https://github.com/e-lab/th2caffe) Easy to install, doesn't support **SpatialCrossMapLRN** layer, need to see a workaround for that

* [More Links](https://discuss.pytorch.org/t/convert-import-torch-model-to-pytorch/37)

* [Deserialize torch-serialized objects from Python](https://github.com/bshillingford/python-torchfile) Needs to be checked

* [Opposite](https://github.com/fanq15/caffe_to_torch_to_pytorch) caffe to torch, although not sure will it work

* [Mocha](https://github.com/kuangliu/mocha) Doesn't support **nn.SpatialConvolutionMM** layer, either change this layer to **SpatialConvolution**: process mentioned [here](https://discuss.pytorch.org/t/load-lua-got-unknown-lua-class/1678) or enhance the code to support the said layer. Also there can be other unsupported layers present

* [pytorch](https://stackoverflow.com/questions/41861354/loading-torch7-trained-models-t7-in-pytorch) Can use this feature to convert torch models in pytorch to numpy array and then convert directly to caffe or use some other modules(needs to be checked). **Problem** it doesn't support **nn.SpatialConvolutionMM** layer, again alternative already mentioned above needs to be used
