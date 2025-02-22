# SuperpixelGen-using-SLIC-MT
Superpixel generation for segementation and detection done through algorithms like Simple Linear Iterative Clustering(SLIC) is resource-heavy.
The aim of this repository is to write fast code in CUDA C++ that can make this algorithm run in real-time with execution time >24 fps.

The repo borrows the initial implementation of SLIC from this repo:

https://github.com/darshitajain/SLIC


The contributions of this repo are following:

- Write custom CUDA kernels to make compute intensive functions faster
- Embed the CUDA kernels in Python using PyCUDA
