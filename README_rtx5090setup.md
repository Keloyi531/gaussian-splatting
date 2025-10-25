# RTX 5090 Environment Configuration for 3D Gaussian Splatting

Original project by [GraphDECO INRIA](https://github.com/graphdeco-inria/gaussian-splatting).  
This document covers the environment setup and build notes for RTX 5090 users (CUDA 12.8, MSVC v143).

📄 **Full detailed guide:** [`docs/rtx5090_setup/readme.md`](docs/rtx5090_setup/README.md)

---

## Clone and Build Instructions

```bash
git clone --recursive https://github.com/Keloyi531/gaussian-splatting.git
cd gaussian-splatting

# If already cloned without --recursive:
git submodule update --init --recursive

# (Optional) lock to my verified branches
git -C submodules/diff-gaussian-rasterization checkout rtx5090-fixes
git -C submodules/simple-knn checkout rtx5090-fixes
