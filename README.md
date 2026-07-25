# ASGAE: Adaptive Sampling Graph Autoencoder

Official implementation of the **Adaptive Sampling Graph Autoencoder (ASGAE)**, introduced in our paper:

> **Unsupervised Adaptive Sampling Graph Autoencoder for 3D Surface Encoding and Mesh Representation Transfer**
> Ines A. Cruz-Guerrero, Joseph Nagel, Antonio R. Porras.
> *Medical Image Analysis*, 80, 40-49, 2026. [DOI](https://doi.org/10.1016/j.media.2026.104228)

ASGAE is an unsupervised geometric learning framework that decouples surface geometry from mesh topology, enabling standardized, interpretable and topology-agnostic latent surface representations. It supports mesh encoding, reconstruction, and mesh representation transfer across surfaces with variable pose, resolution, point distribution and connectivity.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Features
- Unsupervised training with no need for manual annotations or templates.
- Fuzzy Segmentation module that maps heterogeneous meshes to a geometrically consistent latent space of uniform dimensionality.
- Pose Standardization for pose-invariant encodings.
- Adaptive Sampling decoder that reconstructs surfaces using the topology of any reference mesh (mesh representation transfer).
- Support for downstream tasks such as landmark identification and pathology classification.

---

## Requirements
The model was developed and tested with:
- Python 3.9.23
- PyTorch 2.2.1
- PyTorch3D 0.7.8
- PyTorch-Geometric 2.6.1
- Scikit-Learn 1.6.1
- CUDA-capable GPU (experiments used an NVIDIA RTX A4000, 16 GB)

All dependencies are pinned in `requirements.txt`.

---

## Installation

### Option A — Docker (recommended)
Build the image:
```bash
docker build -t asgae .
```

Run the container, mounting the folder that contains your data. Replace `/path/to/your/data` with the absolute path on your machine:
```bash
# Linux / macOS
docker run --gpus all -p 8000:8000 -v /path/to/your/data:/data asgae
```
```powershell
# Windows (PowerShell) - quote the path if it contains spaces
docker run --gpus all -p 8000:8000 -v "C:\path\to\your\data:/data" asgae
```
The `-v` flag mounts your local data folder into `/data` inside the container so the model can access it. The container will not see files outside the mounted folder.

### Option B — Local install
```bash
git clone https://github.com/uchicago-bsd-niclab/ASGAE.git
cd <repo>
conda create -n asgae python=3.9
conda activate asgae
pip install -r requirements.txt
```

---

## Data
- **Input format:** surface meshes represented as connected graphs, where each node carries a feature vector with its Euclidean coordinates and, optionally, RGB texture. <!-- Confirm accepted file extensions, e.g. .obj / .ply -->
- **Public dataset:** experiments used the public COMA dataset (Ranjan et al., 2018), available at http://coma.is.tue.mpg.de/ after accepting its license.
- **Clinical dataset:** the 3D photogrammetry dataset of children with craniofacial pathology cannot be shared publicly due to patient privacy and institutional (IRB) restrictions.

---

## Usage
<!-- TODO -->

Key hyperparameters used in the paper: learning rate = 0.001, Chebyshev polynomial order k = 4, latent dimension D_S = 512, template update rate alpha = 0.9.

---

## Repository Structure
```
.
├── Dockerfile
├── requirements.txt
├── README.md
├── models/
    ├── ASGAE_model.pt
└── app/
    ├── ASGAE.py
    ├── Dataloaders.py
    ├── TestDocker.py
    ├── main.py
    ├── util.py
    ├── utils.py
    └── custom_chamfer.py
```

---

## Citation
If you use this code, please cite:
```bibtex
@article{CRUZGUERRERO2026104228,
title = {Unsupervised adaptive sampling graph autoencoder for 3D surface encoding and mesh representation transfer},
journal = {Medical Image Analysis},
pages = {104228},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104228},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526002975},
author = {Ines A. Cruz-Guerrero and Joseph Nagel and Antonio R. Porras},
keywords = {Graph autoencoder, Graph embedding, Geometric learning, Mesh representation transfer},
abstract = {Standardizing anatomical surface representations across heterogeneous datasets is critical for population-level modeling and downstream medical image analysis tasks. Although existing mesh autoencoders can achieve standardized latent surface representations, most produce uninterpretable encodings and rely on spatial correspondences, pose assumptions, pre-processing or simple dimensionality reduction operations such as pooling that compromise accuracy, reproducibility and generalizability. We present the Adaptive Sampling Graph Autoencoder (ASGAE), an unsupervised geometric learning framework that decouples surface geometry from mesh topology to enable standardized surface representations. ASGAE integrates a novel Fuzzy Segmentation module that learns probabilistic mappings between input surfaces with variable pose, resolution, point distribution and connectivity, and a geometrically consistent latent subspace with uniform dimensionality identified in a fully unsupervised manner from heterogeneous training data, enabling interpretable and pose-standardized encodings. A novel Adaptive Sampling Decoder reconstructs surfaces from their latent encodings using either their original mesh topology or the topology from any other reference mesh, thus enabling mesh representation transfer. ASGAE achieved state-of-the-art encoding and reconstruction performance in both a public synthetic dataset and a 3D photogrammetry dataset of children with craniofacial pathology, providing point-to-surface coordinate and texture reconstruction errors of 0.43 ± 0.07 mm and 4.61 ± 0.87%, respectively. It also enabled mesh representation transfer between heterogeneous surfaces with a point-to-surface error of 0.44 ± 0.08 mm (p = 0.001). Finally, we demonstrated the use of ASGAE’s capabilities in two common downstream analysis tasks: landmark identification and pathology classification. Unlike existing methods, ASGAE produces interpretable surface encodings and enables mesh standardization, which is key in many medical image analysis tasks.}
}
```

---

## License
This project is released under the terms in the [LICENSE](LICENSE) file.

---

## Acknowledgments
This work was supported by the National Institute of Dental and Craniofacial Research of the National Institutes of Health under Award Number R01DE032681. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

---

## Contact
For questions, please contact Ines A. Cruz-Guerrero (Alejandro.cruz@bsd.uchicago.edu) or open an issue in this repository.
