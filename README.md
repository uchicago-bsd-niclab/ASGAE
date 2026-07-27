# ASGAE: Adaptive Sampling Graph Autoencoder

Implementation of the **Adaptive Sampling Graph Autoencoder (ASGAE)**, introduced in our paper:

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
- PyTorch 2.2.1 with CUDA 11.8
- PyTorch3D 0.7.8
- PyTorch-Geometric 2.6.1
- Scikit-Learn 1.6.1
- CUDA-capable GPU (experiments used an NVIDIA RTX A4000, 16 GB)

The Docker image is built on top of `gwangjin/pytorch3d:torch2.2.1-cuda11.8`, which already provides PyTorch, CUDA and PyTorch3D. Remaining Python dependencies are listed in `requirements.txt` (FastAPI, Uvicorn, NumPy, Open3D, SimpleITK, scikit-learn, SciPy, tqdm, trimesh, OpenCV, VTK) and are installed automatically during the build.

---

## Installation
 
The recommended way to run ASGAE is through Docker, since it resolves the CUDA and PyTorch3D dependencies for you.
 
### Build the image
```bash
git clone https://github.com/uchicago-bsd-niclab/ASGAE.git
cd ASGAE
docker build -t asgae .
```
 
### Run the container
Start the API, exposing port 8000 and mounting the folder that contains your data into `/data` inside the container. Replace `/path/to/your/data` with the absolute path on your machine:

```bash
# Linux / macOS
docker run --gpus all -p 8000:8000 -v /path/to/your/data:/data asgae
```
```powershell
# Windows (PowerShell) - quote the path if it contains spaces
docker run --gpus all -p 8000:8000 -v "C:\path\to\your\data:/data" asgae
```
The `--gpus all` flag requires the NVIDIA Container Toolkit. The `-v` flag mounts your local data folder so the model can read the meshes; the container will not see files outside the mounted folder.
 
Once running, the API is available at `http://localhost:8000`. You can confirm the service and GPU access with:
```bash
curl http://localhost:8000/          # -> {"status":"ok"}
curl http://localhost:8000/gpu       # -> {"gpu_available": true}
```

---

## Data
 
The model expects each surface mesh to be stored as an individual **PyTorch Geometric graph** saved as a `.pt` file (loaded with `torch.load`). Each graph object must contain:
 
- `pos`: tensor of shape `[N, 3]` with the Euclidean coordinates of the `N` mesh vertices.
- `edge_index`: tensor of shape `[2, E]` with the mesh connectivity (edges).
- `x`: node feature tensor. The first three columns are the vertex normals and the remaining columns hold RGB texture (texture is normalized internally by its maximum).
All `.pt` files for a run must live in a single directory, which is passed to the API as `db_path`. The loader collects every matching `.pt` file in that folder and splits it into train/validation/test sets according to the requested percentages (with a fixed random seed for reproducibility).
 
**Datasets used in the paper**
- **COMA** (public): available at http://coma.is.tue.mpg.de/ after accepting its license.
- **3D photogrammetry dataset** of children with craniofacial pathology: cannot be shared publicly due to patient privacy and institutional (IRB) restrictions.
---


## Usage
 
With the container running, interact with the API through HTTP requests. All parameters are sent as a JSON body.
 
### Train
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
        "db_path": "/data",
        "batch_size": 4,
        "test_batch_size": 2,
        "k_order": 4,
        "embedding_dim": 512,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 300,
        "model_path": "/app/model/ASGAE_model.pt",
        "train_set_percentage": 0.8,
        "val_set_percentage": 0.1,
        "test_set_percentage": 0.1,
        "full_test_dataset": false,
        "verbose": false
      }'
```
Training uses the Adam (or SGD) optimizer and applies early stopping when the validation loss stops improving over five consecutive epochs. The best-performing weights are saved to `model_path`.
 
### Validate / run inference
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
        "db_path": "/data",
        "batch_size": 4,
        "test_batch_size": 2,
        "k_order": 4,
        "embedding_dim": 512,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 1,
        "model_path": "/app/model/ASGAE_model.pt",
        "train_set_percentage": 0.8,
        "val_set_percentage": 0.1,
        "test_set_percentage": 0.1,
        "full_test_dataset": true,
        "verbose": true
      }'
```
Set `full_test_dataset` to `true` to evaluate on the entire dataset; set it to `false` to evaluate only on the test split defined by the percentages above. A pre-trained model is provided at `model/ASGAE_model.pt`.
 
### Parameters
| Parameter | Description |
|-----------|-------------|
| `db_path` | Directory (inside the container) with the `.pt` mesh files. |
| `batch_size` / `test_batch_size` | Batch sizes for training and evaluation (minimum 2). |
| `k_order` | Order of the Chebyshev polynomials (paper uses 4). |
| `embedding_dim` | Latent dimensionality D_S (paper uses 512). |
| `optimizer` | `adam` or `sgd`. |
| `learning_rate` | Learning rate (paper uses 0.001). |
| `epochs` | Maximum number of training epochs. |
| `model_path` | Path where the trained weights are saved / loaded. |
| `train/val/test_set_percentage` | Dataset split fractions. |
| `full_test_dataset` | If true, use the whole dataset for testing. |
| `verbose` | Print per-batch progress. |
 
Key hyperparameters used in the paper: learning rate = 0.001, Chebyshev polynomial order k = 4, latent dimension D_S = 512, template update rate alpha = 0.9.
 
---
 
## API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check. Returns `{"status":"ok"}`. |
| `GET` | `/gpu` | Reports whether a CUDA GPU is available. |
| `POST` | `/train` | Trains ASGAE with the provided parameters. |
| `POST` | `/validate` | Runs inference/validation with a trained model. |
 
Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs` while the container is running.
 
---
 
## Repository Structure
```
.
├── Dockerfile
├── requirements.txt
├── README.md
├── LICENSE
├── model/
│   └── ASGAE_model.pt        # pre-trained weights
└── app/
    ├── main.py               # FastAPI app and API endpoints
    ├── utils.py              # training / testing orchestration
    ├── ASGAE.py              # ASGAE model definition
    ├── DataLoaders.py        # dataset and dataloader logic
    ├── custom_chamfer.py     # Chamfer distance loss
    ├── util.py               # geometry / mesh helper functions
    └── TestDocker.py         # minimal local training example (no API)
```
 
---
 
## Citation
If you use this code, please cite:
```bibtex
@article{CRUZGUERRERO2026104228,
  title   = {Unsupervised adaptive sampling graph autoencoder for 3D surface encoding and mesh representation transfer},
  journal = {Medical Image Analysis},
  year    = {2026},
  issn    = {1361-8415},
  doi     = {10.1016/j.media.2026.104228},
  url     = {https://www.sciencedirect.com/science/article/pii/S1361841526002975},
  author  = {Ines A. Cruz-Guerrero and Joseph Nagel and Antonio R. Porras},
  keywords = {Graph autoencoder, Graph embedding, Geometric learning, Mesh representation transfer}
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
