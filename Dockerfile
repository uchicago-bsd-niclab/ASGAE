# Base image with PyTorch 2.2.1 and CUDA 11.8.
FROM gwangjin/pytorch3d:torch2.2.1-cuda11.8

# Set the working directory.
WORKDIR /app

# Copy project files.
COPY . /app

RUN apt-get update -y \
    && apt-get install -y \
        make \
	cmake \
	gcc \
	g++ \
	libopenblas-dev \
	libomp-dev \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
#        libncurses5-dev \
#        libncursesw5-dev \
#        xz-utils \
#        tk-dev \
        libffi-dev \
        liblzma-dev \
        python3-openssl \
        git \
        libxml2-dev \
        libxmlsec1-dev \
	libgl1 \
	libxrender1 \
	libxtst6 \
	libxi6 \
	libsm6 \
	libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install fvcore iopath

#RUN pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu118

RUN pip install torch_geometric
RUN pip install fvcore iopath ninja
#RUN pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu118.html

# Optional PyTorch3D installation alternative:
#RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

#RUN git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && pip install -e . --no-build-isolation

COPY . /app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
