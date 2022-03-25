FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

LABEL maintainer="gmarkall@nvidia.com"

RUN apt update && apt upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt install tzdata -y

RUN apt install curl git pkg-config build-essential make gcc g++ -y

RUN curl https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -L -o /root/minimamba.sh

RUN sh /root/minimamba.sh -b

ENV PATH=/root/mambaforge/bin:${PATH}

RUN /bin/bash -c "conda init"

RUN mamba create -y -n filigree numba jupyter libpng cmake glib ninja pytest giflib jbig lcms2 lerc libdeflate libtiff libwebp openjpeg

RUN cd /root && git clone https://github.com/gmarkall/numba-accelerated-udfs.git && echo 1

RUN /bin/bash -c 'eval "$(/root/mambaforge/bin/conda shell.bash hook)" && cd /root/numba-accelerated-udfs && conda activate filigree && ./build.sh'

EXPOSE 8888

CMD /bin/bash -c 'eval "$(/root/mambaforge/bin/conda shell.bash hook)" && conda activate filigree && jupyter notebook --ip 0.0.0.0 --allow-root --notebook-dir=/root/numba-accelerated-udfs/notebooks'
