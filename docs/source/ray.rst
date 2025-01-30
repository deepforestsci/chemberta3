Using Ray with Chemberta3
=========================

Ray is a distributed computing framework for scaling AI and python applications. To this end, Ray 
provides utilities to set up a distributed cluster and run a task on the cluster. A ray training task 
takes a python script, containing a path to the dataset, model definition, definition of loss function 
and other training configuration. The script also contains a training function, which performs the 
actual training of the model (containing the forward pass, loss computation and others). The training 
function is replicated on multiple worker nodes to perform distributed training.

Pre-training Chemberta/MoLFormer models locally
-----------------------------------------------

1. Installation 

Create a new conda environment

.. code-block:: bash

    conda create -n training-env python=3.10.13 -y

Activate the environment

.. code-block:: bash

    conda activate training-env

Install PyTorch with CUDA 12.1

.. code-block:: bash

    conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

Install DeepChem

.. code-block:: bash

    pip install --pre deepchem

Install Ray

.. code-block:: bash

    pip install ray


2. Update the paramaters in `chemberta3/benchmarking/working_dir/config.json` file to submit a training job.

3. Submit the training job using:

.. code-block:: bash

    python3 main.py --config_path config.json
