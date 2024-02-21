sudo apt-get update
sudo apt-get install python3-pip
pip3 install --pre deepchem
# plain torch is for CUDA 12.1 and dgs for cuda 12.1
pip3 install torch 
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
# for cpu only torch and dgl
# pip3 install torch --index-url https://download.pytorch.org/whl/cpu
# pip install  dgl -f https://data.dgl.ai/wheels/repo.html
pip3 install lightning transformers torch_geometric dgllife
pip3 install ray==2.9.1
