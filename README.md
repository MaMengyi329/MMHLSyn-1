# MMHLSyn
This repository contains the code and data for "MMHLSyn:Synergistic prediction of drug combinations based on multi-view feature learning and multi-task supervision"
Initially,multi-view learning is independently applied to the multimodal data of drugs and cell lines.  
Within this framework, the fine-tuned ChemBERTa model, bolstered by contrastive learning,effectively captures the contextual information of the drug SMILES.  
Subsequently, enhanced hypergraph neural networks equipped with a multi-head attention mechanism are designed,capturing the complex topological information between drugs and cell lines and addressing the limited ability of the hypergraph to capture global information. Additionally, the similarity-based multi-task supervision module further stabilizes the model.
# Overview
The repository is organised as follows:  
Data/ contains the DrugCombDB dataset and the O'neil dataset;  
FineBert/ contains the fine-tuned Chemberta model;  
Model/MMHLSyn contains the model's overall framework.  
# Data
DrugCombDB dataset comprises 548 drugs, 68 cell lines, and a total of 60932 samples.  
O'neil dataset comprises 38 drugs, 32 cell lines, and a total of 18929 samples.
# Requirements
The MMHLSyn network is built using PyTorch .You need to first create a conda environment, then install the following dependencies：    
Torch 2.0.1  
Cuda 11.8  
Python 3.10  
numpy==1.26.4  
pandas==2.2.2  
scipy==1.13.0  
rdkit==2023.9.6  
scikit-learn==1.4.2  
torch_scatter-2.1.1  
torch_sparse-0.6.17  
torch_cluster-1.6.1  
torch_spline_conv-1.2.2  
torch_geometric==2.4.0    
transformers==4.41.1  
sentence_transformers=3.0.0  
deepchem==2.8.0  
# Running the Code
cd Model/MMHLSyn  
python main_reg.py


