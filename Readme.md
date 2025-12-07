Generative Regularities in Multi-Layer Networks: A Shared-Latent Space Representation Approach
==============================================================================================================

Usages
------------------------

Type:
    `python support_main.py --dataset Aarhus`
to have a try of SupportNet on the Aarhus network.


Type:
    `python support_main.py --dataset Aarhus --onlyTest`
to only load the model for prediction.


Type:
    `python support_main.py --dataset Aarhus --epochs 100 --EarlyStop --patience 20`
    Use early stopping strategy. Set the maximum number of epochs for model training. Set the number of patient epochs for the early stop strategy.


Type:
    `python support_main.py --dataset XXX --best_metric aupr`
--best_metric: Which metric should be chosen as the best model performance indicator: AUC, AP, AUPR, ...


Type:
    `python support_main.py --dataset XXX --set_seed 42`
    --set_seed to set the random seed, 42 or 2025 or ...


Type:
    `--gcn_type GCN`
    `--gcn_type JK_GCN`
Configure the GCN type used by the Layer-wise Representation Extractor in SupportNet: Currently, both GCN and JK_GCN are supported.


Type:
    `--gcn_layer 4`
Configure the number of layers in the Layer-wise Representation Extractor's GCN in SupportNet to aggregate information across multiple hops. e.g., 2, 3, 4.

------------------------

Type:
    `python CLGC_main.py --Eval_layers Aarhus_1 Aarhus_2`
Evaluate the cross-layer generation consistency score (CLGC score) of Aarhus_1 and Aarhus_2.


Type:
    `python CLGC_main.py --Eval_layers Aarhus_1 small_world`
Evaluate the cross-layer generation consistency score (CLGC score) of Aarhus_1 and the theoretical network small_world.
Currently, three theoretical networks are supported: small_world, scale_free, and random_graph.


Type:
    `python CLGC_main.py --Eval_layers Aarhus_1 Enron_1 Kapferer_1 LonRail_1`
Evaluate the cross-layer generation consistency score (CLGC score) of the group: Aarhus_1 Enron_1 Kapferer_1 LonRail_1.


Type:
    `python run.py`
can directly reproduce CLGC-related experiments.



Requirements
------------------------
Latest tested combination: Python 3.8 + Pytorch 2.1

Package                       Version
----------------------------- ----------------
imbalanced-learn              0.12.4
matplotlib                    3.7.5
networkx                      2.8.8
numpy                         1.24.1
node2vec                      0.4.6
pandas                        2.0.3
scikit-learn                  1.3.2
torch                         2.1.0
torch-cluster                 1.6.2+pt21cu121
torch_geometric               2.4.0
torch-scatter                 2.1.2+pt21cu121
torch-sparse                  0.6.18+pt21cu121
torch-spline-conv             1.2.2+pt21cu121
torchaudio                    2.1.0
torchvision                   0.16.0
tqdm                          4.67.1





