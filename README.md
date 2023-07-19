# Stock prediction module
This repo is for K-Quant project, stock forecasting module.
### ---------------------------------[Update] July 19th----------------------------------------
Add incremental learning module. Run
```commandline
python exp/learn_incre.py
```
Then your model file will be stored in the folder you choose.
### ---------------------------------[Update] July 13th----------------------------------------
Update pretrain models, now we support those following models in prediction mode:
```commandline
# No knowledge
MLP
GRU
LSTM
ALSTM
SFM
GATs
PatchTST
# Knowledge embed
HIST
RSR [is, hidy, hidy_is, dueefin_is, sht_is]
# to save a prediction result:
python exp/prediction.py --model_path './output/for_platform/RSR_hidy_is' --pkl_path './pred_output/csi_300_rsr_hidy_is.pkl'
```
## Attention:
For knowledge empowered model, we only support use THE SAME file while you train the model

So when it comes to dynamic knowledge, you need to update the knowledge file and cover the path in ```exp/prediction.py main()```

For example, HIST needs up-to-date market value, and we use old one now which may could impact the model
s performance.

### ---------------------------------[Update] July 8th-----------------------------------------
We rewrite prediction.py and re-structure the pipeline;

Now you can try the mlp model in this pipeline, from training to predicting;

To get the prediction result, run the following command line:

```commandline
python exp/prediction.py -pkl_path './pred_output/xxx.pkl'
```
The prediction result will be stored in ```pred_output``` folder

We recommend using an up-to-date Qlib data source
```commandline
wget https://github.com/chenditc/investment_data/releases/download/2023-07-01/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C cn_data --strip-components=2
```
Now other models are under training, and the pipleline might have some bugs on models except MLP
### Introduction
Now we provide the following models that could be used in stock regression/forecasting/recommendation:
```
-----------------------basic models--------------------
MLP
GRU
LSTM
ALSTM
SFM
GATs
------------------models powered by knowledge-----------
HIST
RSR
relation_GATs
Triple_Att
------models that SOTA on other time series library-----
DLinear [AAAI 2023]
Autoformer [NeurIPS 2023]
Crossformer [ICLR 2023]
ETSformer
FEDformer [ICML 2022]
FiLM [NeurIPS 2022]
Informer [AAAI 2021]
PatchTST [ICLR 2023]
---------------------------------------------------------
```
## Environment
1. Install python3.8(recommend) 
2. Install the requirements in [requirements.txt].
3. Install the quantitative investment platform [Qlib](https://github.com/microsoft/qlib) and download the data from Qlib:
    ```
    # install Qlib from source
    pip install --upgrade  cython
    git clone https://github.com/microsoft/qlib.git && cd qlib
    python setup.py install

    # Download the stock features of Alpha360 from Qlib
    # the target_dir is the same as provider_url in utils/dataloader.py
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
    ```
4. Download market_value, index file for knowledge empowered models from [this link](https://drive.google.com/file/d/1KBwZ_lX___bYBIHx9VWRzRgLFb8N3-NK/view?usp=sharing)
## Run experiments
    python learn.py --model_name [model you choose] --outdir 'output/[folder your named]'
## Results
The result will be stored in output folder, if you need some well-trained models, we provide in [this link](https://drive.google.com/file/d/1yGHXZDcCgY4AAp_UM_gKXyKo25Atmoft/view?usp=sharing)

## acknowledgement

Thanks to research work [HIST](https://github.com/Wentao-Xu/HIST) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library/)
