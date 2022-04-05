Neural Collaborative Filtering - PyTorch implementation | ECE-269 Project
==========================================================================

In this project, I worked on implementing Neural Collaborative
Filtering models in PyTorch from the paper: <br>

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.  <br>

Through this process we set out to build primarily 3 models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP) and Neural Matrix Factorization (NMF) model. We shall then use our implementation to perform prediction and comparative evaluation on the MovieLens dataset. <br>

Please refer to [report]((https://github.com/Rajasvi/neural_collaborative_filtering/blob/master/report.pdf)) for detailed information about the project.

## Environment Settings
We solely used PyTorch for model implementation and Plotly-express for plotting model metrics.
- PyTorch: '1.5.0'
- Plotly-express: '5.6.0'

## Examples to run model code using scripts
Following sample commands can be used to run individual model scripts with requried parameters.<br>
GMF:<br>
```
python src/GMF.py --dataset ratings.dat --epochs 20 --batch_size 256 --num_factors [8] --neg_per_pos 4 --lr 0.001 --learner adam --verbose 1 --out 0
```
MLP: <br>
```
python src/MLP.py --dataset ratings.dat --epochs 2 --batch_size 256 --num_factors [8] --neg_per_pos 4 --lr 0.001 --learner adam --verbose 1 --out 0
```
NeuMF (without pre-trained embedding layer): <br>
```
python src/NeuMF.py --dataset ratings.dat --epochs 2 --batch_size 256 --num_factors [8] --neg_per_pos 4 --lr 0.001 --learner adam --verbose 1 --out 0 --alpha 0.2
```
NeuMF (with pre-trained embedding layer): <br>
```
python src/NeuMF.py --dataset ratings.dat --epochs 2 --batch_size 256 --num_factors [8] --neg_per_pos 4 --lr 0.001 --learner adam --verbose 1 --out 0 --pre_train_path ../pretrain_models/ --alpha 0.2
```

## Dataset
<b>Base 1M MovieLens Dataset</b>: The experiments are conducted on the MovieLens dataset. This movie rating dataset has been widely used to evaluate collaborative filtering algorithms. It consists of 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000. It comprises of User information such as gender, age, and occupation, as well as Movie information such as titles and genre. For our experiment we need user, ratings columns since we are solving for implicit feedback data. While the dataset consists of explicit feedback data, we have modified it to implicit data, where each entry is marked as 0 or 1 indicating whether the user has rated the item. <br>
<br>
<b>Negative Sampling</b>: Since base data has only positive user-rating interactions (positive feedback) we generated 4 negative samples per positive interaction which is randomly sampled from set of movies a user didn’t interact with.

## Project Organization

    ├── LICENSE
    ├── README.md                       <- The top-level README for developers using this project.   
    ├── report.pdf                      <- Final submission report
    │
    ├── data                            <- Raw Data (Movie Lens)
    ├── pre_train_models                <- Main Code Directory
    ├── reference                       <- Reference NCF paper
    ├── plots                           <- Model performance metrics plots
    ├── notebooks                       <- Jupyter notebooks with unclean code
    │
    ├── src                             <- Main code directory
    │   ├── dataset.py                  <- code to pre-process raw data
    │   ├── utils.py                    <- Training & Evaluation utils 
    │   ├── GMF.py                      <- Generalized Matrix Factorization Model
    │   └── MLP.py                      <- Multi-Layered Perceptron Model
    │   └── NeuMF.py                    <- Neural Matrix Factorization Model

