# Defi-IA 2022
> Ranked 26th out of 84 on the [Kaggle Competition](https://www.kaggle.com/c/defi-ia-2022/overview).<br />
> Predict the accumulated daily rainfall on ground stations.  
*Quentin Douzery, Alexia Ghozland, Dario Moed*  


The solution focuses on targeted feature engineering and the use of a regressor model. 
The validation process is not presented.

You will find: 
* **requirement.txt** that describes required packages to run the code
* **train.py** that trains the model and outputs the final trained model as a .h5 file and predictions on the test data as a .csv format.
* **utils.py** gathers the useful functions for preprocessing steps (see preprocess.py)
* **models.py** contains the models to train
* **preprocess.py** that manages preprocessing functions for training and testing datasets


1. Have Python 3 installed
2. Run the following command to access the Kaggle API using the command line: 
 ```sh
$ pip install kaggle
```
3. Download the Kaggle weather data from [here](https://www.kaggle.com/c/defi-ia-2022/data), or by executing this command on your terminal: 
 ```sh
$ kaggle competitions download -c defi-ia-2022
```
4. Run `pip install -r requirements.txt` 

You need to call the script with the following command: 
```sh
$ python train.py --data_path Data --output_folder Results
```


## Évaluation
**The git should contain a clear markdown Readme, which describes : (33%)**
- Which result you achieved? In which computation time? On which engine?
- What do I have to install to be able to reproduce the code?
- Which command do I have to run to reproduce the results?

**The code has to be easily reproducible. (33%)**
- Packages required has to be well described. (a requirements.txt files is the best)
- Conda command or docker command can be furnish

**The code should be clear and easily readable. (33%)**
- Final results can be run in a script and not a notebook.
- Only final code can be found in this script. 
