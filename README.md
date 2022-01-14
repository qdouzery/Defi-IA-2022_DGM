# Défi-IA 2022
> Ranked 26th out of 84 on the [Kaggle Competition](https://www.kaggle.com/c/defi-ia-2022/overview).<br />
> Predict the accumulated daily rainfall on ground stations. <br />
*Quentin Douzery, Alexia Ghozland, Dario Moed*  


The solution focuses on targeted feature engineering and the use of a regressor model. 
The validation process is not presented.

You will find: 
* **requirements.txt** that describes required packages to run the code
* **train.py** that trains the model and outputs the final trained model as a .h5 file and predictions on the test data as a .csv format.
* **utils.py** gathers the useful functions for preprocessing steps (see preprocess.py)
* **preprocess.py** that manages preprocessing functions for train and test datasets
* **models.py** contains the models to train

How to reproduce the results :
1. Have Python 3 installed
2. Run the following command to access the Kaggle API using the command line: 
 ```sh
$ pip install kaggle
```
3. Download the Kaggle weather data from [here](https://www.kaggle.com/c/defi-ia-2022/data), or by executing this command on your terminal: 
 ```sh
$ kaggle competitions download -c defi-ia-2022
```
4. Run `pip install -r requirements.txt` using a virtual environment,

5. You need to call the script with the following command: 
```sh
$ python train.py --data_path Data --output_folder Results
```
Data is the path to the folder containing the data files downloaded on Kaggle (see 3.) <br />
Results is the path to the folder where to output the model and predictions. <br/>

Execution time: 7mn on following engine: MacBook Air 2017, 1.8GHz intel Core i5 <br/>
