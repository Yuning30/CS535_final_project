# CS535_final_project

## Project structure

Our code are structured as follows
- dataset.py: Parse the original data and create train/val/test data for DDRSA model.

- model.py: Implementation of encoder and decoder

- train.py: Implementation of DDRSA and WBI model and their training procedure.

- inference.py: Implementation of Equation 9, 10 of the original paper and the function to make decisions for a whole test sequence

- ddrsa.py: Entry point of running the DDRSA model to obtain experiment results

- wbi.py: Entry point of running the WBI model to obtain experiments results.

- plot.py: Used to plot figures used in the report


**To reproduce our result, please follow the steps below:**

## Clone the dataset repository
This step should be skipped if you are using the code downloaded from Canvas.

```
git clone https://github.com/Yuning30/AMLWorkshop
```

## Train a new model
```
python3 train.py --model ddrsa      # train a new ddrsa model
python3 train.py --model wbi        # train a new wbi model
```

## Run the trained model to make predictions
Remember to change the file path for the model to be loaded in corresponding files.

```
python3 wbi.py      # run the wbi model on test dataset
python3 ddrsa.py    # run the ddrsa model on test dataset
```

## Plot figures from log data
Uncomment the corresponding line for the plot you want to create and set the path for the log files. The log files should be obtained by saving the output of wbi.py and ddrsa.py into a file.

```
python3 plot.py
```

