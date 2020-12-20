# DataSentiment

![GitHub](https://img.shields.io/github/license/EricaWei053/COMS4995)
![Travis (.org) branch](https://travis-ci.org/EricaWei053/DataSentiment.svg?branch=master)
![Codecov](https://img.shields.io/codecov/c/github/EricaWei053/DataSentiment)
[![Read the Docs](https://img.shields.io/readthedocs/stocksentiment)](https://stocksentiment.readthedocs.io/en/latest/?)

## Description
Sometimes the data analysis is a very time-consuming and dry process. 
This platform provides an automatically statistic analysis results, cumulative return plot, 
histogram and 3D visualization for given time-series data. The data could be ticker price or 
profit and loss(pnl) data. The input data reqiuremtns are two columns, one column is date, 
another one is value depending on the date. 
 

## Usage 
Backend mainly in Python in processing file, data, calculating values. 
Use Plotly Dash to illustrate the plotting and Analysis. 

- Time-series data  
- PnL or ticker price 
- Statistic Analysis Table 
- View plots with several choices  

## How to run it 
```
pip install -r req.txt 
```
```
python ./src/plot.py [file1] [file2]
```

And then you can click the url link generated and viewing the contents.

Please make sure the file is in csv format and 
you enter the full or absolute path for the current root. 
 
##  Disclaim: 
This app will try to make sure everything is calculating correctly, but it could be the case the data is dismatch and get insane results. If there's anything wrong and could be improved, please add to issue and send PR. 

## Optional Develop: 
Will add Deep Learning network for potential development feature. 

