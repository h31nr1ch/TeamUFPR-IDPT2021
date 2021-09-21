# TeamUFPR-IDPT2021

TeamUFPR at IDPT 2021. Complete set of results and source code.

## Table of Contents ##
- [Running the code](#Running-the-code)
- [Contributors](#Contributors)
- [IberLEF-2021](#IberLEF-2021)
- [License](#License)

## Running the code ##

If you wish to run the code, three options are available (hardcoded, sry :/ )

```
372    def main(self):                                                                                                                           
373         # self.makerMain() # Run all the tests to check which algorithm is better                                                                                             
374         # self.finalNews() # Our solution to News dataset                                                                                             
375         self.finalTweet()  # Our solution to Tweet dataset
```

To run the code, use the following command:

```
python3 main.py --dataset-train ../dataset/Training/training_tweets.csv --dataset-test ../dataset/Test/test_tweets.csv 

python3 main.py --dataset-train ../dataset/Training/training_news.csv --dataset-test ../dataset/Test/test_news.csv 
```

## IberLEF-2021 ##

Published in Iberian Languages Evaluation Forum 2021:
```
@inproceedings{paper3,
    author={Tiago Heinrich and Fabrício Ceschin and Felipe Marchi},
    title={TeamUFPR at IDPT 2021: Equalizing a Strategy Using Machine Learning for Two Types of Data in Detecting Irony},
    year={2021},
    series={Iberian Languages Evaluation Forum 2021}
}
```
The paper could be found [here](https://www.researchgate.net/publication/354723705_TeamUFPR_at_IDPT_2021_Equalizing_a_Strategy_Using_Machine_Learning_for_Two_Types_of_Data_in_Detecting_Irony) or [here](http://ceur-ws.org/Vol-2943/).

## Contributors ##
* [h31nr1ch](https://github.com/h31nr1ch) (Tiago Heinrich) (owner)
* [fabriciojoc](https://github.com/fabriciojoc) (Fabrício Ceschin) (owner)
* [Markhyz](https://github.com/Markhyz) (Felipe Marchi) (owner)
