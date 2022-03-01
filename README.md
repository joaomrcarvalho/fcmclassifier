## Classification of 1D-signals using Extended-Alphabet Finite-Context models

This package provides an easy to use solution to perform classification on uni-dimensional signals, using an approach that relies on Extended-Alphabet Finite-Context Models. The code was developed from scratch by myself for research purposes, but feel free to reproduce it for whatever you like. Just make sure to cite the papers mentioned on the [citations](#CITATION) section.

The code was developed with the Linux system in mind and should work out of the box in either Ubuntu machines or WSL (Windows Subsystem for Linux) -- either WSL1 or WSL2.




## INSTALLATION
To install the package, all you need to do is run these commands on a Linux terminal (or WSL):
```bash
# Download repo
git clone https://github.com/joaomrcarvalho/fcmclassifier.git

# Enter repo
cd fcmclassifier

# Install requirements
python -m pip install -r requirements.txt
```
All dependencies will be automatically installed.


## USAGE
The program was developed using the "click" package, so you can use the command

```bash
python fcmclassifier.py --help
```

to get information about the parameters expected.

## EXAMPLE
A simple example is provided in download_and_test_sample.sh. This script will make sure you have the zip package, download a sample database, unzip it and attempt to run the classifier on its' data (folder "sample_data"), producing a folder named "quantized_data" with the quantized data, as well as the file "results_nrc.csv" with the computer NRC values. 

To run this script, navigate to the main directory of the proect and type the following command:
```bash
sh download_and_test_sample.sh
```


## CITATION
This is part of the research I have done during my PhD and there are a couple of papers related with the subject. Please cite the following two papers if you use this project:

* João M. Carvalho, Susana Brás, Diogo Pratas, Jacqueline Ferreira, Sandra C. Soares and Armando J. Pinho. "**Extended-Alphabet Finite-Context Models**", Pattern Recognition Letters Volume 112, 1 September 2018, pp. 49-55. doi: [10.1016/J.PATREC.2018.05.026](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302095?via%3Dihub)


*    João M. Carvalho, Susana Brás and Armando J. Pinho. "**Compression-Based Classification of ECG Using First-Order Derivatives**" on Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering, Springer International Publishing. 2019 pp. 27--36
doi: [10.1007/978-3-030-16447-8_3](https://link.springer.com/chapter/10.1007/978-3-030-16447-8_3)


## ISSUES
Please let me know if you find any issue, at https://github.com/joaomrcarvalho/fcmclassifier/issues

## LICENSE
GPL v3.

For more information, please visit http://www.gnu.org/licenses/gpl-3.0.html
