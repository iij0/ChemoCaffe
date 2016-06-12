
ChemoCaffe is a python wrapper for Caffe.  It's purpose is to automate the training and testing of DNN's while only having to modify one configuration file.


ChemoCaffe requires the following software:
================================================================================
List of software required:
1) Caffe rc3 : https://github.com/BVLC/caffe/releases/tag/rc3
2) Python 2.7.9
3) hdf5 : http://www.h5py.org/
4) CUDA 7.5: https://developer.nvidia.com/cuda-downloads
5) Boost Version 1.59.0 : http://www.boost.org/users/history/version_1_59_0.html
6) Scikit-learn: http://scikit-learn.org/stable/
7) Rdkit: http://www.rdkit.org/
================================================================================
---Preparing the data---
First, we need to generate our descriptors using Rdkit.

1) Use RdkitDescriptors.py located in Scripts/Prepare Data/
2) Prepare your data in a format similar to the example given in Data/Training_Testing.csv
3) Type "python RdkitDescriptors.py path/to/data.csv" to generate the descriptors

Once you follow the instructions above, it is time to prepare the data to be read by Caffe.  

1) Move ShuffleSplit.py to the directory that will contain your train/test splits
2) Type "python ShuffeSplit.py /path/to/data.csv [number of splits] [fraction of data to be withheld for testing]"
For example, "python ShuffleSplit.py /path/to/data.csv 10 0.1", would create 10 train/test splits where 10% of the data is withheld for testing and 90% of the data is used for testing

---Tuning the network---
1) Open Scripts/Tune/example.cfg
2) Prepare the configuration file.
	-[DEFAULT] section contains global parameters that apply to all of your networks.
	-[Net1], [Net2], ..., [NetN] are used to define different network configurations to be tested.  Just add another section if you would like to test more than 2 networks.
3) After you have set up your configuration files and prepared your data, type "python ChemoCaffe_tune.py path/to/example.cfg" to run the program
4) After some time, your results should be located in the CSV file specified in the configuration file.

---Generating Predictions---
1) Open Scripts/Predict/predict.cfg
2) Select a few networks that gave you good results while tuning the network and add them to the configuration file.
	-[DEFAULT] section contains global parameters that apply to all of your networks.
	-[Net1], [Net2], ..., [NetN] are used to define different network configurations to be tested.  Just add another section if you would like to test more than 2 networks.
3) After you have set up your configuration files and prepared your data, type "python ChemoCaffe_predict.py path/to/predict.cfg" to run the program
4) The predictions should be available in the file specified in the output field in the configuration file.

