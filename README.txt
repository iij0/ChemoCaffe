
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
---Using Rdkit---
---Preparing the data---

Once you follow the instructions above, it is time to prepare the data to be read by ChemoCaffe.  

1) Check that your data is prepared exactly like example.csv.
2) Move ShuffleSplit.py to the directory that will contain your train/test splits
3) Type "python ShuffeSplit.py /path/to/data.csv [number of splits] [fraction of data to be withheld for testing]"
For example, "python ShuffleSplit.py /path/to/data.csv 10 0.1", would create 10 train/test splits where 10% of the data is withheld for testing and 90% of the data is used for testing

---Preparing configuration files---

1) Open the relevant config file
	-For tuning, open example.cfg to set up the parameters for tuning your network.
	-For predictions, open predict.cfg
2) The [DEFAULT] section contains global parameters that apply to all of your networks.
3) [Net1], [Net2], ..., [NetN] are used to define different network configurations to be tested.  Just add another section if you would like to test more than 2 networks.

---Using the program---
1) It is recommended that you use first ChemoCaffe_tune.py to find the network(s) that gives you the best results.  
2) After you have set up your configuration files and prepared your data, type "python ChemoCaffe_tune.py path/to/example.cfg" to run the program
3) The results should be available in the file specified in the output field in the configuration file.
4) After you have chosen a couple of networks that gave you good results, use ChemoCaffe_predict.py
5) After you have set up your configuration files and prepared your data, type "python ChemoCaffe_predict.py path/to/predict.cfg" to run the program
6) The predictions should be available in the file specified in the output field in the configuration file.
