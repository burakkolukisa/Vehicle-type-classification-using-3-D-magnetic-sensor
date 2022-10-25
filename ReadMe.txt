If you are using this dataset or model, please cite these papers.

1. Kolukisa, B., Yildirim, V. C., Ayyildiz, C., & Gungor, V. C. (2022). A deep neural network approach with hyper-parameter optimization for vehicle type classification using 3-D magnetic sensor. Computer Standards & Interfaces, 103703.

2. Kolukisa, B., Yildirim, V. C., Elmas, B., Ayyildiz, C., & Gungor, V. C. (2022). Deep learning approaches for vehicle type classification with 3-D magnetic sensor. Computer Networks, 217, 109326.


1. Dataset

A 3-D magnetic sensor is built and mounted on a single-lane road. While the vehicle passes over the sensor, the 3-D magnetic sensor records the disturbances due to the metal materials in the vehicle. These distortions cause signal changes and differ from each other. This difference is mostly caused by the speed, physical characteristics of the vehicle, and environmental factors. The vehicle types are divided into three categories considering the structure of the vehicle: light (L: class 1-motorcycles), medium (M: class 2-passenger cars), and heavy (H: class 3-buses). In total, 376 vehicle samples are labeled (light: 46, medium: 298, and heavy: 32).

The dataset and the feature extracted dataset are shared as "class3.csv" and "class3_FE.csv" respectively.

For vehicle type classification, first apply "class3.py" or "class3_FE.py", and the dataset is ready for running classifier.

2. Vehicle Classification

Classification algorithms are implemented using Python with the Scikit-Learn and Keras libraries.

For the DNN model's loss function, the focal-loss is set up with the help of the tensorflow-addons library.

The DNN model applied in this study is available as "DNN_model.py".

The hyper-parameter codes of the DNN model are available as "DNN_hyper-parameter.py". The training set is divided into five parts, and the best validation scores are printed, and using the best parameters, models are retrained with all the dataset as available on "DNN_example".

The best scores obtained with the DNN model are stored in the format of ".h5" files. You can directly use these models and run "DNN_load-model.py" to obtain the results.

During the evaluation, we used GPU on Google Colab Pro.
