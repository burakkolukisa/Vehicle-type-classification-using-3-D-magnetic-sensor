If you are using this dataset, please cite these papers.

1.

2. Kolukisa, B., Yildirim, V. C., Elmas, B., Ayyildiz, C., & Gungor, V. C. (2022). Deep learning approaches for vehicle type classification with 3-D magnetic sensor. Computer Networks, 217, 109326.

1. Dataset

A 3-D magnetic sensor is built and mounted on a single-lane road. 
While the vehicle passes over the sensor, the 3-D magnetic sensor records the disturbances due to the metal materials in the vehicle. 
These distortions cause signal changes and differ from each other. This difference is mostly caused by the speed, physical characteristics of the vehicle, and environmental factors. 
The vehicle types are divided into three categories considering the structure of the vehicle: light (L: class 1-motorcycles), medium (M: class 2-passenger cars), and heavy (H: class 3-buses).


In total, 376 vehicle samples are labeled (light: 46, medium: 298, and heavy: 32).

The dataset and the feature extracted dataset are shared as a ".csv" file.



2. Vehicle Classification

Classification algorithms are implemented using Python with the Scikit-Learn and Keras libraries.

The focal-loss is used with the tensorflow-addons library for the loss function of the DNN model.