# SAAN
THIS IS A METHOD ABOUT SAAN MODEL.
A novel signal adaptive augmentation network (SAAN) to effectively construct artificial samples for amplifying fault data volume.

This project is used to realize the adaptive augmentation of fault data under small sample conditions. For vibration signals, this method takes local fault impulse as the minimum diagnosable unit of mechanical equipment. This method first extracts the local fault impulses with different fault modes from small sample fault data, and then combines them according to the theoretical impulse time interval. Secondly, the real fault samples and the combined artificial fault samples are fed into the SAAN at the same time. The SAAN can regulate the combined artificial fault samples with reference to the characteristic distribution of the real fault samples, so as to make them have similar characteristic distribution and noise level. Finally, the obtained artificial fault samples effectively expand the fault data volume. Several experiments show that the generated artificial samples can effectively improve the classification performance of different fault classifiers. More details are available in the paper.

The code of SAAN implementation is given in this project. Due to the protection of intellectual property rights, the author can’t public the datasets in the paper, but this code can be verified on various public vibration signal datasets (such as Case Western Reserve University bearing failure public dataset).

Running environment: Python 1.5.0, Python 3.7.6.
This project first runs the train_data_preparation.py, in which you can modify your own training dataset path.
The hyper-parameters of the SAAN model can be set in the SAAN_ params.py.
Running the SAAN_main.py file can generate artificial fault samples.
Note that leaving a message to the authors if you have any questions.


