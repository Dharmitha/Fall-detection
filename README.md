# Neural-Networks

Fall is one of the most common causes of accidental health. Elderly are more to falling when compared to others. So a cheaper aletrnative is required which can be implemented in retirement and nursing homes. In this project we are using a published dataset called "MobiAct" to build machine learning and neural network models for fall detection. The techniques used are Long short term memory, Multilayer perceptron, Support Vector Machines, Weighted K-nearest neughbours, Boosted trees and Bagged trees. Long short term memory model is built using Keras with Tensorflow as backend. Rest of all the models are built using Matlab R2017b classification learner and neural network toolbox.

The MobiAct dataset contains labeled information about four kinds of simulated falls from fifty-four subjects, and nine kinds of daily activities from fifty subjects. Each activity in the dataset contains time stamp, raw accelerometer values, raw gyroscope values and orientation data. In our experiments, we used a subset of the data containing only two kinds of daily activities and four kinds of falls. The daily activities are standing and lying, and the falls are as follows:

•	FOL: forward-lying (fall forward from standing, use of hands to dampen fall)

•	FKL: front-knees-lying (fall forward from standing, first impact on knees)

•	SDL: sideward-lying (fall sideward from standing, bending legs)

•	BSC: back-sitting-chair (fall backward while trying to sit on a chair)

A total of 58 Features are extracted from raw accelerometer values. They are further normalized using min-max scaling algorithm. A feature selection algorithm called Relief is used to select the most imprtant features. A total of six experiments is performed using the features generated and selected by above methods.

Experiment 1: 58 features, 4 kinds of falls, 2 kinds of activities

Experiment 2: 58 features, 2 kinds of falls, 2 kinds of activities

Experiment 3: 10 features, 4 kinds of falls, 2 kinds of activities

Experiemnt 4: 10 features, 2 kinds of falls, 2 kinds of activities

Experiemnt 5: Balanced combined dataset with only fall and non-fall as label

Experiemnt 6: Balanced selected dataset with only fall and non-fall as label

Collaborator:
Sazia Mahfuz

References:
Vavoulas G., Chatzaki C., Malliotakis T., Pediaditis M. and Tsiknakis M.., "The MobiAct Dataset: Recognition of Activities of Daily Living using Smartphones.," In Proceedings of the International Conference on Information and Communication Technologies for Ageing Well and e-Health - Volume 1, pp. 143-151, 2016.
