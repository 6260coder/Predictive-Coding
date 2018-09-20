 Purpose of model:  
Given a sequence, predict the next token according to a particular style, which implemented in a loop, generates a sequence.

Structure of model:  
Multi-layer one-directional recurrent neural network with shared softmax across each time step, which gives prediction of the token at the next time step.

Management of files:  
* **data_helpers.py** holds utility functions regarding handling of data.
* **data_preprocessing.py** takes the raw dataset and pickles it as a list of indexed sequences.
* **Predictive_Coding_Model.py** holds the structure of the neural network as a class.
* **Sequence_Generator.py** contains a class that takes care of the sequence generation using a pretrained network of the structure described in **Predictive_Coding_Model.py**.
* **Train.py** implements the training of the network.
* **Sequence_Generation** uses the class hosted in **Sequence_Generator.py** and does the actual sequence generation

Dataset:  
English movie review text dataset with negative and positive polarities. The dataset contains 10662 documents with maximum length document length of 56.

Results:  
When trained on a dataset as small as 100 sequences, training accuracy could get as high as 0.9. But genenated sequences are far from satisfactory, which even bare little resemblance to proper English. 
Takes a long time to train on a dataset of a considerable scale. Didn't wait to see the results.


