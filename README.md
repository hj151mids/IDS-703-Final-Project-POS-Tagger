# Solving POS-Tagging With Two Methods
Final Project of IDS 703: Natural Language Processing at Duke University  
Author: Haoliang Jiang, Athena Liu, Mingjung Lee
## Problem Concentration:
The topic concentration of the final project is to perform Part of Speech Tagging for text. Part of Speech (also known as POS) Tagging is the process of labeling each of a sequence of words within a text with their grammatical attribute. We aimed to construct both a generative probabilistic model – the Hidden Markov Model(HMM) -- and a neural network – Bidirectional LSTM – that is capable of correctly labeling synthetic data and real world text. And we compared the two models and figured out the pros and cons of each.
## Generative Probability Model: Hidden Markov Chain and Viterbi Decoder
The generative probabilistic model we implemented is the Hidden Markov Chain with Viterbi Algorithm as the decoder. The Hidden Markov Model is a preferable probabilistic model in POS tagging because it considers both the observed states (words) and the hidden states (tags) as causal factors for the final prediction. In wrote self-defined functions to calculate the transition matrix, emission matrix, and the initial probability distribution. These informational matrices encoded the relationship between the observed states and the hidden states, which will be decoded by the Viterbi Algorithm. OOV function and matrix smoothing are also included in the calculation of these matrices. The self-defined function Viterbi takes the transition matrix, emission matrix, initial probability distribution, and observations as input. Viterbi algorithm helps to determine the most probable set of tags when given a set of observations. The recursive algorithm computes and searches for the most optimized
tag sequences by following the mathematical formula below:  

![Screenshot 2022-01-28 144126](https://user-images.githubusercontent.com/90075179/151611134-afd1daab-3b3f-4bfc-b414-9609bab49bb0.jpg)  

Training of the model requires a dataset that contains both the observed states (words) and the hidden states(tags). Luckily, we are able to use the NLTK corpus treebank to train the model. Testing of the algorithm requires simply a sequence of observations (words), and the expected output is a sequence of tags that corresponds to each observation.  

## Discriminative Neural Netowrk: Bidirectional LSTM Neural Network
The structure of our Pytorch POSTagger neural network consists of three major parts. The word tokens from Pytorch’s torchtext data library will first enter an embedding layer for encoding. We used pre-trained word vectors from GloVe for encoding the tokens. And then, the encoded vectors are fed to a bidirectional LSTM layer with their corresponding pre-trained weights. As for the bidirectional LSTM layer, we have decided to use the adam optimizer to optimize the gradients. We have also picked cross-entropy loss as our loss function. Last but not least, we can get predicted tags after decoding using the linear layer. One thing to notice is that the dropout layers are added to prevent overfitting. As for the training of the model, we have enabled parallel computing on our GPU to speed up the
process. We printed out the training, validation, and testing accuracy in the attached notebook in each epoch. We have run ten epochs to ensure that these accuracies do not vary by a lot any longer.  

![Screenshot 2022-01-28 144555](https://user-images.githubusercontent.com/90075179/151611711-aee7d058-ecb9-45f6-b5cb-b25e1ad6bf21.jpg)  

## Performance on Synthetic Data
We implemented our own synthetic data generator. Since our concentration is Part-of-Speech tagging, we need to generate both the tags and the words. To obtain initial words and tags probability distribution, we trained a Hidden Markov Model and extracted three matrices -- transition matrix, emission matrix, and initial state distribution. Using the NLTK Brown corpus as the training set, words and tags are generated randomly by the probability distribution matrices extracted from HMM. In total, 100 synthetic sentences were generated, and each contains ten tokens.  
Due to unknown word issues, the Hidden Markov Model has to be trained again with the NLTK Brown corpus. The 100 sentences were imputed as testing data, and the performance of the model will be evaluated by the pre-built sklearn classification_report() function. The evaluation metrics we used are precision, recall, and the F1 Score. Precision measures how many of the positive predictions are correct. Recall measures how many of the positive cases were correctly classified. And the F1 Score is the harmonic mean of precision and recall, which is a good indicator for overall performance of the classifier. We utilized the Scikit-Learn Classification Report function to calculate and display the values of precision, recall, and F1.  

![Screenshot 2022-01-28 150357](https://user-images.githubusercontent.com/90075179/151613887-8dfff94d-7acd-4253-8a9e-1f468db03d51.jpg)  

The table above displayed the performance of the Hidden Markov Model on the synthetic data. While the 0.845 accuracy score might look relatively promising, it is important to acknowledge that Brown was used as training data. A major cons of using Hidden Markov Model is its ability to handle unseen words, and the solution will greatly depend on the implementation of the OOV function. We believe that the model is overfitting on the synthetic data, which led to the high accuracy.  
As for the Pytorch neural network model, we first altered the input data structure from a list of lists of word-tag pairs into Tensor objects. We then implemented the test using the trained model from step 3. The model outputs are tensor objects consisting of probabilities, so we transformed them into flattened lists. Last but not least, we used the sklearn library to generate the metrics (accuracy, precision, recall) by class.As for the Pytorch neural network model, we first altered the input data structure from a list of lists of word-tag pairs into Tensor objects. We then implemented the test using the trained model from step 3. The model outputs are tensor objects consisting of probabilities, so we transformed them into flattened lists. Last but not least, we used the sklearn library to generate the metrics (accuracy, precision, recall) by class.  

![Screenshot 2022-01-28 150528](https://user-images.githubusercontent.com/90075179/151614069-fcf1a149-4a59-4099-b2de-b50336ea220b.jpg)  

As we can see from the above table, the neural network model correctly predicted most of the tags. However, the model did poorly when predicting PART, NOUN, and VERB. Especially for particles, the F-score is only 0.09. The reason behind this is that the data is synthetic, and each word-tag pair is drawn from a distribution, so we might get some extreme or unreal data in the end because of the smoothing we implemented in step 2.

## Performance on Real Data
The real data used in testing the Hidden Markov Model in this step is the NLTK Brown corpus. The dataset was split into training and testing. Again, the model has to be trained on brown in order to avoid potential unseen words. 100 sentences are randomly selected from the Brown corpus as testing data, while the remaining are used as training data.
![Screenshot 2022-01-28 150943](https://user-images.githubusercontent.com/90075179/151614639-3f5e3141-7ce8-41fc-aacd-92ccd92b7bb2.jpg)  

The above table displayed the performance of the Hidden Markov Model. The quality of the classifier is
promising because both the precision rate and the recall rate of the classifier are in the high 80s. The accuracy of the model is also 87%, which is relatively high. It is important to acknowledge that this testing data is significantly smaller than the dataset used for testing for neural networks. The limited testing set could cause potential bias in the evaluation of the model.  
Since the Pytorch neural network model is trained on torchtext’s UDPOS dataset instead of NLTK’s brown
dataset, it has never seen data from the brown dataset. As a result, for this part, we will feed the brown dataset as the real-world data into our POSTagger neural network to test out its performance. We adopted similar data cleaning, restructuring, and testing techniques presented in step 4. Here is a table of metrics by class.  
![Screenshot 2022-01-28 151626](https://user-images.githubusercontent.com/90075179/151615423-000664f7-29f2-450b-8bd0-d7b539c6e3f9.jpg)  

Overall, the model did perform a lot better than step 4. We believe that the reason behind this is that real-world data preserve more natural language properties. For instance, you will never observe two determinants appearing in a roll. Nevertheless, this can happen in the synthetic dataset because every word-tag pair is generated by chance. So, it is not surprising to see that the neural net model performs better when real-world data is fed into it. However, we still notice that the model performs poorly on classifying pronouns. We hypothesize that this can be the limitation of our model. Maybe the neural network model cannot learn the ambiguity of pronouns under different contexts.  

## Conclusion: Pros and Cons

![Screenshot 2022-01-28 152019](https://user-images.githubusercontent.com/90075179/151615924-84eaa3ea-4233-4d0e-afd7-7a22a870aba9.jpg)
