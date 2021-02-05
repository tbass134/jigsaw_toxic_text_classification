# Toxic Text Classification Kaggle Competition

The goal of this project is to build a full end-to-end machine learning based solution on classifing different types of text. 
These classifications include:

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

These types of comments can be hurtful and insentive to others, therefore, being able to removing these types of  messages is benificial in keeping a online community safe and allows for all users enjoy partisipanting with fear of judgement.  

The project is composed into 4 Sections:
* [EDA](#EDA)
* [Basline Model](#Baseline-Model)
* [Improvments](#improvments)
* [Deployment](#deployment)

## EDA
This notebook includes the basic code on understanding the data. This includes how to preprocess the text for model training. This includes:
* Removing Stopwords
* Removing URL's
* Removing newlines

## Baseline Model
For the first iteration, I used a Nieve Bayes model to classifiy these toxic comments. This used the same preprocesses as found in the EDA notebook as well as using sklearn's CountVectorizer to convert the text into a matrix of numbers. This matrix is then feed into the Nieve Bayes model to train.
Currently, the model ROC score is ~0.94.
I've also intergrated [Paperspace Gradient](https://gradient.paperspace.com) into the code commit process. Each time a commit is made, it runs a training job that trains the model and save the model file that can be downloaded from Gradient. 

I've documented more about it [here](https://twitter.com/Thung/status/1349704434268459009)

The training script can be found [here](src/sklearn/train.py)

## Improvments
This is still a work in progress, however, will be using a LSTM to see weather this is better than the baseline model.
My goal is to also try a few other models, including [Google's BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) and a [CharCNN](https://arxiv.org/abs/1509.01626)

## Deployment
Once our model is better than our NB model, we'll deploy to a server. Currently, my goal is to use [KubeFlow](https://www.kubeflow.org) for inference as for retraining. This will be complete once the deeper model is complete.
 
