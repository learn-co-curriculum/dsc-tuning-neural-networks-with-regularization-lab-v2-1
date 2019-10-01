
# Tuning Neural Networks with Regularization - Lab

## Introduction

Recall from the last lab that you had a training accuracy close to 90% and a test set accuracy close to 76%.

As with your previous machine learning work, you should be asking a couple of questions:
- Is there a high bias? yes/no
- Is there a high variance? yes/no 

In this lab, you'll use the a train-validate-test partition to get better insights of how to tune neural networks using regularization techniques. You'll start by repeating the process from the last section: importing the data and performing preprocessing including one-hot encoding. Then, just before you go on to train the model, you'll see how to include a validation set. From there, you'll define and compile the model like before. However, this time, when you are presented with the `history` dictionary of the model, you will have additional data entries for not only the train and test set but also the validation set.

## Objectives

You will be able to:

* Construct and run a basic model in Keras
* Construct a validation set and explain potential benefits
* Apply L1 and L2 regularization
* Apply dropout regularization
* Observe and comment on the effect of using more data

## Import the libraries

As usual, start by importing some of the packages and modules that you intend to use. The first thing you'll be doing is importing the data and taking a random sample, so that should clue you in to what tools to import. If you need more tools down the line, you can always import additional packages later.


```python
#Your code here; import some packages/modules you plan to use
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
```

    /Users/matthew.mitchell/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


## Load the Data

As with the previous lab, the data is stored in a file **Bank_complaints.csv**. Load and preview the dataset.


```python
#Your code here; load and preview the dataset

df = pd.read_csv('Bank_complaints.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product</th>
      <th>Consumer complaint narrative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Student loan</td>
      <td>In XX/XX/XXXX I filled out the Fedlaon applica...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Student loan</td>
      <td>I am being contacted by a debt collector for p...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Student loan</td>
      <td>I cosigned XXXX student loans at SallieMae for...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Student loan</td>
      <td>Navient has sytematically and illegally failed...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Student loan</td>
      <td>My wife became eligible for XXXX Loan Forgiven...</td>
    </tr>
  </tbody>
</table>
</div>



## Preprocessing Overview

Before you begin to practice some of your new tools regarding regularization and optimization, let's practice munging some data as you did in the previous section with bank complaints. Recall some techniques:

* Sampling in order to reduce training time (investigate model accuracy vs data size later on)
* One-hot encoding your complaint text
* Transforming your category labels
* Train - test split

## Preprocessing: Generate a Random Sample

Since you have quite a bit of data and training networks takes a substantial amount of time and resources, downsample in order to test your initial pipeline. Going forward, these can be interesting areas of investigation: how does your models performance change as you increase (or decrease) the size of your dataset?  

Generate the random sample using seed 123 for consistency of results. Make your new sample have 10,000 observations.


```python
#Your code here
random.seed(123)
df = df.sample(10000)
df.index = range(10000)
product = df["Product"]
complaints = df["Consumer complaint narrative"]
```

## Preprocessing: One-hot Encoding of the Complaints

As before, you need to do some preprocessing and data manipulationg before building the neural network. 

Keep the 2,000 most common words and use one-hot encoding to reformat the complaints into a matrix of vectors.


```python
#Your code here; use one-hot encoding to reformat the complaints into a matrix of vectors.
#Only keep the 2000 most common words.

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(complaints)

one_hot_results= tokenizer.texts_to_matrix(complaints, mode='binary')
word_index = tokenizer.word_index
np.shape(one_hot_results)
```




    (10000, 2000)



## Preprocessing: Encoding the Products

Similarly, now transform the descriptive product labels to integers labels. After transforming them to integer labels, retransform them into a matrix of binary flags, one for each of the various product labels.  
  
> **Note**: This is similar to your previous work with dummy variables. Each of the various product categories will be its own column, and each observation will be a row. In turn, each of these observation rows will have a 1 in the column associated with it's label, and all other entries for the row will be zero.


```python
#Your code here; transform the product labels to numerical values
le = preprocessing.LabelEncoder()
le.fit(product)
product_cat = le.transform(product) 

#Then transform these integer values into a matrix of binary flags
product_onehot = to_categorical(product_cat)
```

## Train-test Split

Now onto the ever familiar train-test split! 
Below, perform an appropriate train test split.
> Be sure to split both the complaint data (now transformed into word vectors) as well as their associated labels. 


```python
X_train, X_test, y_train, y_test = train_test_split(one_hot_results, product_onehot, test_size=1500, random_state=42)

#Alternative custom script:
# random.seed(123)
# test_index = random.sample(range(1,10000), 1500)
# test = one_hot_results[test_index]
# train = np.delete(one_hot_results, test_index, 0)
# label_test = product_onehot[test_index]
# label_train = np.delete(product_onehot, test_index, 0)
```

## Running the model using a validation set.

## Creating the Validation Set

In the lecture, you saw that in deep learning, you generally set aside a validation set, which is then used during hyperparameter tuning. Afterwards, when you have decided upon a final model, the test can then be used to define the final model perforance. 

In this example, take the first 1000 cases out of the training set to create a validation set. You should do this for both `train` and `label_train`.


```python
random.seed(123)
val = X_train[:1000]
train_final = X_train[1000:]
label_val = y_train[:1000]
label_train_final = y_train[1000:]
```

## Creating the Model

Rebuild a fully connected (Dense) layer network with relu activations in Keras.

Recall that you used 2 hidden with 50 units in the first layer and 25 in the second, both with a `relu` activation function. Because you are dealing with a multiclass problem (classifying the complaints into 7 classes), use a softmax classifyer in order to output 7 class probabilities per case. 


```python
#Your code here; build a neural network using Keras as described above.
random.seed(123)
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
```

## Compiling the Model
In the compiler, you'll be passing the optimizer, loss function, and metrics. Train the model for 120 epochs in mini-batches of 256 samples. This time, include the argument `validation_data` and assign it `(val, label_val)`


```python
#Your code here
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Training the Model

Ok, now for the resource intensive part: time to train your model! Note that this is where you also introduce the validation data to the model.


```python
model_val = model.fit(train_final,
                    label_train_final,
                    epochs=120,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/120
    7500/7500 [==============================] - 1s 71us/step - loss: 1.9372 - acc: 0.1848 - val_loss: 1.9234 - val_acc: 0.2070
    Epoch 2/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9128 - acc: 0.2172 - val_loss: 1.9009 - val_acc: 0.2180
    Epoch 3/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8898 - acc: 0.2355 - val_loss: 1.8783 - val_acc: 0.2390
    Epoch 4/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8655 - acc: 0.2589 - val_loss: 1.8528 - val_acc: 0.2640
    Epoch 5/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.8377 - acc: 0.2865 - val_loss: 1.8239 - val_acc: 0.2790
    Epoch 6/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8057 - acc: 0.3081 - val_loss: 1.7894 - val_acc: 0.3090
    Epoch 7/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7687 - acc: 0.3404 - val_loss: 1.7493 - val_acc: 0.3580
    Epoch 8/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7269 - acc: 0.3715 - val_loss: 1.7046 - val_acc: 0.3880
    Epoch 9/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6816 - acc: 0.4024 - val_loss: 1.6564 - val_acc: 0.4020
    Epoch 10/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6329 - acc: 0.4280 - val_loss: 1.6056 - val_acc: 0.4350
    Epoch 11/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5815 - acc: 0.4576 - val_loss: 1.5529 - val_acc: 0.4640
    Epoch 12/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5291 - acc: 0.4861 - val_loss: 1.4988 - val_acc: 0.4930
    Epoch 13/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4766 - acc: 0.5100 - val_loss: 1.4474 - val_acc: 0.5060
    Epoch 14/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.4251 - acc: 0.5315 - val_loss: 1.3938 - val_acc: 0.5480
    Epoch 15/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3746 - acc: 0.5513 - val_loss: 1.3445 - val_acc: 0.5510
    Epoch 16/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3265 - acc: 0.5660 - val_loss: 1.2958 - val_acc: 0.5730
    Epoch 17/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2798 - acc: 0.5799 - val_loss: 1.2489 - val_acc: 0.5960
    Epoch 18/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2349 - acc: 0.5987 - val_loss: 1.2040 - val_acc: 0.6090
    Epoch 19/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1924 - acc: 0.6107 - val_loss: 1.1630 - val_acc: 0.6200
    Epoch 20/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1520 - acc: 0.6309 - val_loss: 1.1233 - val_acc: 0.6410
    Epoch 21/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1139 - acc: 0.6456 - val_loss: 1.0871 - val_acc: 0.6450
    Epoch 22/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0782 - acc: 0.6617 - val_loss: 1.0537 - val_acc: 0.6690
    Epoch 23/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0444 - acc: 0.6736 - val_loss: 1.0235 - val_acc: 0.6680
    Epoch 24/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0134 - acc: 0.6791 - val_loss: 0.9938 - val_acc: 0.6770
    Epoch 25/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9842 - acc: 0.6896 - val_loss: 0.9656 - val_acc: 0.6810
    Epoch 26/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9566 - acc: 0.6947 - val_loss: 0.9383 - val_acc: 0.6930
    Epoch 27/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9310 - acc: 0.7033 - val_loss: 0.9152 - val_acc: 0.6940
    Epoch 28/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9068 - acc: 0.7080 - val_loss: 0.8941 - val_acc: 0.7050
    Epoch 29/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8847 - acc: 0.7148 - val_loss: 0.8731 - val_acc: 0.7070
    Epoch 30/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8638 - acc: 0.7221 - val_loss: 0.8551 - val_acc: 0.7120
    Epoch 31/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8440 - acc: 0.7269 - val_loss: 0.8384 - val_acc: 0.7190
    Epoch 32/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8260 - acc: 0.7303 - val_loss: 0.8205 - val_acc: 0.7190
    Epoch 33/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8083 - acc: 0.7336 - val_loss: 0.8048 - val_acc: 0.7240
    Epoch 34/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7918 - acc: 0.7380 - val_loss: 0.7903 - val_acc: 0.7360
    Epoch 35/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7765 - acc: 0.7428 - val_loss: 0.7772 - val_acc: 0.7320
    Epoch 36/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7618 - acc: 0.7464 - val_loss: 0.7656 - val_acc: 0.7430
    Epoch 37/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.7483 - acc: 0.7465 - val_loss: 0.7549 - val_acc: 0.7430
    Epoch 38/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7354 - acc: 0.7524 - val_loss: 0.7444 - val_acc: 0.7350
    Epoch 39/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.7230 - acc: 0.7537 - val_loss: 0.7326 - val_acc: 0.7460
    Epoch 40/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.7109 - acc: 0.7557 - val_loss: 0.7218 - val_acc: 0.7470
    Epoch 41/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.7001 - acc: 0.7591 - val_loss: 0.7142 - val_acc: 0.7500
    Epoch 42/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6895 - acc: 0.7619 - val_loss: 0.7057 - val_acc: 0.7540
    Epoch 43/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6797 - acc: 0.7677 - val_loss: 0.6970 - val_acc: 0.7500
    Epoch 44/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6702 - acc: 0.7663 - val_loss: 0.6891 - val_acc: 0.7570
    Epoch 45/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6608 - acc: 0.7712 - val_loss: 0.6835 - val_acc: 0.7540
    Epoch 46/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.6518 - acc: 0.7724 - val_loss: 0.6824 - val_acc: 0.7570
    Epoch 47/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6433 - acc: 0.7748 - val_loss: 0.6710 - val_acc: 0.7630
    Epoch 48/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6353 - acc: 0.7775 - val_loss: 0.6642 - val_acc: 0.7620
    Epoch 49/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6269 - acc: 0.7812 - val_loss: 0.6593 - val_acc: 0.7610
    Epoch 50/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6197 - acc: 0.7855 - val_loss: 0.6546 - val_acc: 0.7620
    Epoch 51/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.6125 - acc: 0.7880 - val_loss: 0.6490 - val_acc: 0.7620
    Epoch 52/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6052 - acc: 0.7897 - val_loss: 0.6444 - val_acc: 0.7690
    Epoch 53/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5981 - acc: 0.7919 - val_loss: 0.6391 - val_acc: 0.7630
    Epoch 54/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5909 - acc: 0.7952 - val_loss: 0.6347 - val_acc: 0.7650
    Epoch 55/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.5846 - acc: 0.7959 - val_loss: 0.6344 - val_acc: 0.7650
    Epoch 56/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.5786 - acc: 0.7995 - val_loss: 0.6285 - val_acc: 0.7650
    Epoch 57/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5719 - acc: 0.8032 - val_loss: 0.6258 - val_acc: 0.7730
    Epoch 58/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5663 - acc: 0.8025 - val_loss: 0.6218 - val_acc: 0.7750
    Epoch 59/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5606 - acc: 0.8077 - val_loss: 0.6164 - val_acc: 0.7710
    Epoch 60/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5544 - acc: 0.8083 - val_loss: 0.6147 - val_acc: 0.7640
    Epoch 61/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5493 - acc: 0.8107 - val_loss: 0.6109 - val_acc: 0.7780
    Epoch 62/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5432 - acc: 0.8121 - val_loss: 0.6064 - val_acc: 0.7730
    Epoch 63/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.5378 - acc: 0.8135 - val_loss: 0.6056 - val_acc: 0.7770
    Epoch 64/120
    7500/7500 [==============================] - 0s 20us/step - loss: 0.5329 - acc: 0.8153 - val_loss: 0.6031 - val_acc: 0.7750
    Epoch 65/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5276 - acc: 0.8183 - val_loss: 0.6000 - val_acc: 0.7760
    Epoch 66/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5222 - acc: 0.8209 - val_loss: 0.6003 - val_acc: 0.7730
    Epoch 67/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5177 - acc: 0.8203 - val_loss: 0.5945 - val_acc: 0.7750
    Epoch 68/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5122 - acc: 0.8221 - val_loss: 0.5925 - val_acc: 0.7830
    Epoch 69/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5081 - acc: 0.8243 - val_loss: 0.5901 - val_acc: 0.7730
    Epoch 70/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5029 - acc: 0.8256 - val_loss: 0.5890 - val_acc: 0.7750
    Epoch 71/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4987 - acc: 0.8289 - val_loss: 0.5869 - val_acc: 0.7760
    Epoch 72/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4939 - acc: 0.8295 - val_loss: 0.5846 - val_acc: 0.7730
    Epoch 73/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4899 - acc: 0.8328 - val_loss: 0.5844 - val_acc: 0.7780
    Epoch 74/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4847 - acc: 0.8333 - val_loss: 0.5821 - val_acc: 0.7750
    Epoch 75/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4806 - acc: 0.8339 - val_loss: 0.5811 - val_acc: 0.7770
    Epoch 76/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4767 - acc: 0.8368 - val_loss: 0.5789 - val_acc: 0.7780
    Epoch 77/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4721 - acc: 0.8380 - val_loss: 0.5770 - val_acc: 0.7770
    Epoch 78/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4681 - acc: 0.8403 - val_loss: 0.5753 - val_acc: 0.7770
    Epoch 79/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4640 - acc: 0.8421 - val_loss: 0.5730 - val_acc: 0.7770
    Epoch 80/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4598 - acc: 0.8429 - val_loss: 0.5721 - val_acc: 0.7750
    Epoch 81/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.4560 - acc: 0.8448 - val_loss: 0.5694 - val_acc: 0.7820
    Epoch 82/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.4520 - acc: 0.8456 - val_loss: 0.5699 - val_acc: 0.7750
    Epoch 83/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4479 - acc: 0.8477 - val_loss: 0.5677 - val_acc: 0.7800
    Epoch 84/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.4442 - acc: 0.8496 - val_loss: 0.5663 - val_acc: 0.7800
    Epoch 85/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4405 - acc: 0.8500 - val_loss: 0.5657 - val_acc: 0.7850
    Epoch 86/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4368 - acc: 0.8525 - val_loss: 0.5658 - val_acc: 0.7800
    Epoch 87/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4337 - acc: 0.8563 - val_loss: 0.5640 - val_acc: 0.7830
    Epoch 88/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4296 - acc: 0.8549 - val_loss: 0.5632 - val_acc: 0.7770
    Epoch 89/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4259 - acc: 0.8572 - val_loss: 0.5620 - val_acc: 0.7800
    Epoch 90/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4224 - acc: 0.8575 - val_loss: 0.5602 - val_acc: 0.7820
    Epoch 91/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4188 - acc: 0.8613 - val_loss: 0.5594 - val_acc: 0.7830
    Epoch 92/120
    7500/7500 [==============================] - 0s 22us/step - loss: 0.4156 - acc: 0.8637 - val_loss: 0.5608 - val_acc: 0.7890
    Epoch 93/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4121 - acc: 0.8603 - val_loss: 0.5592 - val_acc: 0.7760
    Epoch 94/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4083 - acc: 0.8640 - val_loss: 0.5587 - val_acc: 0.7860
    Epoch 95/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.4056 - acc: 0.8663 - val_loss: 0.5585 - val_acc: 0.7790
    Epoch 96/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.4022 - acc: 0.8659 - val_loss: 0.5564 - val_acc: 0.7850
    Epoch 97/120
    7500/7500 [==============================] - 0s 20us/step - loss: 0.3989 - acc: 0.8676 - val_loss: 0.5569 - val_acc: 0.7860
    Epoch 98/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.3952 - acc: 0.8705 - val_loss: 0.5563 - val_acc: 0.7810
    Epoch 99/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.3923 - acc: 0.8709 - val_loss: 0.5570 - val_acc: 0.7810
    Epoch 100/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.3890 - acc: 0.8737 - val_loss: 0.5551 - val_acc: 0.7870
    Epoch 101/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.3857 - acc: 0.8748 - val_loss: 0.5569 - val_acc: 0.7810
    Epoch 102/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3833 - acc: 0.8743 - val_loss: 0.5548 - val_acc: 0.7810
    Epoch 103/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3802 - acc: 0.8764 - val_loss: 0.5518 - val_acc: 0.7860
    Epoch 104/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3767 - acc: 0.8788 - val_loss: 0.5539 - val_acc: 0.7850
    Epoch 105/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3737 - acc: 0.8803 - val_loss: 0.5522 - val_acc: 0.7840
    Epoch 106/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3710 - acc: 0.8803 - val_loss: 0.5518 - val_acc: 0.7820
    Epoch 107/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3677 - acc: 0.8844 - val_loss: 0.5560 - val_acc: 0.7860
    Epoch 108/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3652 - acc: 0.8829 - val_loss: 0.5510 - val_acc: 0.7860
    Epoch 109/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3623 - acc: 0.8859 - val_loss: 0.5557 - val_acc: 0.7880
    Epoch 110/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.3593 - acc: 0.8865 - val_loss: 0.5500 - val_acc: 0.7830
    Epoch 111/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3568 - acc: 0.8884 - val_loss: 0.5503 - val_acc: 0.7910
    Epoch 112/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3541 - acc: 0.8876 - val_loss: 0.5511 - val_acc: 0.7840
    Epoch 113/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3508 - acc: 0.8871 - val_loss: 0.5482 - val_acc: 0.7850
    Epoch 114/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3484 - acc: 0.8920 - val_loss: 0.5492 - val_acc: 0.7860
    Epoch 115/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.3457 - acc: 0.8919 - val_loss: 0.5492 - val_acc: 0.7880
    Epoch 116/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.3431 - acc: 0.8925 - val_loss: 0.5486 - val_acc: 0.7850
    Epoch 117/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.3403 - acc: 0.8944 - val_loss: 0.5481 - val_acc: 0.7860
    Epoch 118/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.3379 - acc: 0.8948 - val_loss: 0.5505 - val_acc: 0.7840
    Epoch 119/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.3351 - acc: 0.8968 - val_loss: 0.5505 - val_acc: 0.7890
    Epoch 120/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.3328 - acc: 0.8963 - val_loss: 0.5524 - val_acc: 0.7920


## Retrieving Performance Results: the `history` dictionary

The dictionary `history` contains four entries this time: one per metric that was being monitored during training and during validation.


```python
model_val_dict = model_val.history
model_val_dict.keys()
```




    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])




```python
results_train = model.evaluate(train_final, label_train_final)
```

    7500/7500 [==============================] - 0s 18us/step



```python
results_test = model.evaluate(X_test, y_test)
```

    1500/1500 [==============================] - 0s 27us/step



```python
results_train
```




    [0.33027180240948995, 0.8962666666666667]




```python
results_test
```




    [0.7113916211128235, 0.7339999998410542]



Note that the result isn't exactly the same as before. Note that this because the training set is slightly different! you remove 1000 instances for validation!

## Plotting the Results

Plot the loss function versus the number of epochs. Be sure to include the training and the validation loss in the same plot. Then, create a second plot comparing training and validation accuracy to the number of epochs.


```python
plt.clf()

import matplotlib.pyplot as plt
loss_values = model_val_dict['loss']
val_loss_values = model_val_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'g', label='Training loss')
plt.plot(epochs, val_loss_values, 'blue', label='Validation loss')

plt.title('Training & validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


![png](index_files/index_36_0.png)



```python
plt.clf()

acc_values = model_val_dict['acc'] 
val_acc_values = model_val_dict['val_acc']

plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.plot(epochs, val_acc_values, 'blue', label='Validation acc')
plt.title('Training & validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


![png](index_files/index_37_0.png)


Notice an interesting pattern here: although the training accuracy keeps increasing when going through more epochs, and the training loss keeps decreasing, the validation accuracy and loss seem to be reaching a limit around the 60th epoch. This means that you're probably **overfitting** the model to the training data when you train for many epochs past this dropoff point of around 40 epochs. Luckily, you learned how to tackle overfitting in the previous lecture! Since it seems clear that you are training too long, include early stopping at the 60th epoch first.

## Early Stopping

Below, observe how to update the model to include an earlier cutoff point:


```python
random.seed(123)
model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

final_model = model.fit(train_final,
                    label_train_final,
                    epochs=60,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/60
    7500/7500 [==============================] - 0s 39us/step - loss: 1.9643 - acc: 0.1275 - val_loss: 1.9566 - val_acc: 0.1230
    Epoch 2/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.9425 - acc: 0.1461 - val_loss: 1.9419 - val_acc: 0.1460
    Epoch 3/60
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9294 - acc: 0.1709 - val_loss: 1.9312 - val_acc: 0.1570
    Epoch 4/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.9183 - acc: 0.1907 - val_loss: 1.9213 - val_acc: 0.1690
    Epoch 5/60
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9075 - acc: 0.2140 - val_loss: 1.9109 - val_acc: 0.1900
    Epoch 6/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8960 - acc: 0.2352 - val_loss: 1.8993 - val_acc: 0.2320
    Epoch 7/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8829 - acc: 0.2600 - val_loss: 1.8857 - val_acc: 0.2600
    Epoch 8/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8676 - acc: 0.2880 - val_loss: 1.8698 - val_acc: 0.2770
    Epoch 9/60
    7500/7500 [==============================] - 0s 19us/step - loss: 1.8498 - acc: 0.3117 - val_loss: 1.8517 - val_acc: 0.2970
    Epoch 10/60
    7500/7500 [==============================] - 0s 19us/step - loss: 1.8299 - acc: 0.3317 - val_loss: 1.8320 - val_acc: 0.3220
    Epoch 11/60
    7500/7500 [==============================] - 0s 21us/step - loss: 1.8079 - acc: 0.3533 - val_loss: 1.8098 - val_acc: 0.3280
    Epoch 12/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7835 - acc: 0.3693 - val_loss: 1.7855 - val_acc: 0.3520
    Epoch 13/60
    7500/7500 [==============================] - 0s 20us/step - loss: 1.7565 - acc: 0.3920 - val_loss: 1.7581 - val_acc: 0.3830
    Epoch 14/60
    7500/7500 [==============================] - 0s 19us/step - loss: 1.7262 - acc: 0.4172 - val_loss: 1.7274 - val_acc: 0.4060
    Epoch 15/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6929 - acc: 0.4455 - val_loss: 1.6936 - val_acc: 0.4280
    Epoch 16/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6563 - acc: 0.4703 - val_loss: 1.6565 - val_acc: 0.4600
    Epoch 17/60
    7500/7500 [==============================] - 0s 20us/step - loss: 1.6169 - acc: 0.4988 - val_loss: 1.6162 - val_acc: 0.4950
    Epoch 18/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5745 - acc: 0.5223 - val_loss: 1.5726 - val_acc: 0.5280
    Epoch 19/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5297 - acc: 0.5444 - val_loss: 1.5269 - val_acc: 0.5570
    Epoch 20/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.4828 - acc: 0.5755 - val_loss: 1.4794 - val_acc: 0.5670
    Epoch 21/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4347 - acc: 0.5912 - val_loss: 1.4304 - val_acc: 0.5790
    Epoch 22/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3856 - acc: 0.6039 - val_loss: 1.3816 - val_acc: 0.6030
    Epoch 23/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3366 - acc: 0.6264 - val_loss: 1.3331 - val_acc: 0.6230
    Epoch 24/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2883 - acc: 0.6363 - val_loss: 1.2861 - val_acc: 0.6440
    Epoch 25/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2413 - acc: 0.6555 - val_loss: 1.2412 - val_acc: 0.6470
    Epoch 26/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1959 - acc: 0.6651 - val_loss: 1.1965 - val_acc: 0.6630
    Epoch 27/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1526 - acc: 0.6767 - val_loss: 1.1553 - val_acc: 0.6690
    Epoch 28/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1115 - acc: 0.6859 - val_loss: 1.1147 - val_acc: 0.6800
    Epoch 29/60
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0723 - acc: 0.6961 - val_loss: 1.0784 - val_acc: 0.6780
    Epoch 30/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0358 - acc: 0.7029 - val_loss: 1.0428 - val_acc: 0.6980
    Epoch 31/60
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0016 - acc: 0.7072 - val_loss: 1.0103 - val_acc: 0.7030
    Epoch 32/60
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9698 - acc: 0.7129 - val_loss: 0.9798 - val_acc: 0.7020
    Epoch 33/60
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9399 - acc: 0.7188 - val_loss: 0.9518 - val_acc: 0.7180
    Epoch 34/60
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9123 - acc: 0.7244 - val_loss: 0.9262 - val_acc: 0.7190
    Epoch 35/60
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8859 - acc: 0.7305 - val_loss: 0.9038 - val_acc: 0.7270
    Epoch 36/60
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8626 - acc: 0.7299 - val_loss: 0.8822 - val_acc: 0.7230
    Epoch 37/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8404 - acc: 0.7348 - val_loss: 0.8600 - val_acc: 0.7240
    Epoch 38/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8195 - acc: 0.7405 - val_loss: 0.8416 - val_acc: 0.7290
    Epoch 39/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8008 - acc: 0.7449 - val_loss: 0.8247 - val_acc: 0.7320
    Epoch 40/60
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7827 - acc: 0.7479 - val_loss: 0.8091 - val_acc: 0.7280
    Epoch 41/60
    7500/7500 [==============================] - 0s 17us/step - loss: 0.7654 - acc: 0.7539 - val_loss: 0.7939 - val_acc: 0.7380
    Epoch 42/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7501 - acc: 0.7583 - val_loss: 0.7796 - val_acc: 0.7350
    Epoch 43/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7357 - acc: 0.7599 - val_loss: 0.7679 - val_acc: 0.7310
    Epoch 44/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.7217 - acc: 0.7624 - val_loss: 0.7577 - val_acc: 0.7390
    Epoch 45/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7089 - acc: 0.7692 - val_loss: 0.7457 - val_acc: 0.7390
    Epoch 46/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6967 - acc: 0.7691 - val_loss: 0.7380 - val_acc: 0.7390
    Epoch 47/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6850 - acc: 0.7724 - val_loss: 0.7283 - val_acc: 0.7410
    Epoch 48/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6742 - acc: 0.7756 - val_loss: 0.7180 - val_acc: 0.7410
    Epoch 49/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6635 - acc: 0.7803 - val_loss: 0.7113 - val_acc: 0.7420
    Epoch 50/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6539 - acc: 0.7800 - val_loss: 0.7015 - val_acc: 0.7440
    Epoch 51/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6445 - acc: 0.7852 - val_loss: 0.6955 - val_acc: 0.7490
    Epoch 52/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6358 - acc: 0.7856 - val_loss: 0.6876 - val_acc: 0.7460
    Epoch 53/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6269 - acc: 0.7909 - val_loss: 0.6823 - val_acc: 0.7510
    Epoch 54/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6187 - acc: 0.7929 - val_loss: 0.6751 - val_acc: 0.7500
    Epoch 55/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6104 - acc: 0.7947 - val_loss: 0.6710 - val_acc: 0.7500
    Epoch 56/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.6031 - acc: 0.7976 - val_loss: 0.6638 - val_acc: 0.7570
    Epoch 57/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.5954 - acc: 0.7997 - val_loss: 0.6609 - val_acc: 0.7530
    Epoch 58/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.5885 - acc: 0.7996 - val_loss: 0.6554 - val_acc: 0.7560
    Epoch 59/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.5816 - acc: 0.8035 - val_loss: 0.6545 - val_acc: 0.7490
    Epoch 60/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.5751 - acc: 0.8056 - val_loss: 0.6451 - val_acc: 0.7570


Now, you can use the test set to make label predictions


```python
results_train = model.evaluate(train_final, label_train_final)
```

    7500/7500 [==============================] - 0s 22us/step



```python
results_test = model.evaluate(X_test, y_test)
```

    1500/1500 [==============================] - 0s 30us/step



```python
results_train
```




    [0.5689828497727712, 0.8097333333651224]




```python
results_test
```




    [0.7319343857765198, 0.7146666668256124]



We've significantly reduced the variance, so this is already pretty good! your test set accuracy is slightly worse, but this model will definitely be more robust than the 120 epochs model you originally fit.

Now, take a look at how regularization techniques can further improve your model performance.

## L2 Regularization

First, take a look at L2 regularization. Keras makes L2 regularization easy. Simply add the `kernel_regularizer=kernel_regulizers.l2(lamda_coeff)` parameter to any model layer. The lambda_coeff parameter determines the strength of the regularization you wish to perform.


```python
from keras import regularizers
random.seed(123)
model = models.Sequential()
model.add(layers.Dense(50, activation='relu',kernel_regularizer=regularizers.l2(0.005), input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, kernel_regularizer=regularizers.l2(0.005), activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

L2_model = model.fit(train_final,
                    label_train_final,
                    epochs=120,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/120
    7500/7500 [==============================] - 0s 49us/step - loss: 2.5877 - acc: 0.1883 - val_loss: 2.5762 - val_acc: 0.2080
    Epoch 2/120
    7500/7500 [==============================] - 0s 21us/step - loss: 2.5691 - acc: 0.2079 - val_loss: 2.5575 - val_acc: 0.2190
    Epoch 3/120
    7500/7500 [==============================] - 0s 20us/step - loss: 2.5497 - acc: 0.2324 - val_loss: 2.5375 - val_acc: 0.2280
    Epoch 4/120
    7500/7500 [==============================] - 0s 21us/step - loss: 2.5287 - acc: 0.2564 - val_loss: 2.5153 - val_acc: 0.2640
    Epoch 5/120
    7500/7500 [==============================] - 0s 19us/step - loss: 2.5055 - acc: 0.2739 - val_loss: 2.4901 - val_acc: 0.2880
    Epoch 6/120
    7500/7500 [==============================] - 0s 20us/step - loss: 2.4792 - acc: 0.2927 - val_loss: 2.4610 - val_acc: 0.3030
    Epoch 7/120
    7500/7500 [==============================] - 0s 20us/step - loss: 2.4494 - acc: 0.3100 - val_loss: 2.4278 - val_acc: 0.3190
    Epoch 8/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.4154 - acc: 0.3272 - val_loss: 2.3914 - val_acc: 0.3300
    Epoch 9/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.3778 - acc: 0.3483 - val_loss: 2.3507 - val_acc: 0.3560
    Epoch 10/120
    7500/7500 [==============================] - 0s 18us/step - loss: 2.3363 - acc: 0.3733 - val_loss: 2.3073 - val_acc: 0.3840
    Epoch 11/120
    7500/7500 [==============================] - 0s 20us/step - loss: 2.2913 - acc: 0.3968 - val_loss: 2.2606 - val_acc: 0.4050
    Epoch 12/120
    7500/7500 [==============================] - 0s 20us/step - loss: 2.2433 - acc: 0.4241 - val_loss: 2.2111 - val_acc: 0.4270
    Epoch 13/120
    7500/7500 [==============================] - 0s 19us/step - loss: 2.1925 - acc: 0.4504 - val_loss: 2.1588 - val_acc: 0.4490
    Epoch 14/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.1396 - acc: 0.4716 - val_loss: 2.1051 - val_acc: 0.4790
    Epoch 15/120
    7500/7500 [==============================] - 0s 19us/step - loss: 2.0857 - acc: 0.5059 - val_loss: 2.0512 - val_acc: 0.4980
    Epoch 16/120
    7500/7500 [==============================] - 0s 18us/step - loss: 2.0318 - acc: 0.5285 - val_loss: 1.9977 - val_acc: 0.5160
    Epoch 17/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.9781 - acc: 0.5541 - val_loss: 1.9446 - val_acc: 0.5420
    Epoch 18/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9257 - acc: 0.5752 - val_loss: 1.8930 - val_acc: 0.5700
    Epoch 19/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8746 - acc: 0.6016 - val_loss: 1.8445 - val_acc: 0.5980
    Epoch 20/120
    7500/7500 [==============================] - 0s 21us/step - loss: 1.8259 - acc: 0.6216 - val_loss: 1.7989 - val_acc: 0.6050
    Epoch 21/120
    7500/7500 [==============================] - 0s 23us/step - loss: 1.7803 - acc: 0.6367 - val_loss: 1.7554 - val_acc: 0.6220
    Epoch 22/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.7368 - acc: 0.6564 - val_loss: 1.7150 - val_acc: 0.6480
    Epoch 23/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.6960 - acc: 0.6733 - val_loss: 1.6757 - val_acc: 0.6580
    Epoch 24/120
    7500/7500 [==============================] - 0s 26us/step - loss: 1.6581 - acc: 0.6849 - val_loss: 1.6401 - val_acc: 0.6680
    Epoch 25/120
    7500/7500 [==============================] - 0s 22us/step - loss: 1.6223 - acc: 0.6947 - val_loss: 1.6071 - val_acc: 0.6800
    Epoch 26/120
    7500/7500 [==============================] - 0s 22us/step - loss: 1.5888 - acc: 0.7085 - val_loss: 1.5777 - val_acc: 0.6930
    Epoch 27/120
    7500/7500 [==============================] - 0s 24us/step - loss: 1.5584 - acc: 0.7153 - val_loss: 1.5495 - val_acc: 0.6820
    Epoch 28/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5295 - acc: 0.7213 - val_loss: 1.5218 - val_acc: 0.7000
    Epoch 29/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.5027 - acc: 0.7275 - val_loss: 1.4988 - val_acc: 0.7120
    Epoch 30/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.4778 - acc: 0.7319 - val_loss: 1.4762 - val_acc: 0.7120
    Epoch 31/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.4545 - acc: 0.7369 - val_loss: 1.4543 - val_acc: 0.7190
    Epoch 32/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4324 - acc: 0.7389 - val_loss: 1.4340 - val_acc: 0.7200
    Epoch 33/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4119 - acc: 0.7439 - val_loss: 1.4166 - val_acc: 0.7270
    Epoch 34/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3928 - acc: 0.7495 - val_loss: 1.3999 - val_acc: 0.7310
    Epoch 35/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3745 - acc: 0.7535 - val_loss: 1.3842 - val_acc: 0.7340
    Epoch 36/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3582 - acc: 0.7564 - val_loss: 1.3701 - val_acc: 0.7330
    Epoch 37/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3421 - acc: 0.7589 - val_loss: 1.3554 - val_acc: 0.7380
    Epoch 38/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3268 - acc: 0.7624 - val_loss: 1.3423 - val_acc: 0.7340
    Epoch 39/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3127 - acc: 0.7655 - val_loss: 1.3302 - val_acc: 0.7370
    Epoch 40/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2993 - acc: 0.7677 - val_loss: 1.3206 - val_acc: 0.7330
    Epoch 41/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.2863 - acc: 0.7717 - val_loss: 1.3094 - val_acc: 0.7430
    Epoch 42/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2741 - acc: 0.7715 - val_loss: 1.2968 - val_acc: 0.7420
    Epoch 43/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2623 - acc: 0.7755 - val_loss: 1.2885 - val_acc: 0.7410
    Epoch 44/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2507 - acc: 0.7772 - val_loss: 1.2778 - val_acc: 0.7490
    Epoch 45/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.2402 - acc: 0.7795 - val_loss: 1.2701 - val_acc: 0.7470
    Epoch 46/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2296 - acc: 0.7811 - val_loss: 1.2616 - val_acc: 0.7490
    Epoch 47/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2197 - acc: 0.7844 - val_loss: 1.2539 - val_acc: 0.7480
    Epoch 48/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2102 - acc: 0.7867 - val_loss: 1.2453 - val_acc: 0.7560
    Epoch 49/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2009 - acc: 0.7881 - val_loss: 1.2399 - val_acc: 0.7490
    Epoch 50/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.1921 - acc: 0.7909 - val_loss: 1.2310 - val_acc: 0.7580
    Epoch 51/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1832 - acc: 0.7947 - val_loss: 1.2243 - val_acc: 0.7570
    Epoch 52/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.1747 - acc: 0.7944 - val_loss: 1.2181 - val_acc: 0.7620
    Epoch 53/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.1663 - acc: 0.7968 - val_loss: 1.2124 - val_acc: 0.7540
    Epoch 54/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.1583 - acc: 0.8011 - val_loss: 1.2060 - val_acc: 0.7610
    Epoch 55/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1505 - acc: 0.8027 - val_loss: 1.2006 - val_acc: 0.7600
    Epoch 56/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1428 - acc: 0.8032 - val_loss: 1.1948 - val_acc: 0.7640
    Epoch 57/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1354 - acc: 0.8073 - val_loss: 1.1891 - val_acc: 0.7650
    Epoch 58/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.1286 - acc: 0.8081 - val_loss: 1.1848 - val_acc: 0.7650
    Epoch 59/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1209 - acc: 0.8097 - val_loss: 1.1798 - val_acc: 0.7630
    Epoch 60/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.1139 - acc: 0.8119 - val_loss: 1.1741 - val_acc: 0.7630
    Epoch 61/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1071 - acc: 0.8152 - val_loss: 1.1685 - val_acc: 0.7670
    Epoch 62/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.1006 - acc: 0.8169 - val_loss: 1.1638 - val_acc: 0.7610
    Epoch 63/120
    7500/7500 [==============================] - 0s 24us/step - loss: 1.0937 - acc: 0.8177 - val_loss: 1.1594 - val_acc: 0.7640
    Epoch 64/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.0873 - acc: 0.8184 - val_loss: 1.1559 - val_acc: 0.7610
    Epoch 65/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0811 - acc: 0.8188 - val_loss: 1.1529 - val_acc: 0.7670
    Epoch 66/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0748 - acc: 0.8220 - val_loss: 1.1477 - val_acc: 0.7660
    Epoch 67/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.0686 - acc: 0.8229 - val_loss: 1.1428 - val_acc: 0.7620
    Epoch 68/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0627 - acc: 0.8247 - val_loss: 1.1396 - val_acc: 0.7640
    Epoch 69/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0570 - acc: 0.8264 - val_loss: 1.1332 - val_acc: 0.7660
    Epoch 70/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0512 - acc: 0.8264 - val_loss: 1.1306 - val_acc: 0.7670
    Epoch 71/120
    7500/7500 [==============================] - 0s 20us/step - loss: 1.0450 - acc: 0.8272 - val_loss: 1.1253 - val_acc: 0.7640
    Epoch 72/120
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0395 - acc: 0.8316 - val_loss: 1.1234 - val_acc: 0.7680
    Epoch 73/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0343 - acc: 0.8317 - val_loss: 1.1196 - val_acc: 0.7710
    Epoch 74/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0285 - acc: 0.8336 - val_loss: 1.1161 - val_acc: 0.7690
    Epoch 75/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0229 - acc: 0.8367 - val_loss: 1.1110 - val_acc: 0.7720
    Epoch 76/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0178 - acc: 0.8376 - val_loss: 1.1078 - val_acc: 0.7710
    Epoch 77/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0124 - acc: 0.8411 - val_loss: 1.1067 - val_acc: 0.7710
    Epoch 78/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0072 - acc: 0.8404 - val_loss: 1.1016 - val_acc: 0.7710
    Epoch 79/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0023 - acc: 0.8403 - val_loss: 1.0983 - val_acc: 0.7750
    Epoch 80/120
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9968 - acc: 0.8425 - val_loss: 1.0955 - val_acc: 0.7710
    Epoch 81/120
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9920 - acc: 0.8444 - val_loss: 1.0937 - val_acc: 0.7700
    Epoch 82/120
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9872 - acc: 0.8444 - val_loss: 1.0904 - val_acc: 0.7690
    Epoch 83/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9819 - acc: 0.8469 - val_loss: 1.0852 - val_acc: 0.7750
    Epoch 84/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9772 - acc: 0.8476 - val_loss: 1.0827 - val_acc: 0.7710
    Epoch 85/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9724 - acc: 0.8497 - val_loss: 1.0796 - val_acc: 0.7710
    Epoch 86/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9675 - acc: 0.8485 - val_loss: 1.0761 - val_acc: 0.7730
    Epoch 87/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9629 - acc: 0.8521 - val_loss: 1.0747 - val_acc: 0.7720
    Epoch 88/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9588 - acc: 0.8513 - val_loss: 1.0722 - val_acc: 0.7750
    Epoch 89/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9539 - acc: 0.8520 - val_loss: 1.0690 - val_acc: 0.7710
    Epoch 90/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9489 - acc: 0.8525 - val_loss: 1.0660 - val_acc: 0.7750
    Epoch 91/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9448 - acc: 0.8568 - val_loss: 1.0652 - val_acc: 0.7700
    Epoch 92/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9404 - acc: 0.8559 - val_loss: 1.0611 - val_acc: 0.7750
    Epoch 93/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9360 - acc: 0.8595 - val_loss: 1.0617 - val_acc: 0.7800
    Epoch 94/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9317 - acc: 0.8569 - val_loss: 1.0566 - val_acc: 0.7740
    Epoch 95/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9273 - acc: 0.8620 - val_loss: 1.0534 - val_acc: 0.7800
    Epoch 96/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9234 - acc: 0.8611 - val_loss: 1.0532 - val_acc: 0.7760
    Epoch 97/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9192 - acc: 0.8612 - val_loss: 1.0493 - val_acc: 0.7800
    Epoch 98/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9151 - acc: 0.8635 - val_loss: 1.0460 - val_acc: 0.7840
    Epoch 99/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9107 - acc: 0.8661 - val_loss: 1.0462 - val_acc: 0.7760
    Epoch 100/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9067 - acc: 0.8671 - val_loss: 1.0416 - val_acc: 0.7830
    Epoch 101/120
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9029 - acc: 0.8676 - val_loss: 1.0392 - val_acc: 0.7850
    Epoch 102/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8986 - acc: 0.8696 - val_loss: 1.0370 - val_acc: 0.7830
    Epoch 103/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8947 - acc: 0.8696 - val_loss: 1.0355 - val_acc: 0.7810
    Epoch 104/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8912 - acc: 0.8692 - val_loss: 1.0322 - val_acc: 0.7820
    Epoch 105/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8868 - acc: 0.8724 - val_loss: 1.0303 - val_acc: 0.7830
    Epoch 106/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8832 - acc: 0.8732 - val_loss: 1.0301 - val_acc: 0.7750
    Epoch 107/120
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8794 - acc: 0.8720 - val_loss: 1.0262 - val_acc: 0.7840
    Epoch 108/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8754 - acc: 0.8741 - val_loss: 1.0251 - val_acc: 0.7860
    Epoch 109/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8719 - acc: 0.8772 - val_loss: 1.0247 - val_acc: 0.7840
    Epoch 110/120
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8685 - acc: 0.8755 - val_loss: 1.0204 - val_acc: 0.7860
    Epoch 111/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8642 - acc: 0.8764 - val_loss: 1.0191 - val_acc: 0.7790
    Epoch 112/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8605 - acc: 0.8777 - val_loss: 1.0182 - val_acc: 0.7830
    Epoch 113/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8573 - acc: 0.8787 - val_loss: 1.0144 - val_acc: 0.7860
    Epoch 114/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8536 - acc: 0.8799 - val_loss: 1.0146 - val_acc: 0.7780
    Epoch 115/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8503 - acc: 0.8817 - val_loss: 1.0124 - val_acc: 0.7810
    Epoch 116/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8466 - acc: 0.8820 - val_loss: 1.0082 - val_acc: 0.7850
    Epoch 117/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8428 - acc: 0.8812 - val_loss: 1.0076 - val_acc: 0.7860
    Epoch 118/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8400 - acc: 0.8832 - val_loss: 1.0052 - val_acc: 0.7840
    Epoch 119/120
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8361 - acc: 0.8844 - val_loss: 1.0061 - val_acc: 0.7840
    Epoch 120/120
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8326 - acc: 0.8848 - val_loss: 1.0022 - val_acc: 0.7830



```python
L2_model_dict = L2_model.history
L2_model_dict.keys()
```




    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])



Now, look at the training accuracy as well as the validation accuracy for both the L2 and the model without regularization (for 120 epochs).


```python
plt.clf()

acc_values = L2_model_dict['acc'] 
val_acc_values = L2_model_dict['val_acc']
model_acc = model_val_dict['acc']
model_val_acc = model_val_dict['val_acc']

epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'g', label='Training acc L2')
plt.plot(epochs, val_acc_values, 'g', label='Validation acc L2')
plt.plot(epochs, model_acc, 'r', label='Training acc')
plt.plot(epochs, model_val_acc, 'r', label='Validation acc')
plt.title('Training & validation accuracy L2 vs regular')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


![png](index_files/index_52_0.png)


The results of L2 regularization are quite disappointing here. Notice the discrepancy between validation and training accuracy seems to have decreased slightly, but the end result is definitely not getting better. 

## L1 Regularization

Have a look at L1 regularization. Will this work better?


```python
random.seed(123)
model = models.Sequential()
model.add(layers.Dense(50, activation='relu',kernel_regularizer=regularizers.l1(0.005), input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, kernel_regularizer=regularizers.l1(0.005), activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

L1_model = model.fit(train_final,
                    label_train_final,
                    epochs=120,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/120
    7500/7500 [==============================] - 0s 53us/step - loss: 15.9745 - acc: 0.1556 - val_loss: 15.5532 - val_acc: 0.2140
    Epoch 2/120
    7500/7500 [==============================] - 0s 17us/step - loss: 15.2106 - acc: 0.1984 - val_loss: 14.8059 - val_acc: 0.2450
    Epoch 3/120
    7500/7500 [==============================] - 0s 17us/step - loss: 14.4736 - acc: 0.2201 - val_loss: 14.0836 - val_acc: 0.2480
    Epoch 4/120
    7500/7500 [==============================] - 0s 16us/step - loss: 13.7598 - acc: 0.2349 - val_loss: 13.3820 - val_acc: 0.2570
    Epoch 5/120
    7500/7500 [==============================] - 0s 19us/step - loss: 13.0662 - acc: 0.2596 - val_loss: 12.6999 - val_acc: 0.2720
    Epoch 6/120
    7500/7500 [==============================] - 0s 16us/step - loss: 12.3921 - acc: 0.2869 - val_loss: 12.0366 - val_acc: 0.3020
    Epoch 7/120
    7500/7500 [==============================] - 0s 17us/step - loss: 11.7375 - acc: 0.3177 - val_loss: 11.3926 - val_acc: 0.3280
    Epoch 8/120
    7500/7500 [==============================] - 0s 16us/step - loss: 11.1022 - acc: 0.3473 - val_loss: 10.7682 - val_acc: 0.3490
    Epoch 9/120
    7500/7500 [==============================] - 0s 16us/step - loss: 10.4869 - acc: 0.3772 - val_loss: 10.1632 - val_acc: 0.3820
    Epoch 10/120
    7500/7500 [==============================] - 0s 17us/step - loss: 9.8901 - acc: 0.3980 - val_loss: 9.5764 - val_acc: 0.4140
    Epoch 11/120
    7500/7500 [==============================] - 0s 18us/step - loss: 9.3120 - acc: 0.4251 - val_loss: 9.0100 - val_acc: 0.4340
    Epoch 12/120
    7500/7500 [==============================] - 0s 16us/step - loss: 8.7557 - acc: 0.4520 - val_loss: 8.4658 - val_acc: 0.4370
    Epoch 13/120
    7500/7500 [==============================] - 0s 17us/step - loss: 8.2216 - acc: 0.4669 - val_loss: 7.9433 - val_acc: 0.4670
    Epoch 14/120
    7500/7500 [==============================] - 0s 16us/step - loss: 7.7097 - acc: 0.4907 - val_loss: 7.4435 - val_acc: 0.4900
    Epoch 15/120
    7500/7500 [==============================] - 0s 17us/step - loss: 7.2203 - acc: 0.5073 - val_loss: 6.9665 - val_acc: 0.5200
    Epoch 16/120
    7500/7500 [==============================] - 0s 16us/step - loss: 6.7537 - acc: 0.5272 - val_loss: 6.5129 - val_acc: 0.5420
    Epoch 17/120
    7500/7500 [==============================] - 0s 17us/step - loss: 6.3106 - acc: 0.5464 - val_loss: 6.0820 - val_acc: 0.5510
    Epoch 18/120
    7500/7500 [==============================] - 0s 17us/step - loss: 5.8906 - acc: 0.5601 - val_loss: 5.6740 - val_acc: 0.5550
    Epoch 19/120
    7500/7500 [==============================] - 0s 17us/step - loss: 5.4929 - acc: 0.5724 - val_loss: 5.2884 - val_acc: 0.5630
    Epoch 20/120
    7500/7500 [==============================] - 0s 17us/step - loss: 5.1184 - acc: 0.5849 - val_loss: 4.9263 - val_acc: 0.5750
    Epoch 21/120
    7500/7500 [==============================] - 0s 18us/step - loss: 4.7667 - acc: 0.5973 - val_loss: 4.5873 - val_acc: 0.5840
    Epoch 22/120
    7500/7500 [==============================] - 0s 17us/step - loss: 4.4379 - acc: 0.6039 - val_loss: 4.2699 - val_acc: 0.5750
    Epoch 23/120
    7500/7500 [==============================] - 0s 18us/step - loss: 4.1317 - acc: 0.6067 - val_loss: 3.9752 - val_acc: 0.6020
    Epoch 24/120
    7500/7500 [==============================] - 0s 17us/step - loss: 3.8486 - acc: 0.6133 - val_loss: 3.7034 - val_acc: 0.6080
    Epoch 25/120
    7500/7500 [==============================] - 0s 18us/step - loss: 3.5871 - acc: 0.6187 - val_loss: 3.4535 - val_acc: 0.6060
    Epoch 26/120
    7500/7500 [==============================] - 0s 18us/step - loss: 3.3472 - acc: 0.6213 - val_loss: 3.2241 - val_acc: 0.6020
    Epoch 27/120
    7500/7500 [==============================] - 0s 18us/step - loss: 3.1293 - acc: 0.6225 - val_loss: 3.0172 - val_acc: 0.6070
    Epoch 28/120
    7500/7500 [==============================] - 0s 18us/step - loss: 2.9327 - acc: 0.6244 - val_loss: 2.8325 - val_acc: 0.6140
    Epoch 29/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.7575 - acc: 0.6288 - val_loss: 2.6677 - val_acc: 0.6040
    Epoch 30/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.6030 - acc: 0.6272 - val_loss: 2.5236 - val_acc: 0.6100
    Epoch 31/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.4688 - acc: 0.6265 - val_loss: 2.3992 - val_acc: 0.6210
    Epoch 32/120
    7500/7500 [==============================] - 0s 18us/step - loss: 2.3545 - acc: 0.6309 - val_loss: 2.2984 - val_acc: 0.6120
    Epoch 33/120
    7500/7500 [==============================] - 0s 18us/step - loss: 2.2602 - acc: 0.6300 - val_loss: 2.2109 - val_acc: 0.6080
    Epoch 34/120
    7500/7500 [==============================] - 0s 17us/step - loss: 2.1843 - acc: 0.6292 - val_loss: 2.1445 - val_acc: 0.6300
    Epoch 35/120
    7500/7500 [==============================] - 0s 18us/step - loss: 2.1258 - acc: 0.6300 - val_loss: 2.0918 - val_acc: 0.6260
    Epoch 36/120
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0818 - acc: 0.6305 - val_loss: 2.0550 - val_acc: 0.6190
    Epoch 37/120
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0489 - acc: 0.6312 - val_loss: 2.0253 - val_acc: 0.6310
    Epoch 38/120
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0228 - acc: 0.6319 - val_loss: 2.0022 - val_acc: 0.6180
    Epoch 39/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9998 - acc: 0.6331 - val_loss: 1.9782 - val_acc: 0.6360
    Epoch 40/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9784 - acc: 0.6319 - val_loss: 1.9587 - val_acc: 0.6350
    Epoch 41/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.9590 - acc: 0.6333 - val_loss: 1.9397 - val_acc: 0.6260
    Epoch 42/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9410 - acc: 0.6324 - val_loss: 1.9207 - val_acc: 0.6370
    Epoch 43/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9237 - acc: 0.6347 - val_loss: 1.9056 - val_acc: 0.6420
    Epoch 44/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9075 - acc: 0.6359 - val_loss: 1.8885 - val_acc: 0.6400
    Epoch 45/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8919 - acc: 0.6373 - val_loss: 1.8739 - val_acc: 0.6380
    Epoch 46/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8766 - acc: 0.6371 - val_loss: 1.8577 - val_acc: 0.6410
    Epoch 47/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8622 - acc: 0.6371 - val_loss: 1.8447 - val_acc: 0.6410
    Epoch 48/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8480 - acc: 0.6388 - val_loss: 1.8294 - val_acc: 0.6460
    Epoch 49/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8346 - acc: 0.6385 - val_loss: 1.8205 - val_acc: 0.6480
    Epoch 50/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8219 - acc: 0.6411 - val_loss: 1.8043 - val_acc: 0.6540
    Epoch 51/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8087 - acc: 0.6412 - val_loss: 1.7925 - val_acc: 0.6570
    Epoch 52/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7964 - acc: 0.6451 - val_loss: 1.7778 - val_acc: 0.6570
    Epoch 53/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7843 - acc: 0.6500 - val_loss: 1.7667 - val_acc: 0.6550
    Epoch 54/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7725 - acc: 0.6513 - val_loss: 1.7559 - val_acc: 0.6600
    Epoch 55/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7609 - acc: 0.6563 - val_loss: 1.7440 - val_acc: 0.6590
    Epoch 56/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7507 - acc: 0.6604 - val_loss: 1.7344 - val_acc: 0.6560
    Epoch 57/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7391 - acc: 0.6620 - val_loss: 1.7224 - val_acc: 0.6600
    Epoch 58/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7282 - acc: 0.6672 - val_loss: 1.7104 - val_acc: 0.6660
    Epoch 59/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.7183 - acc: 0.6692 - val_loss: 1.6997 - val_acc: 0.6720
    Epoch 60/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7081 - acc: 0.6760 - val_loss: 1.6922 - val_acc: 0.6700
    Epoch 61/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6985 - acc: 0.6773 - val_loss: 1.6809 - val_acc: 0.6810
    Epoch 62/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6881 - acc: 0.6783 - val_loss: 1.6719 - val_acc: 0.6680
    Epoch 63/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6784 - acc: 0.6832 - val_loss: 1.6623 - val_acc: 0.6850
    Epoch 64/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6692 - acc: 0.6828 - val_loss: 1.6520 - val_acc: 0.6750
    Epoch 65/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6591 - acc: 0.6852 - val_loss: 1.6435 - val_acc: 0.6850
    Epoch 66/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6499 - acc: 0.6883 - val_loss: 1.6342 - val_acc: 0.6850
    Epoch 67/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6412 - acc: 0.6860 - val_loss: 1.6251 - val_acc: 0.6860
    Epoch 68/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6317 - acc: 0.6904 - val_loss: 1.6140 - val_acc: 0.6900
    Epoch 69/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6231 - acc: 0.6937 - val_loss: 1.6061 - val_acc: 0.6880
    Epoch 70/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6144 - acc: 0.6936 - val_loss: 1.5976 - val_acc: 0.6850
    Epoch 71/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6060 - acc: 0.6975 - val_loss: 1.5906 - val_acc: 0.6940
    Epoch 72/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5974 - acc: 0.6955 - val_loss: 1.5813 - val_acc: 0.7010
    Epoch 73/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5889 - acc: 0.6995 - val_loss: 1.5760 - val_acc: 0.6920
    Epoch 74/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5809 - acc: 0.7004 - val_loss: 1.5651 - val_acc: 0.7020
    Epoch 75/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5725 - acc: 0.7051 - val_loss: 1.5550 - val_acc: 0.7000
    Epoch 76/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5647 - acc: 0.7047 - val_loss: 1.5468 - val_acc: 0.7000
    Epoch 77/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5571 - acc: 0.7040 - val_loss: 1.5405 - val_acc: 0.6960
    Epoch 78/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5493 - acc: 0.7049 - val_loss: 1.5328 - val_acc: 0.6990
    Epoch 79/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5423 - acc: 0.7068 - val_loss: 1.5262 - val_acc: 0.6940
    Epoch 80/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5348 - acc: 0.7076 - val_loss: 1.5196 - val_acc: 0.6970
    Epoch 81/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5274 - acc: 0.7077 - val_loss: 1.5128 - val_acc: 0.7020
    Epoch 82/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5201 - acc: 0.7083 - val_loss: 1.5035 - val_acc: 0.7030
    Epoch 83/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5131 - acc: 0.7076 - val_loss: 1.4976 - val_acc: 0.7050
    Epoch 84/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5061 - acc: 0.7112 - val_loss: 1.4932 - val_acc: 0.7040
    Epoch 85/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4993 - acc: 0.7095 - val_loss: 1.4819 - val_acc: 0.7050
    Epoch 86/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.4922 - acc: 0.7093 - val_loss: 1.4779 - val_acc: 0.7070
    Epoch 87/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4850 - acc: 0.7125 - val_loss: 1.4723 - val_acc: 0.7010
    Epoch 88/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4793 - acc: 0.7099 - val_loss: 1.4629 - val_acc: 0.7000
    Epoch 89/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4725 - acc: 0.7108 - val_loss: 1.4634 - val_acc: 0.7070
    Epoch 90/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4656 - acc: 0.7109 - val_loss: 1.4517 - val_acc: 0.7080
    Epoch 91/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4592 - acc: 0.7125 - val_loss: 1.4458 - val_acc: 0.7070
    Epoch 92/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4526 - acc: 0.7127 - val_loss: 1.4400 - val_acc: 0.7080
    Epoch 93/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4465 - acc: 0.7128 - val_loss: 1.4311 - val_acc: 0.7050
    Epoch 94/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4403 - acc: 0.7128 - val_loss: 1.4251 - val_acc: 0.7090
    Epoch 95/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4339 - acc: 0.7160 - val_loss: 1.4182 - val_acc: 0.7060
    Epoch 96/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4272 - acc: 0.7137 - val_loss: 1.4128 - val_acc: 0.7110
    Epoch 97/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4210 - acc: 0.7149 - val_loss: 1.4082 - val_acc: 0.7070
    Epoch 98/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4157 - acc: 0.7155 - val_loss: 1.4060 - val_acc: 0.7120
    Epoch 99/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4099 - acc: 0.7151 - val_loss: 1.3942 - val_acc: 0.7140
    Epoch 100/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4033 - acc: 0.7164 - val_loss: 1.3888 - val_acc: 0.7090
    Epoch 101/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3972 - acc: 0.7156 - val_loss: 1.3827 - val_acc: 0.7120
    Epoch 102/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3914 - acc: 0.7161 - val_loss: 1.3809 - val_acc: 0.7060
    Epoch 103/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3864 - acc: 0.7165 - val_loss: 1.3715 - val_acc: 0.7150
    Epoch 104/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3805 - acc: 0.7159 - val_loss: 1.3712 - val_acc: 0.7120
    Epoch 105/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3750 - acc: 0.7161 - val_loss: 1.3623 - val_acc: 0.7130
    Epoch 106/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3689 - acc: 0.7179 - val_loss: 1.3684 - val_acc: 0.7100
    Epoch 107/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3651 - acc: 0.7176 - val_loss: 1.3544 - val_acc: 0.7120
    Epoch 108/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3587 - acc: 0.7179 - val_loss: 1.3462 - val_acc: 0.7120
    Epoch 109/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3530 - acc: 0.7196 - val_loss: 1.3425 - val_acc: 0.7120
    Epoch 110/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3479 - acc: 0.7183 - val_loss: 1.3401 - val_acc: 0.7140
    Epoch 111/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3430 - acc: 0.7189 - val_loss: 1.3285 - val_acc: 0.7180
    Epoch 112/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3379 - acc: 0.7195 - val_loss: 1.3245 - val_acc: 0.7180
    Epoch 113/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3326 - acc: 0.7205 - val_loss: 1.3204 - val_acc: 0.7190
    Epoch 114/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3276 - acc: 0.7201 - val_loss: 1.3151 - val_acc: 0.7170
    Epoch 115/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3220 - acc: 0.7212 - val_loss: 1.3114 - val_acc: 0.7140
    Epoch 116/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3171 - acc: 0.7217 - val_loss: 1.3068 - val_acc: 0.7180
    Epoch 117/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3127 - acc: 0.7209 - val_loss: 1.2996 - val_acc: 0.7160
    Epoch 118/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3076 - acc: 0.7220 - val_loss: 1.2981 - val_acc: 0.7170
    Epoch 119/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3028 - acc: 0.7239 - val_loss: 1.2927 - val_acc: 0.7200
    Epoch 120/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2984 - acc: 0.7217 - val_loss: 1.2882 - val_acc: 0.7170



```python
L1_model_dict = L1_model.history
plt.clf()

acc_values = L1_model_dict['acc'] 
val_acc_values = L1_model_dict['val_acc']

epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'g', label='Training acc L1')
plt.plot(epochs, val_acc_values, 'g.', label='Validation acc L1')
plt.title('Training & validation accuracy with L1 regularization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


![png](index_files/index_57_0.png)


Notice how the training and validation accuracy don't diverge as much as before. Unfortunately, the validation accuracy doesn't reach rates much higher than 70%. It does seem like you can still improve the model by training much longer.


```python
# ⏰ This cell may take several minutes to run
random.seed(123)
model = models.Sequential()
model.add(layers.Dense(50, activation='relu',kernel_regularizer=regularizers.l1(0.005), input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, kernel_regularizer=regularizers.l1(0.005), activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

L1_model = model.fit(train_final,
                    label_train_final,
                    epochs=1000,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/1000
    7500/7500 [==============================] - 0s 55us/step - loss: 16.0140 - acc: 0.1877 - val_loss: 15.6118 - val_acc: 0.2100
    Epoch 2/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 15.2533 - acc: 0.2067 - val_loss: 14.8668 - val_acc: 0.2240
    Epoch 3/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 14.5171 - acc: 0.2219 - val_loss: 14.1433 - val_acc: 0.2290
    Epoch 4/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 13.8011 - acc: 0.2360 - val_loss: 13.4388 - val_acc: 0.2440
    Epoch 5/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 13.1038 - acc: 0.2528 - val_loss: 12.7534 - val_acc: 0.2580
    Epoch 6/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 12.4253 - acc: 0.2772 - val_loss: 12.0869 - val_acc: 0.2720
    Epoch 7/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 11.7658 - acc: 0.3027 - val_loss: 11.4387 - val_acc: 0.3140
    Epoch 8/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 11.1259 - acc: 0.3431 - val_loss: 10.8103 - val_acc: 0.3470
    Epoch 9/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 10.5062 - acc: 0.3781 - val_loss: 10.2028 - val_acc: 0.3780
    Epoch 10/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 9.9068 - acc: 0.4124 - val_loss: 9.6137 - val_acc: 0.3970
    Epoch 11/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 9.3279 - acc: 0.4441 - val_loss: 9.0463 - val_acc: 0.4310
    Epoch 12/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 8.7708 - acc: 0.4668 - val_loss: 8.5007 - val_acc: 0.4610
    Epoch 13/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 8.2357 - acc: 0.4971 - val_loss: 7.9774 - val_acc: 0.4730
    Epoch 14/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 7.7237 - acc: 0.5213 - val_loss: 7.4776 - val_acc: 0.4990
    Epoch 15/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 7.2345 - acc: 0.5405 - val_loss: 7.0011 - val_acc: 0.5270
    Epoch 16/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 6.7686 - acc: 0.5620 - val_loss: 6.5474 - val_acc: 0.5490
    Epoch 17/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 6.3256 - acc: 0.5787 - val_loss: 6.1182 - val_acc: 0.5860
    Epoch 18/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 5.9053 - acc: 0.5941 - val_loss: 5.7067 - val_acc: 0.5940
    Epoch 19/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 5.5074 - acc: 0.6119 - val_loss: 5.3218 - val_acc: 0.5940
    Epoch 20/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 5.1324 - acc: 0.6183 - val_loss: 4.9565 - val_acc: 0.6200
    Epoch 21/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 4.7803 - acc: 0.6311 - val_loss: 4.6160 - val_acc: 0.6160
    Epoch 22/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 4.4521 - acc: 0.6364 - val_loss: 4.2978 - val_acc: 0.6320
    Epoch 23/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 4.1454 - acc: 0.6468 - val_loss: 4.0029 - val_acc: 0.6420
    Epoch 24/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 3.8615 - acc: 0.6531 - val_loss: 3.7284 - val_acc: 0.6590
    Epoch 25/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 3.5996 - acc: 0.6593 - val_loss: 3.4779 - val_acc: 0.6590
    Epoch 26/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 3.3602 - acc: 0.6624 - val_loss: 3.2500 - val_acc: 0.6740
    Epoch 27/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 3.1425 - acc: 0.6635 - val_loss: 3.0417 - val_acc: 0.6780
    Epoch 28/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 2.9463 - acc: 0.6691 - val_loss: 2.8538 - val_acc: 0.6790
    Epoch 29/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 2.7710 - acc: 0.6711 - val_loss: 2.6896 - val_acc: 0.6810
    Epoch 30/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 2.6167 - acc: 0.6717 - val_loss: 2.5434 - val_acc: 0.6810
    Epoch 31/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 2.4825 - acc: 0.6703 - val_loss: 2.4194 - val_acc: 0.6810
    Epoch 32/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 2.3687 - acc: 0.6731 - val_loss: 2.3143 - val_acc: 0.6780
    Epoch 33/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 2.2740 - acc: 0.6723 - val_loss: 2.2270 - val_acc: 0.6820
    Epoch 34/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 2.1980 - acc: 0.6713 - val_loss: 2.1602 - val_acc: 0.6810
    Epoch 35/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 2.1386 - acc: 0.6725 - val_loss: 2.1087 - val_acc: 0.6800
    Epoch 36/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 2.0942 - acc: 0.6736 - val_loss: 2.0704 - val_acc: 0.6760
    Epoch 37/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0616 - acc: 0.6712 - val_loss: 2.0393 - val_acc: 0.6720
    Epoch 38/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0343 - acc: 0.6735 - val_loss: 2.0143 - val_acc: 0.6710
    Epoch 39/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0114 - acc: 0.6720 - val_loss: 1.9902 - val_acc: 0.6790
    Epoch 40/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9903 - acc: 0.6745 - val_loss: 1.9708 - val_acc: 0.6780
    Epoch 41/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.9712 - acc: 0.6747 - val_loss: 1.9519 - val_acc: 0.6830
    Epoch 42/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.9536 - acc: 0.6751 - val_loss: 1.9332 - val_acc: 0.6780
    Epoch 43/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9369 - acc: 0.6753 - val_loss: 1.9168 - val_acc: 0.6820
    Epoch 44/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.9213 - acc: 0.6769 - val_loss: 1.9012 - val_acc: 0.6770
    Epoch 45/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9061 - acc: 0.6791 - val_loss: 1.8871 - val_acc: 0.6790
    Epoch 46/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8920 - acc: 0.6776 - val_loss: 1.8731 - val_acc: 0.6830
    Epoch 47/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8783 - acc: 0.6800 - val_loss: 1.8585 - val_acc: 0.6820
    Epoch 48/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8650 - acc: 0.6796 - val_loss: 1.8430 - val_acc: 0.6850
    Epoch 49/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8519 - acc: 0.6815 - val_loss: 1.8301 - val_acc: 0.6900
    Epoch 50/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8395 - acc: 0.6824 - val_loss: 1.8186 - val_acc: 0.6910
    Epoch 51/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8277 - acc: 0.6844 - val_loss: 1.8058 - val_acc: 0.6900
    Epoch 52/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8157 - acc: 0.6853 - val_loss: 1.7949 - val_acc: 0.6930
    Epoch 53/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 1.8043 - acc: 0.6844 - val_loss: 1.7821 - val_acc: 0.6950
    Epoch 54/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.7931 - acc: 0.6857 - val_loss: 1.7722 - val_acc: 0.6970
    Epoch 55/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7828 - acc: 0.6863 - val_loss: 1.7612 - val_acc: 0.6880
    Epoch 56/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7722 - acc: 0.6876 - val_loss: 1.7494 - val_acc: 0.6940
    Epoch 57/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7616 - acc: 0.6877 - val_loss: 1.7391 - val_acc: 0.6920
    Epoch 58/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7517 - acc: 0.6887 - val_loss: 1.7376 - val_acc: 0.6940
    Epoch 59/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7419 - acc: 0.6892 - val_loss: 1.7217 - val_acc: 0.6940
    Epoch 60/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7319 - acc: 0.6887 - val_loss: 1.7170 - val_acc: 0.6940
    Epoch 61/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7227 - acc: 0.6892 - val_loss: 1.7054 - val_acc: 0.6970
    Epoch 62/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7136 - acc: 0.6903 - val_loss: 1.6913 - val_acc: 0.6980
    Epoch 63/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7041 - acc: 0.6924 - val_loss: 1.6814 - val_acc: 0.6940
    Epoch 64/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6951 - acc: 0.6928 - val_loss: 1.6735 - val_acc: 0.7040
    Epoch 65/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6864 - acc: 0.6944 - val_loss: 1.6635 - val_acc: 0.7020
    Epoch 66/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6778 - acc: 0.6957 - val_loss: 1.6551 - val_acc: 0.7050
    Epoch 67/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6691 - acc: 0.6960 - val_loss: 1.6475 - val_acc: 0.7070
    Epoch 68/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6604 - acc: 0.6987 - val_loss: 1.6402 - val_acc: 0.7010
    Epoch 69/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6518 - acc: 0.6996 - val_loss: 1.6361 - val_acc: 0.7010
    Epoch 70/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6436 - acc: 0.6987 - val_loss: 1.6220 - val_acc: 0.7000
    Epoch 71/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6354 - acc: 0.7015 - val_loss: 1.6156 - val_acc: 0.7080
    Epoch 72/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6274 - acc: 0.7015 - val_loss: 1.6048 - val_acc: 0.7010
    Epoch 73/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6198 - acc: 0.7019 - val_loss: 1.6047 - val_acc: 0.7080
    Epoch 74/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6122 - acc: 0.7016 - val_loss: 1.5891 - val_acc: 0.7060
    Epoch 75/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6044 - acc: 0.7020 - val_loss: 1.5813 - val_acc: 0.7070
    Epoch 76/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5962 - acc: 0.7048 - val_loss: 1.5755 - val_acc: 0.7060
    Epoch 77/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5886 - acc: 0.7039 - val_loss: 1.5679 - val_acc: 0.7060
    Epoch 78/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5816 - acc: 0.7048 - val_loss: 1.5609 - val_acc: 0.7080
    Epoch 79/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5739 - acc: 0.7051 - val_loss: 1.5524 - val_acc: 0.7100
    Epoch 80/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5670 - acc: 0.7047 - val_loss: 1.5454 - val_acc: 0.7080
    Epoch 81/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5596 - acc: 0.7049 - val_loss: 1.5392 - val_acc: 0.7070
    Epoch 82/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5523 - acc: 0.7080 - val_loss: 1.5329 - val_acc: 0.7170
    Epoch 83/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5455 - acc: 0.7063 - val_loss: 1.5254 - val_acc: 0.7150
    Epoch 84/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5384 - acc: 0.7087 - val_loss: 1.5181 - val_acc: 0.7130
    Epoch 85/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5314 - acc: 0.7088 - val_loss: 1.5132 - val_acc: 0.7110
    Epoch 86/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5249 - acc: 0.7084 - val_loss: 1.5042 - val_acc: 0.7140
    Epoch 87/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5176 - acc: 0.7093 - val_loss: 1.5022 - val_acc: 0.7070
    Epoch 88/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5115 - acc: 0.7108 - val_loss: 1.4923 - val_acc: 0.7130
    Epoch 89/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5045 - acc: 0.7117 - val_loss: 1.4951 - val_acc: 0.7090
    Epoch 90/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4983 - acc: 0.7115 - val_loss: 1.4801 - val_acc: 0.7140
    Epoch 91/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4912 - acc: 0.7127 - val_loss: 1.4728 - val_acc: 0.7140
    Epoch 92/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4852 - acc: 0.7104 - val_loss: 1.4667 - val_acc: 0.7170
    Epoch 93/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4784 - acc: 0.7136 - val_loss: 1.4594 - val_acc: 0.7080
    Epoch 94/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4724 - acc: 0.7137 - val_loss: 1.4551 - val_acc: 0.7140
    Epoch 95/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4658 - acc: 0.7149 - val_loss: 1.4498 - val_acc: 0.7160
    Epoch 96/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4602 - acc: 0.7151 - val_loss: 1.4421 - val_acc: 0.7170
    Epoch 97/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4538 - acc: 0.7163 - val_loss: 1.4346 - val_acc: 0.7170
    Epoch 98/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.4470 - acc: 0.7149 - val_loss: 1.4340 - val_acc: 0.7200
    Epoch 99/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4414 - acc: 0.7152 - val_loss: 1.4272 - val_acc: 0.7180
    Epoch 100/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4362 - acc: 0.7188 - val_loss: 1.4224 - val_acc: 0.7150
    Epoch 101/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4301 - acc: 0.7153 - val_loss: 1.4136 - val_acc: 0.7190
    Epoch 102/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4246 - acc: 0.7169 - val_loss: 1.4046 - val_acc: 0.7200
    Epoch 103/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4182 - acc: 0.7196 - val_loss: 1.4068 - val_acc: 0.7190
    Epoch 104/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 1.4132 - acc: 0.7200 - val_loss: 1.3998 - val_acc: 0.7190
    Epoch 105/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 1.4072 - acc: 0.7176 - val_loss: 1.3891 - val_acc: 0.7180
    Epoch 106/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4016 - acc: 0.7204 - val_loss: 1.3859 - val_acc: 0.7220
    Epoch 107/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3966 - acc: 0.7208 - val_loss: 1.3825 - val_acc: 0.7210
    Epoch 108/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3906 - acc: 0.7225 - val_loss: 1.3804 - val_acc: 0.7230
    Epoch 109/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3854 - acc: 0.7219 - val_loss: 1.3692 - val_acc: 0.7220
    Epoch 110/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3802 - acc: 0.7215 - val_loss: 1.3651 - val_acc: 0.7200
    Epoch 111/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3750 - acc: 0.7201 - val_loss: 1.3593 - val_acc: 0.7210
    Epoch 112/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3703 - acc: 0.7241 - val_loss: 1.3563 - val_acc: 0.7220
    Epoch 113/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3650 - acc: 0.7241 - val_loss: 1.3489 - val_acc: 0.7250
    Epoch 114/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3596 - acc: 0.7249 - val_loss: 1.3460 - val_acc: 0.7250
    Epoch 115/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3543 - acc: 0.7255 - val_loss: 1.3375 - val_acc: 0.7250
    Epoch 116/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3497 - acc: 0.7248 - val_loss: 1.3340 - val_acc: 0.7270
    Epoch 117/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3450 - acc: 0.7249 - val_loss: 1.3339 - val_acc: 0.7220
    Epoch 118/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3404 - acc: 0.7247 - val_loss: 1.3267 - val_acc: 0.7210
    Epoch 119/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3356 - acc: 0.7267 - val_loss: 1.3237 - val_acc: 0.7160
    Epoch 120/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3309 - acc: 0.7299 - val_loss: 1.3153 - val_acc: 0.7290
    Epoch 121/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3262 - acc: 0.7273 - val_loss: 1.3085 - val_acc: 0.7270
    Epoch 122/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 1.3214 - acc: 0.7275 - val_loss: 1.3051 - val_acc: 0.7210
    Epoch 123/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 1.3167 - acc: 0.7279 - val_loss: 1.3032 - val_acc: 0.7250
    Epoch 124/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 1.3118 - acc: 0.7288 - val_loss: 1.2985 - val_acc: 0.7240
    Epoch 125/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3073 - acc: 0.7301 - val_loss: 1.2923 - val_acc: 0.7240
    Epoch 126/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.3030 - acc: 0.7299 - val_loss: 1.2887 - val_acc: 0.7270
    Epoch 127/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2984 - acc: 0.7301 - val_loss: 1.2880 - val_acc: 0.7230
    Epoch 128/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2947 - acc: 0.7303 - val_loss: 1.2849 - val_acc: 0.7180
    Epoch 129/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2899 - acc: 0.7319 - val_loss: 1.2752 - val_acc: 0.7220
    Epoch 130/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2858 - acc: 0.7301 - val_loss: 1.2707 - val_acc: 0.7260
    Epoch 131/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2819 - acc: 0.7319 - val_loss: 1.2682 - val_acc: 0.7240
    Epoch 132/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2771 - acc: 0.7319 - val_loss: 1.2628 - val_acc: 0.7270
    Epoch 133/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2734 - acc: 0.7331 - val_loss: 1.2590 - val_acc: 0.7290
    Epoch 134/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2691 - acc: 0.7335 - val_loss: 1.2582 - val_acc: 0.7250
    Epoch 135/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2655 - acc: 0.7348 - val_loss: 1.2505 - val_acc: 0.7250
    Epoch 136/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2614 - acc: 0.7351 - val_loss: 1.2478 - val_acc: 0.7320
    Epoch 137/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2575 - acc: 0.7325 - val_loss: 1.2434 - val_acc: 0.7270
    Epoch 138/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2538 - acc: 0.7355 - val_loss: 1.2403 - val_acc: 0.7290
    Epoch 139/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2497 - acc: 0.7355 - val_loss: 1.2362 - val_acc: 0.7290
    Epoch 140/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2456 - acc: 0.7341 - val_loss: 1.2344 - val_acc: 0.7310
    Epoch 141/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2418 - acc: 0.7356 - val_loss: 1.2292 - val_acc: 0.7350
    Epoch 142/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2380 - acc: 0.7364 - val_loss: 1.2267 - val_acc: 0.7320
    Epoch 143/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2344 - acc: 0.7365 - val_loss: 1.2225 - val_acc: 0.7280
    Epoch 144/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2306 - acc: 0.7371 - val_loss: 1.2170 - val_acc: 0.7280
    Epoch 145/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2272 - acc: 0.7349 - val_loss: 1.2162 - val_acc: 0.7320
    Epoch 146/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2233 - acc: 0.7385 - val_loss: 1.2114 - val_acc: 0.7290
    Epoch 147/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2199 - acc: 0.7385 - val_loss: 1.2098 - val_acc: 0.7310
    Epoch 148/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2166 - acc: 0.7379 - val_loss: 1.2034 - val_acc: 0.7340
    Epoch 149/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2128 - acc: 0.7393 - val_loss: 1.2085 - val_acc: 0.7230
    Epoch 150/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2096 - acc: 0.7389 - val_loss: 1.1977 - val_acc: 0.7310
    Epoch 151/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2066 - acc: 0.7391 - val_loss: 1.1958 - val_acc: 0.7320
    Epoch 152/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2030 - acc: 0.7389 - val_loss: 1.1909 - val_acc: 0.7340
    Epoch 153/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1992 - acc: 0.7396 - val_loss: 1.1875 - val_acc: 0.7350
    Epoch 154/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1969 - acc: 0.7380 - val_loss: 1.1904 - val_acc: 0.7290
    Epoch 155/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1932 - acc: 0.7395 - val_loss: 1.1811 - val_acc: 0.7340
    Epoch 156/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1896 - acc: 0.7405 - val_loss: 1.1802 - val_acc: 0.7370
    Epoch 157/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1861 - acc: 0.7408 - val_loss: 1.1754 - val_acc: 0.7300
    Epoch 158/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1834 - acc: 0.7409 - val_loss: 1.1755 - val_acc: 0.7320
    Epoch 159/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1803 - acc: 0.7425 - val_loss: 1.1680 - val_acc: 0.7360
    Epoch 160/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1766 - acc: 0.7424 - val_loss: 1.1751 - val_acc: 0.7280
    Epoch 161/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1752 - acc: 0.7417 - val_loss: 1.1609 - val_acc: 0.7340
    Epoch 162/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1716 - acc: 0.7423 - val_loss: 1.1612 - val_acc: 0.7360
    Epoch 163/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1687 - acc: 0.7409 - val_loss: 1.1594 - val_acc: 0.7370
    Epoch 164/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1658 - acc: 0.7441 - val_loss: 1.1571 - val_acc: 0.7350
    Epoch 165/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1629 - acc: 0.7419 - val_loss: 1.1572 - val_acc: 0.7390
    Epoch 166/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1598 - acc: 0.7429 - val_loss: 1.1513 - val_acc: 0.7430
    Epoch 167/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1577 - acc: 0.7429 - val_loss: 1.1462 - val_acc: 0.7390
    Epoch 168/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1546 - acc: 0.7427 - val_loss: 1.1453 - val_acc: 0.7410
    Epoch 169/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1517 - acc: 0.7431 - val_loss: 1.1406 - val_acc: 0.7410
    Epoch 170/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1490 - acc: 0.7461 - val_loss: 1.1447 - val_acc: 0.7300
    Epoch 171/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1470 - acc: 0.7456 - val_loss: 1.1377 - val_acc: 0.7410
    Epoch 172/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1438 - acc: 0.7455 - val_loss: 1.1356 - val_acc: 0.7440
    Epoch 173/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1415 - acc: 0.7457 - val_loss: 1.1324 - val_acc: 0.7390
    Epoch 174/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1394 - acc: 0.7433 - val_loss: 1.1291 - val_acc: 0.7370
    Epoch 175/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1362 - acc: 0.7455 - val_loss: 1.1279 - val_acc: 0.7440
    Epoch 176/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1352 - acc: 0.7449 - val_loss: 1.1235 - val_acc: 0.7370
    Epoch 177/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 1.1320 - acc: 0.7447 - val_loss: 1.1211 - val_acc: 0.7380
    Epoch 178/1000
    7500/7500 [==============================] - 0s 29us/step - loss: 1.1293 - acc: 0.7463 - val_loss: 1.1236 - val_acc: 0.7380
    Epoch 179/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.1279 - acc: 0.7457 - val_loss: 1.1186 - val_acc: 0.7400
    Epoch 180/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 1.1250 - acc: 0.7467 - val_loss: 1.1250 - val_acc: 0.7390
    Epoch 181/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 1.1234 - acc: 0.7436 - val_loss: 1.1210 - val_acc: 0.7360
    Epoch 182/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 1.1214 - acc: 0.7468 - val_loss: 1.1116 - val_acc: 0.7430
    Epoch 183/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 1.1188 - acc: 0.7488 - val_loss: 1.1138 - val_acc: 0.7380
    Epoch 184/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.1174 - acc: 0.7473 - val_loss: 1.1087 - val_acc: 0.7440
    Epoch 185/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 1.1145 - acc: 0.7479 - val_loss: 1.1078 - val_acc: 0.7410
    Epoch 186/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 1.1129 - acc: 0.7463 - val_loss: 1.1084 - val_acc: 0.7450
    Epoch 187/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 1.1110 - acc: 0.7476 - val_loss: 1.1017 - val_acc: 0.7440
    Epoch 188/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 1.1081 - acc: 0.7471 - val_loss: 1.1027 - val_acc: 0.7410
    Epoch 189/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 1.1065 - acc: 0.7489 - val_loss: 1.0996 - val_acc: 0.7380
    Epoch 190/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 1.1045 - acc: 0.7473 - val_loss: 1.0977 - val_acc: 0.7430
    Epoch 191/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.1028 - acc: 0.7496 - val_loss: 1.1026 - val_acc: 0.7370
    Epoch 192/1000
    7500/7500 [==============================] - 0s 34us/step - loss: 1.1010 - acc: 0.7475 - val_loss: 1.0954 - val_acc: 0.7370
    Epoch 193/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0997 - acc: 0.7484 - val_loss: 1.0904 - val_acc: 0.7450
    Epoch 194/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0972 - acc: 0.7500 - val_loss: 1.0884 - val_acc: 0.7410
    Epoch 195/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0953 - acc: 0.7503 - val_loss: 1.0877 - val_acc: 0.7400
    Epoch 196/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0937 - acc: 0.7480 - val_loss: 1.0867 - val_acc: 0.7390
    Epoch 197/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0922 - acc: 0.7485 - val_loss: 1.0866 - val_acc: 0.7430
    Epoch 198/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0896 - acc: 0.7497 - val_loss: 1.0832 - val_acc: 0.7440
    Epoch 199/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0879 - acc: 0.7495 - val_loss: 1.0827 - val_acc: 0.7440
    Epoch 200/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0863 - acc: 0.7511 - val_loss: 1.0830 - val_acc: 0.7430
    Epoch 201/1000
    7500/7500 [==============================] - 0s 31us/step - loss: 1.0848 - acc: 0.7515 - val_loss: 1.0847 - val_acc: 0.7440
    Epoch 202/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0835 - acc: 0.7508 - val_loss: 1.0759 - val_acc: 0.7420
    Epoch 203/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 1.0813 - acc: 0.7508 - val_loss: 1.0765 - val_acc: 0.7400
    Epoch 204/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0800 - acc: 0.7508 - val_loss: 1.0759 - val_acc: 0.7480
    Epoch 205/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0782 - acc: 0.7513 - val_loss: 1.0711 - val_acc: 0.7430
    Epoch 206/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 1.0765 - acc: 0.7496 - val_loss: 1.0708 - val_acc: 0.7420
    Epoch 207/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 1.0753 - acc: 0.7517 - val_loss: 1.0696 - val_acc: 0.7420
    Epoch 208/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 1.0740 - acc: 0.7512 - val_loss: 1.0685 - val_acc: 0.7400
    Epoch 209/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0720 - acc: 0.7521 - val_loss: 1.0647 - val_acc: 0.7430
    Epoch 210/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0696 - acc: 0.7523 - val_loss: 1.0644 - val_acc: 0.7400
    Epoch 211/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0693 - acc: 0.7537 - val_loss: 1.0667 - val_acc: 0.7420
    Epoch 212/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 1.0681 - acc: 0.7524 - val_loss: 1.0655 - val_acc: 0.7390
    Epoch 213/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 1.0656 - acc: 0.7537 - val_loss: 1.0602 - val_acc: 0.7420
    Epoch 214/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0650 - acc: 0.7543 - val_loss: 1.0602 - val_acc: 0.7420
    Epoch 215/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 1.0624 - acc: 0.7532 - val_loss: 1.0579 - val_acc: 0.7480
    Epoch 216/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 1.0619 - acc: 0.7535 - val_loss: 1.0581 - val_acc: 0.7390
    Epoch 217/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0599 - acc: 0.7533 - val_loss: 1.0526 - val_acc: 0.7450
    Epoch 218/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0584 - acc: 0.7539 - val_loss: 1.0534 - val_acc: 0.7440
    Epoch 219/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0572 - acc: 0.7547 - val_loss: 1.0521 - val_acc: 0.7430
    Epoch 220/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0560 - acc: 0.7552 - val_loss: 1.0489 - val_acc: 0.7490
    Epoch 221/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0544 - acc: 0.7543 - val_loss: 1.0516 - val_acc: 0.7460
    Epoch 222/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0530 - acc: 0.7552 - val_loss: 1.0571 - val_acc: 0.7400
    Epoch 223/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0526 - acc: 0.7544 - val_loss: 1.0492 - val_acc: 0.7450
    Epoch 224/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0509 - acc: 0.7527 - val_loss: 1.0506 - val_acc: 0.7450
    Epoch 225/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0493 - acc: 0.7540 - val_loss: 1.0441 - val_acc: 0.7420
    Epoch 226/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0479 - acc: 0.7555 - val_loss: 1.0413 - val_acc: 0.7430
    Epoch 227/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0465 - acc: 0.7545 - val_loss: 1.0415 - val_acc: 0.7440
    Epoch 228/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0449 - acc: 0.7561 - val_loss: 1.0418 - val_acc: 0.7440
    Epoch 229/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0438 - acc: 0.7551 - val_loss: 1.0408 - val_acc: 0.7460
    Epoch 230/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0421 - acc: 0.7563 - val_loss: 1.0380 - val_acc: 0.7540
    Epoch 231/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0410 - acc: 0.7564 - val_loss: 1.0371 - val_acc: 0.7440
    Epoch 232/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0400 - acc: 0.7569 - val_loss: 1.0419 - val_acc: 0.7370
    Epoch 233/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0390 - acc: 0.7565 - val_loss: 1.0352 - val_acc: 0.7450
    Epoch 234/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0372 - acc: 0.7571 - val_loss: 1.0390 - val_acc: 0.7400
    Epoch 235/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0359 - acc: 0.7575 - val_loss: 1.0348 - val_acc: 0.7490
    Epoch 236/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0357 - acc: 0.7561 - val_loss: 1.0300 - val_acc: 0.7450
    Epoch 237/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0334 - acc: 0.7564 - val_loss: 1.0296 - val_acc: 0.7510
    Epoch 238/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0328 - acc: 0.7564 - val_loss: 1.0358 - val_acc: 0.7510
    Epoch 239/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0323 - acc: 0.7555 - val_loss: 1.0284 - val_acc: 0.7460
    Epoch 240/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0306 - acc: 0.7579 - val_loss: 1.0279 - val_acc: 0.7490
    Epoch 241/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0292 - acc: 0.7588 - val_loss: 1.0257 - val_acc: 0.7470
    Epoch 242/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0285 - acc: 0.7576 - val_loss: 1.0270 - val_acc: 0.7450
    Epoch 243/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0269 - acc: 0.7560 - val_loss: 1.0248 - val_acc: 0.7420
    Epoch 244/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0257 - acc: 0.7575 - val_loss: 1.0254 - val_acc: 0.7450
    Epoch 245/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0244 - acc: 0.7569 - val_loss: 1.0243 - val_acc: 0.7450
    Epoch 246/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0233 - acc: 0.7599 - val_loss: 1.0241 - val_acc: 0.7460
    Epoch 247/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0225 - acc: 0.7581 - val_loss: 1.0215 - val_acc: 0.7440
    Epoch 248/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0215 - acc: 0.7583 - val_loss: 1.0181 - val_acc: 0.7440
    Epoch 249/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0202 - acc: 0.7609 - val_loss: 1.0187 - val_acc: 0.7470
    Epoch 250/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0196 - acc: 0.7559 - val_loss: 1.0175 - val_acc: 0.7430
    Epoch 251/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0179 - acc: 0.7597 - val_loss: 1.0187 - val_acc: 0.7550
    Epoch 252/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0173 - acc: 0.7612 - val_loss: 1.0198 - val_acc: 0.7440
    Epoch 253/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0165 - acc: 0.7596 - val_loss: 1.0164 - val_acc: 0.7520
    Epoch 254/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 1.0160 - acc: 0.7591 - val_loss: 1.0137 - val_acc: 0.7450
    Epoch 255/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0143 - acc: 0.7608 - val_loss: 1.0162 - val_acc: 0.7460
    Epoch 256/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0129 - acc: 0.7603 - val_loss: 1.0109 - val_acc: 0.7510
    Epoch 257/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0121 - acc: 0.7615 - val_loss: 1.0209 - val_acc: 0.7530
    Epoch 258/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0118 - acc: 0.7607 - val_loss: 1.0087 - val_acc: 0.7530
    Epoch 259/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0103 - acc: 0.7604 - val_loss: 1.0091 - val_acc: 0.7490
    Epoch 260/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0092 - acc: 0.7607 - val_loss: 1.0086 - val_acc: 0.7460
    Epoch 261/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0081 - acc: 0.7613 - val_loss: 1.0151 - val_acc: 0.7430
    Epoch 262/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0073 - acc: 0.7589 - val_loss: 1.0057 - val_acc: 0.7450
    Epoch 263/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0056 - acc: 0.7624 - val_loss: 1.0074 - val_acc: 0.7530
    Epoch 264/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0054 - acc: 0.7617 - val_loss: 1.0050 - val_acc: 0.7430
    Epoch 265/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0038 - acc: 0.7629 - val_loss: 1.0060 - val_acc: 0.7520
    Epoch 266/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0037 - acc: 0.7595 - val_loss: 1.0063 - val_acc: 0.7470
    Epoch 267/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0023 - acc: 0.7624 - val_loss: 1.0027 - val_acc: 0.7550
    Epoch 268/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0012 - acc: 0.7624 - val_loss: 1.0099 - val_acc: 0.7470
    Epoch 269/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0015 - acc: 0.7605 - val_loss: 1.0015 - val_acc: 0.7520
    Epoch 270/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9996 - acc: 0.7640 - val_loss: 0.9987 - val_acc: 0.7490
    Epoch 271/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9989 - acc: 0.7629 - val_loss: 0.9976 - val_acc: 0.7500
    Epoch 272/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9978 - acc: 0.7636 - val_loss: 1.0029 - val_acc: 0.7480
    Epoch 273/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9971 - acc: 0.7621 - val_loss: 0.9973 - val_acc: 0.7540
    Epoch 274/1000
    7500/7500 [==============================] - 0s 29us/step - loss: 0.9967 - acc: 0.7612 - val_loss: 0.9938 - val_acc: 0.7510
    Epoch 275/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.9952 - acc: 0.7636 - val_loss: 0.9967 - val_acc: 0.7530
    Epoch 276/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.9945 - acc: 0.7635 - val_loss: 1.0000 - val_acc: 0.7500
    Epoch 277/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9937 - acc: 0.7629 - val_loss: 0.9950 - val_acc: 0.7470
    Epoch 278/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.9926 - acc: 0.7643 - val_loss: 0.9954 - val_acc: 0.7530
    Epoch 279/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 0.9927 - acc: 0.7635 - val_loss: 0.9956 - val_acc: 0.7470
    Epoch 280/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9908 - acc: 0.7643 - val_loss: 0.9973 - val_acc: 0.7450
    Epoch 281/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9906 - acc: 0.7633 - val_loss: 0.9918 - val_acc: 0.7570
    Epoch 282/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.9895 - acc: 0.7609 - val_loss: 0.9884 - val_acc: 0.7470
    Epoch 283/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9884 - acc: 0.7655 - val_loss: 0.9925 - val_acc: 0.7500
    Epoch 284/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9876 - acc: 0.7648 - val_loss: 0.9932 - val_acc: 0.7490
    Epoch 285/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.9868 - acc: 0.7620 - val_loss: 0.9950 - val_acc: 0.7450
    Epoch 286/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.9861 - acc: 0.7667 - val_loss: 0.9908 - val_acc: 0.7520
    Epoch 287/1000
    7500/7500 [==============================] - 0s 41us/step - loss: 0.9855 - acc: 0.7635 - val_loss: 0.9861 - val_acc: 0.7460
    Epoch 288/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.9847 - acc: 0.7668 - val_loss: 0.9927 - val_acc: 0.7400
    Epoch 289/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.9841 - acc: 0.7645 - val_loss: 0.9864 - val_acc: 0.7560
    Epoch 290/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9829 - acc: 0.7653 - val_loss: 0.9829 - val_acc: 0.7460
    Epoch 291/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9823 - acc: 0.7655 - val_loss: 0.9877 - val_acc: 0.7510
    Epoch 292/1000
    7500/7500 [==============================] - 0s 24us/step - loss: 0.9815 - acc: 0.7645 - val_loss: 0.9833 - val_acc: 0.7570
    Epoch 293/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9808 - acc: 0.7655 - val_loss: 0.9834 - val_acc: 0.7520
    Epoch 294/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9789 - acc: 0.7656 - val_loss: 0.9922 - val_acc: 0.7470
    Epoch 295/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9784 - acc: 0.7665 - val_loss: 0.9815 - val_acc: 0.7480
    Epoch 296/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9785 - acc: 0.7648 - val_loss: 0.9781 - val_acc: 0.7530
    Epoch 297/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9774 - acc: 0.7657 - val_loss: 0.9808 - val_acc: 0.7510
    Epoch 298/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9768 - acc: 0.7657 - val_loss: 0.9815 - val_acc: 0.7480
    Epoch 299/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9760 - acc: 0.7667 - val_loss: 0.9817 - val_acc: 0.7510
    Epoch 300/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9747 - acc: 0.7656 - val_loss: 0.9809 - val_acc: 0.7530
    Epoch 301/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9749 - acc: 0.7655 - val_loss: 0.9872 - val_acc: 0.7500
    Epoch 302/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9742 - acc: 0.7675 - val_loss: 0.9758 - val_acc: 0.7460
    Epoch 303/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9733 - acc: 0.7692 - val_loss: 0.9749 - val_acc: 0.7570
    Epoch 304/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9723 - acc: 0.7675 - val_loss: 0.9831 - val_acc: 0.7460
    Epoch 305/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9721 - acc: 0.7671 - val_loss: 0.9876 - val_acc: 0.7440
    Epoch 306/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9712 - acc: 0.7688 - val_loss: 0.9750 - val_acc: 0.7560
    Epoch 307/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9705 - acc: 0.7691 - val_loss: 0.9740 - val_acc: 0.7520
    Epoch 308/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9695 - acc: 0.7700 - val_loss: 0.9739 - val_acc: 0.7550
    Epoch 309/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9689 - acc: 0.7691 - val_loss: 0.9769 - val_acc: 0.7460
    Epoch 310/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9683 - acc: 0.7680 - val_loss: 0.9741 - val_acc: 0.7440
    Epoch 311/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9672 - acc: 0.7667 - val_loss: 0.9804 - val_acc: 0.7460
    Epoch 312/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9665 - acc: 0.7671 - val_loss: 0.9692 - val_acc: 0.7480
    Epoch 313/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9658 - acc: 0.7677 - val_loss: 0.9884 - val_acc: 0.7550
    Epoch 314/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9668 - acc: 0.7697 - val_loss: 0.9716 - val_acc: 0.7540
    Epoch 315/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9652 - acc: 0.7655 - val_loss: 0.9746 - val_acc: 0.7520
    Epoch 316/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9644 - acc: 0.7688 - val_loss: 0.9681 - val_acc: 0.7560
    Epoch 317/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9641 - acc: 0.7685 - val_loss: 0.9718 - val_acc: 0.7510
    Epoch 318/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9636 - acc: 0.7688 - val_loss: 0.9729 - val_acc: 0.7540
    Epoch 319/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9627 - acc: 0.7673 - val_loss: 0.9664 - val_acc: 0.7490
    Epoch 320/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9621 - acc: 0.7689 - val_loss: 0.9716 - val_acc: 0.7520
    Epoch 321/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9616 - acc: 0.7699 - val_loss: 0.9635 - val_acc: 0.7540
    Epoch 322/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9616 - acc: 0.7688 - val_loss: 0.9703 - val_acc: 0.7440
    Epoch 323/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9603 - acc: 0.7695 - val_loss: 0.9697 - val_acc: 0.7480
    Epoch 324/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9605 - acc: 0.7669 - val_loss: 0.9734 - val_acc: 0.7450
    Epoch 325/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9597 - acc: 0.7693 - val_loss: 0.9655 - val_acc: 0.7500
    Epoch 326/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9586 - acc: 0.7687 - val_loss: 0.9644 - val_acc: 0.7510
    Epoch 327/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9577 - acc: 0.7683 - val_loss: 0.9633 - val_acc: 0.7580
    Epoch 328/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9576 - acc: 0.7681 - val_loss: 0.9751 - val_acc: 0.7540
    Epoch 329/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9574 - acc: 0.7683 - val_loss: 0.9652 - val_acc: 0.7600
    Epoch 330/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9572 - acc: 0.7700 - val_loss: 0.9627 - val_acc: 0.7550
    Epoch 331/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9555 - acc: 0.7677 - val_loss: 0.9716 - val_acc: 0.7520
    Epoch 332/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9556 - acc: 0.7693 - val_loss: 0.9616 - val_acc: 0.7630
    Epoch 333/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9555 - acc: 0.7679 - val_loss: 0.9629 - val_acc: 0.7540
    Epoch 334/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9541 - acc: 0.7692 - val_loss: 0.9640 - val_acc: 0.7460
    Epoch 335/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9541 - acc: 0.7700 - val_loss: 0.9772 - val_acc: 0.7480
    Epoch 336/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9538 - acc: 0.7713 - val_loss: 0.9612 - val_acc: 0.7500
    Epoch 337/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9530 - acc: 0.7688 - val_loss: 0.9576 - val_acc: 0.7620
    Epoch 338/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9510 - acc: 0.7699 - val_loss: 0.9588 - val_acc: 0.7510
    Epoch 339/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9512 - acc: 0.7692 - val_loss: 0.9565 - val_acc: 0.7580
    Epoch 340/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9517 - acc: 0.7689 - val_loss: 0.9629 - val_acc: 0.7490
    Epoch 341/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9504 - acc: 0.7700 - val_loss: 0.9594 - val_acc: 0.7610
    Epoch 342/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9503 - acc: 0.7680 - val_loss: 0.9571 - val_acc: 0.7550
    Epoch 343/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9496 - acc: 0.7681 - val_loss: 0.9551 - val_acc: 0.7600
    Epoch 344/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9488 - acc: 0.7712 - val_loss: 0.9659 - val_acc: 0.7510
    Epoch 345/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9487 - acc: 0.7713 - val_loss: 0.9705 - val_acc: 0.7460
    Epoch 346/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9483 - acc: 0.7709 - val_loss: 0.9521 - val_acc: 0.7580
    Epoch 347/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9468 - acc: 0.7705 - val_loss: 0.9535 - val_acc: 0.7620
    Epoch 348/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9469 - acc: 0.7689 - val_loss: 0.9570 - val_acc: 0.7630
    Epoch 349/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9463 - acc: 0.7717 - val_loss: 0.9548 - val_acc: 0.7530
    Epoch 350/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9472 - acc: 0.7719 - val_loss: 0.9545 - val_acc: 0.7610
    Epoch 351/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9461 - acc: 0.7701 - val_loss: 0.9621 - val_acc: 0.7510
    Epoch 352/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9448 - acc: 0.7716 - val_loss: 0.9535 - val_acc: 0.7650
    Epoch 353/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9450 - acc: 0.7705 - val_loss: 0.9510 - val_acc: 0.7590
    Epoch 354/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9444 - acc: 0.7703 - val_loss: 0.9564 - val_acc: 0.7510
    Epoch 355/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9434 - acc: 0.7708 - val_loss: 0.9491 - val_acc: 0.7660
    Epoch 356/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9426 - acc: 0.7720 - val_loss: 0.9536 - val_acc: 0.7530
    Epoch 357/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9417 - acc: 0.7707 - val_loss: 0.9525 - val_acc: 0.7530
    Epoch 358/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9429 - acc: 0.7719 - val_loss: 0.9516 - val_acc: 0.7600
    Epoch 359/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.9419 - acc: 0.7729 - val_loss: 0.9532 - val_acc: 0.7620
    Epoch 360/1000
    7500/7500 [==============================] - ETA: 0s - loss: 0.9450 - acc: 0.770 - 0s 44us/step - loss: 0.9427 - acc: 0.7716 - val_loss: 0.9465 - val_acc: 0.7610
    Epoch 361/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9407 - acc: 0.7715 - val_loss: 0.9475 - val_acc: 0.7570
    Epoch 362/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9408 - acc: 0.7723 - val_loss: 0.9450 - val_acc: 0.7630
    Epoch 363/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9394 - acc: 0.7712 - val_loss: 0.9461 - val_acc: 0.7660
    Epoch 364/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.9392 - acc: 0.7729 - val_loss: 0.9468 - val_acc: 0.7550
    Epoch 365/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9388 - acc: 0.7712 - val_loss: 0.9449 - val_acc: 0.7540
    Epoch 366/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.9387 - acc: 0.7731 - val_loss: 0.9479 - val_acc: 0.7640
    Epoch 367/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.9381 - acc: 0.7711 - val_loss: 0.9654 - val_acc: 0.7400
    Epoch 368/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 0.9390 - acc: 0.7708 - val_loss: 0.9449 - val_acc: 0.7600
    Epoch 369/1000
    7500/7500 [==============================] - 0s 47us/step - loss: 0.9373 - acc: 0.7705 - val_loss: 0.9459 - val_acc: 0.7630
    Epoch 370/1000
    7500/7500 [==============================] - 0s 47us/step - loss: 0.9374 - acc: 0.7713 - val_loss: 0.9447 - val_acc: 0.7660
    Epoch 371/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9356 - acc: 0.7717 - val_loss: 0.9426 - val_acc: 0.7650
    Epoch 372/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9351 - acc: 0.7747 - val_loss: 0.9450 - val_acc: 0.7640
    Epoch 373/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9352 - acc: 0.7708 - val_loss: 0.9525 - val_acc: 0.7530
    Epoch 374/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9345 - acc: 0.7724 - val_loss: 0.9514 - val_acc: 0.7500
    Epoch 375/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9351 - acc: 0.7739 - val_loss: 0.9410 - val_acc: 0.7650
    Epoch 376/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9336 - acc: 0.7701 - val_loss: 0.9415 - val_acc: 0.7610
    Epoch 377/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9341 - acc: 0.7712 - val_loss: 0.9454 - val_acc: 0.7580
    Epoch 378/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9335 - acc: 0.7724 - val_loss: 0.9444 - val_acc: 0.7510
    Epoch 379/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.9326 - acc: 0.7729 - val_loss: 0.9418 - val_acc: 0.7610
    Epoch 380/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9327 - acc: 0.7732 - val_loss: 0.9391 - val_acc: 0.7640
    Epoch 381/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9313 - acc: 0.7736 - val_loss: 0.9389 - val_acc: 0.7680
    Epoch 382/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.9310 - acc: 0.7736 - val_loss: 0.9373 - val_acc: 0.7640
    Epoch 383/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9310 - acc: 0.7708 - val_loss: 0.9374 - val_acc: 0.7620
    Epoch 384/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9302 - acc: 0.7748 - val_loss: 0.9408 - val_acc: 0.7600
    Epoch 385/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9297 - acc: 0.7739 - val_loss: 0.9381 - val_acc: 0.7620
    Epoch 386/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9301 - acc: 0.7727 - val_loss: 0.9401 - val_acc: 0.7540
    Epoch 387/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9296 - acc: 0.7729 - val_loss: 0.9441 - val_acc: 0.7550
    Epoch 388/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.9287 - acc: 0.7724 - val_loss: 0.9380 - val_acc: 0.7680
    Epoch 389/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9281 - acc: 0.7731 - val_loss: 0.9398 - val_acc: 0.7630
    Epoch 390/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9281 - acc: 0.7731 - val_loss: 0.9389 - val_acc: 0.7570
    Epoch 391/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9277 - acc: 0.7737 - val_loss: 0.9360 - val_acc: 0.7630
    Epoch 392/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9273 - acc: 0.7733 - val_loss: 0.9399 - val_acc: 0.7600
    Epoch 393/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9268 - acc: 0.7725 - val_loss: 0.9360 - val_acc: 0.7590
    Epoch 394/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.9270 - acc: 0.7736 - val_loss: 0.9385 - val_acc: 0.7550
    Epoch 395/1000
    7500/7500 [==============================] - 0s 31us/step - loss: 0.9268 - acc: 0.7729 - val_loss: 0.9420 - val_acc: 0.7660
    Epoch 396/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9260 - acc: 0.7745 - val_loss: 0.9486 - val_acc: 0.7580
    Epoch 397/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9260 - acc: 0.7741 - val_loss: 0.9360 - val_acc: 0.7610
    Epoch 398/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9244 - acc: 0.7737 - val_loss: 0.9329 - val_acc: 0.7590
    Epoch 399/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9237 - acc: 0.7745 - val_loss: 0.9386 - val_acc: 0.7650
    Epoch 400/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9235 - acc: 0.7752 - val_loss: 0.9405 - val_acc: 0.7580
    Epoch 401/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9235 - acc: 0.7735 - val_loss: 0.9440 - val_acc: 0.7600
    Epoch 402/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9241 - acc: 0.7725 - val_loss: 0.9371 - val_acc: 0.7580
    Epoch 403/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9232 - acc: 0.7736 - val_loss: 0.9312 - val_acc: 0.7650
    Epoch 404/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9228 - acc: 0.7744 - val_loss: 0.9327 - val_acc: 0.7630
    Epoch 405/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9222 - acc: 0.7721 - val_loss: 0.9306 - val_acc: 0.7700
    Epoch 406/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9211 - acc: 0.7735 - val_loss: 0.9347 - val_acc: 0.7560
    Epoch 407/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9210 - acc: 0.7735 - val_loss: 0.9288 - val_acc: 0.7670
    Epoch 408/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9204 - acc: 0.7720 - val_loss: 0.9274 - val_acc: 0.7680
    Epoch 409/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9204 - acc: 0.7763 - val_loss: 0.9447 - val_acc: 0.7620
    Epoch 410/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9207 - acc: 0.7756 - val_loss: 0.9429 - val_acc: 0.7510
    Epoch 411/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9194 - acc: 0.7733 - val_loss: 0.9497 - val_acc: 0.7530
    Epoch 412/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9207 - acc: 0.7737 - val_loss: 0.9272 - val_acc: 0.7650
    Epoch 413/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9191 - acc: 0.7760 - val_loss: 0.9390 - val_acc: 0.7590
    Epoch 414/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9193 - acc: 0.7736 - val_loss: 0.9308 - val_acc: 0.7650
    Epoch 415/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9188 - acc: 0.7751 - val_loss: 0.9294 - val_acc: 0.7620
    Epoch 416/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9179 - acc: 0.7748 - val_loss: 0.9351 - val_acc: 0.7590
    Epoch 417/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9184 - acc: 0.7747 - val_loss: 0.9257 - val_acc: 0.7670
    Epoch 418/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9167 - acc: 0.7748 - val_loss: 0.9266 - val_acc: 0.7690
    Epoch 419/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9168 - acc: 0.7740 - val_loss: 0.9312 - val_acc: 0.7640
    Epoch 420/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9165 - acc: 0.7756 - val_loss: 0.9301 - val_acc: 0.7620
    Epoch 421/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9163 - acc: 0.7763 - val_loss: 0.9313 - val_acc: 0.7630
    Epoch 422/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9161 - acc: 0.7743 - val_loss: 0.9274 - val_acc: 0.7700
    Epoch 423/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9154 - acc: 0.7759 - val_loss: 0.9281 - val_acc: 0.7530
    Epoch 424/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9152 - acc: 0.7751 - val_loss: 0.9285 - val_acc: 0.7660
    Epoch 425/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9149 - acc: 0.7756 - val_loss: 0.9289 - val_acc: 0.7560
    Epoch 426/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9145 - acc: 0.7756 - val_loss: 0.9243 - val_acc: 0.7690
    Epoch 427/1000
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9138 - acc: 0.7768 - val_loss: 0.9295 - val_acc: 0.7560
    Epoch 428/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.9137 - acc: 0.7743 - val_loss: 0.9252 - val_acc: 0.7630
    Epoch 429/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 0.9131 - acc: 0.7760 - val_loss: 0.9238 - val_acc: 0.7650
    Epoch 430/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9133 - acc: 0.7751 - val_loss: 0.9230 - val_acc: 0.7660
    Epoch 431/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9129 - acc: 0.7724 - val_loss: 0.9237 - val_acc: 0.7600
    Epoch 432/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9137 - acc: 0.7729 - val_loss: 0.9236 - val_acc: 0.7670
    Epoch 433/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9113 - acc: 0.7757 - val_loss: 0.9300 - val_acc: 0.7560
    Epoch 434/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.9125 - acc: 0.7745 - val_loss: 0.9280 - val_acc: 0.7640
    Epoch 435/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9113 - acc: 0.7727 - val_loss: 0.9250 - val_acc: 0.7630
    Epoch 436/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9114 - acc: 0.7753 - val_loss: 0.9233 - val_acc: 0.7580
    Epoch 437/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9115 - acc: 0.7728 - val_loss: 0.9228 - val_acc: 0.7650
    Epoch 438/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9119 - acc: 0.7747 - val_loss: 0.9248 - val_acc: 0.7570
    Epoch 439/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9101 - acc: 0.7753 - val_loss: 0.9370 - val_acc: 0.7540
    Epoch 440/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9103 - acc: 0.7777 - val_loss: 0.9350 - val_acc: 0.7510
    Epoch 441/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9098 - acc: 0.7752 - val_loss: 0.9286 - val_acc: 0.7580
    Epoch 442/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9102 - acc: 0.7740 - val_loss: 0.9217 - val_acc: 0.7650
    Epoch 443/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.9085 - acc: 0.7751 - val_loss: 0.9227 - val_acc: 0.7610
    Epoch 444/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9097 - acc: 0.7771 - val_loss: 0.9202 - val_acc: 0.7650
    Epoch 445/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.9076 - acc: 0.7768 - val_loss: 0.9242 - val_acc: 0.7530
    Epoch 446/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.9082 - acc: 0.7761 - val_loss: 0.9350 - val_acc: 0.7640
    Epoch 447/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.9085 - acc: 0.7753 - val_loss: 0.9177 - val_acc: 0.7690
    Epoch 448/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.9064 - acc: 0.7777 - val_loss: 0.9183 - val_acc: 0.7660
    Epoch 449/1000
    7500/7500 [==============================] - 0s 32us/step - loss: 0.9064 - acc: 0.7748 - val_loss: 0.9190 - val_acc: 0.7610
    Epoch 450/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9062 - acc: 0.7765 - val_loss: 0.9218 - val_acc: 0.7620
    Epoch 451/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9070 - acc: 0.7749 - val_loss: 0.9217 - val_acc: 0.7680
    Epoch 452/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9063 - acc: 0.7764 - val_loss: 0.9162 - val_acc: 0.7660
    Epoch 453/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9058 - acc: 0.7761 - val_loss: 0.9189 - val_acc: 0.7640
    Epoch 454/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9059 - acc: 0.7767 - val_loss: 0.9195 - val_acc: 0.7640
    Epoch 455/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9049 - acc: 0.7780 - val_loss: 0.9222 - val_acc: 0.7580
    Epoch 456/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9043 - acc: 0.7783 - val_loss: 0.9232 - val_acc: 0.7610
    Epoch 457/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9049 - acc: 0.7772 - val_loss: 0.9200 - val_acc: 0.7640
    Epoch 458/1000
    7500/7500 [==============================] - 0s 24us/step - loss: 0.9046 - acc: 0.7784 - val_loss: 0.9249 - val_acc: 0.7580
    Epoch 459/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.9037 - acc: 0.7787 - val_loss: 0.9218 - val_acc: 0.7600
    Epoch 460/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9035 - acc: 0.7779 - val_loss: 0.9174 - val_acc: 0.7690
    Epoch 461/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9032 - acc: 0.7773 - val_loss: 0.9202 - val_acc: 0.7660
    Epoch 462/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9029 - acc: 0.7767 - val_loss: 0.9192 - val_acc: 0.7660
    Epoch 463/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9029 - acc: 0.7755 - val_loss: 0.9171 - val_acc: 0.7670
    Epoch 464/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9024 - acc: 0.7784 - val_loss: 0.9144 - val_acc: 0.7670
    Epoch 465/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9019 - acc: 0.7775 - val_loss: 0.9148 - val_acc: 0.7670
    Epoch 466/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9014 - acc: 0.7773 - val_loss: 0.9198 - val_acc: 0.7520
    Epoch 467/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9014 - acc: 0.7776 - val_loss: 0.9144 - val_acc: 0.7610
    Epoch 468/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9016 - acc: 0.7767 - val_loss: 0.9141 - val_acc: 0.7670
    Epoch 469/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9006 - acc: 0.7781 - val_loss: 0.9152 - val_acc: 0.7620
    Epoch 470/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9012 - acc: 0.7745 - val_loss: 0.9206 - val_acc: 0.7540
    Epoch 471/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8994 - acc: 0.7773 - val_loss: 0.9142 - val_acc: 0.7640
    Epoch 472/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9009 - acc: 0.7777 - val_loss: 0.9187 - val_acc: 0.7640
    Epoch 473/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9009 - acc: 0.7785 - val_loss: 0.9140 - val_acc: 0.7620
    Epoch 474/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8999 - acc: 0.7775 - val_loss: 0.9120 - val_acc: 0.7670
    Epoch 475/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8987 - acc: 0.7792 - val_loss: 0.9345 - val_acc: 0.7490
    Epoch 476/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.9005 - acc: 0.7776 - val_loss: 0.9133 - val_acc: 0.7700
    Epoch 477/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8992 - acc: 0.7785 - val_loss: 0.9191 - val_acc: 0.7580
    Epoch 478/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8987 - acc: 0.7788 - val_loss: 0.9105 - val_acc: 0.7640
    Epoch 479/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8982 - acc: 0.7780 - val_loss: 0.9209 - val_acc: 0.7580
    Epoch 480/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8986 - acc: 0.7792 - val_loss: 0.9102 - val_acc: 0.7650
    Epoch 481/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8975 - acc: 0.7781 - val_loss: 0.9128 - val_acc: 0.7690
    Epoch 482/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8969 - acc: 0.7769 - val_loss: 0.9133 - val_acc: 0.7560
    Epoch 483/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8982 - acc: 0.7776 - val_loss: 0.9196 - val_acc: 0.7680
    Epoch 484/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8989 - acc: 0.7796 - val_loss: 0.9117 - val_acc: 0.7690
    Epoch 485/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8973 - acc: 0.7791 - val_loss: 0.9126 - val_acc: 0.7600
    Epoch 486/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8967 - acc: 0.7781 - val_loss: 0.9115 - val_acc: 0.7590
    Epoch 487/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8961 - acc: 0.7791 - val_loss: 0.9100 - val_acc: 0.7660
    Epoch 488/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8964 - acc: 0.7771 - val_loss: 0.9090 - val_acc: 0.7660
    Epoch 489/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8956 - acc: 0.7788 - val_loss: 0.9136 - val_acc: 0.7560
    Epoch 490/1000
    7500/7500 [==============================] - 0s 49us/step - loss: 0.8959 - acc: 0.7793 - val_loss: 0.9222 - val_acc: 0.7490
    Epoch 491/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8955 - acc: 0.7792 - val_loss: 0.9107 - val_acc: 0.7670
    Epoch 492/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8950 - acc: 0.7809 - val_loss: 0.9091 - val_acc: 0.7700
    Epoch 493/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8940 - acc: 0.7764 - val_loss: 0.9135 - val_acc: 0.7680
    Epoch 494/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8941 - acc: 0.7785 - val_loss: 0.9099 - val_acc: 0.7600
    Epoch 495/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8928 - acc: 0.7781 - val_loss: 0.9082 - val_acc: 0.7660
    Epoch 496/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8932 - acc: 0.7785 - val_loss: 0.9113 - val_acc: 0.7620
    Epoch 497/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8926 - acc: 0.7807 - val_loss: 0.9074 - val_acc: 0.7710
    Epoch 498/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8939 - acc: 0.7772 - val_loss: 0.9051 - val_acc: 0.7680
    Epoch 499/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8925 - acc: 0.7781 - val_loss: 0.9156 - val_acc: 0.7650
    Epoch 500/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8932 - acc: 0.7793 - val_loss: 0.9136 - val_acc: 0.7660
    Epoch 501/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8932 - acc: 0.7797 - val_loss: 0.9077 - val_acc: 0.7670
    Epoch 502/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8929 - acc: 0.7780 - val_loss: 0.9101 - val_acc: 0.7660
    Epoch 503/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8921 - acc: 0.7788 - val_loss: 0.9118 - val_acc: 0.7580
    Epoch 504/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8917 - acc: 0.7781 - val_loss: 0.9090 - val_acc: 0.7700
    Epoch 505/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8913 - acc: 0.7797 - val_loss: 0.9057 - val_acc: 0.7640
    Epoch 506/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8922 - acc: 0.7783 - val_loss: 0.9152 - val_acc: 0.7520
    Epoch 507/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8911 - acc: 0.7776 - val_loss: 0.9081 - val_acc: 0.7620
    Epoch 508/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8917 - acc: 0.7783 - val_loss: 0.9075 - val_acc: 0.7630
    Epoch 509/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8913 - acc: 0.7773 - val_loss: 0.9061 - val_acc: 0.7660
    Epoch 510/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8903 - acc: 0.7791 - val_loss: 0.9047 - val_acc: 0.7680
    Epoch 511/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8894 - acc: 0.7792 - val_loss: 0.9070 - val_acc: 0.7620
    Epoch 512/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8905 - acc: 0.7783 - val_loss: 0.9043 - val_acc: 0.7690
    Epoch 513/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.8899 - acc: 0.7793 - val_loss: 0.9092 - val_acc: 0.7670
    Epoch 514/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8912 - acc: 0.7788 - val_loss: 0.9194 - val_acc: 0.7610
    Epoch 515/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8890 - acc: 0.7799 - val_loss: 0.9028 - val_acc: 0.7710
    Epoch 516/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8892 - acc: 0.7804 - val_loss: 0.9060 - val_acc: 0.7610
    Epoch 517/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8895 - acc: 0.7793 - val_loss: 0.9112 - val_acc: 0.7640
    Epoch 518/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8895 - acc: 0.7792 - val_loss: 0.9020 - val_acc: 0.7700
    Epoch 519/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8881 - acc: 0.7779 - val_loss: 0.9022 - val_acc: 0.7650
    Epoch 520/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8873 - acc: 0.7775 - val_loss: 0.9023 - val_acc: 0.7700
    Epoch 521/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8875 - acc: 0.7791 - val_loss: 0.9098 - val_acc: 0.7650
    Epoch 522/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8873 - acc: 0.7819 - val_loss: 0.8998 - val_acc: 0.7700
    Epoch 523/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8862 - acc: 0.7803 - val_loss: 0.9126 - val_acc: 0.7620
    Epoch 524/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8866 - acc: 0.7807 - val_loss: 0.9049 - val_acc: 0.7710
    Epoch 525/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8878 - acc: 0.7803 - val_loss: 0.9050 - val_acc: 0.7620
    Epoch 526/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8868 - acc: 0.7817 - val_loss: 0.9017 - val_acc: 0.7600
    Epoch 527/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8861 - acc: 0.7813 - val_loss: 0.9029 - val_acc: 0.7710
    Epoch 528/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8866 - acc: 0.7809 - val_loss: 0.9049 - val_acc: 0.7630
    Epoch 529/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8865 - acc: 0.7808 - val_loss: 0.9038 - val_acc: 0.7690
    Epoch 530/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8863 - acc: 0.7809 - val_loss: 0.9004 - val_acc: 0.7650
    Epoch 531/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8860 - acc: 0.7809 - val_loss: 0.9061 - val_acc: 0.7650
    Epoch 532/1000
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8848 - acc: 0.7819 - val_loss: 0.9010 - val_acc: 0.7680
    Epoch 533/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8845 - acc: 0.7815 - val_loss: 0.9010 - val_acc: 0.7630
    Epoch 534/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8846 - acc: 0.7801 - val_loss: 0.9220 - val_acc: 0.7600
    Epoch 535/1000
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8864 - acc: 0.7803 - val_loss: 0.9060 - val_acc: 0.7660
    Epoch 536/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8851 - acc: 0.7827 - val_loss: 0.9006 - val_acc: 0.7710
    Epoch 537/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8846 - acc: 0.7805 - val_loss: 0.9013 - val_acc: 0.7700
    Epoch 538/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8848 - acc: 0.7805 - val_loss: 0.9001 - val_acc: 0.7680
    Epoch 539/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8849 - acc: 0.7813 - val_loss: 0.8981 - val_acc: 0.7710
    Epoch 540/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8842 - acc: 0.7795 - val_loss: 0.9019 - val_acc: 0.7670
    Epoch 541/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8833 - acc: 0.7808 - val_loss: 0.8983 - val_acc: 0.7640
    Epoch 542/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8842 - acc: 0.7808 - val_loss: 0.8989 - val_acc: 0.7600
    Epoch 543/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8830 - acc: 0.7813 - val_loss: 0.9020 - val_acc: 0.7700
    Epoch 544/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8840 - acc: 0.7820 - val_loss: 0.9038 - val_acc: 0.7600
    Epoch 545/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8820 - acc: 0.7812 - val_loss: 0.9092 - val_acc: 0.7610
    Epoch 546/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8823 - acc: 0.7809 - val_loss: 0.8995 - val_acc: 0.7680
    Epoch 547/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8825 - acc: 0.7797 - val_loss: 0.9331 - val_acc: 0.7580
    Epoch 548/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8835 - acc: 0.7816 - val_loss: 0.8995 - val_acc: 0.7680
    Epoch 549/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8814 - acc: 0.7823 - val_loss: 0.9037 - val_acc: 0.7570
    Epoch 550/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8822 - acc: 0.7815 - val_loss: 0.8956 - val_acc: 0.7760
    Epoch 551/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8825 - acc: 0.7803 - val_loss: 0.9003 - val_acc: 0.7600
    Epoch 552/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8818 - acc: 0.7804 - val_loss: 0.8978 - val_acc: 0.7700
    Epoch 553/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8806 - acc: 0.7816 - val_loss: 0.8948 - val_acc: 0.7750
    Epoch 554/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8805 - acc: 0.7829 - val_loss: 0.9006 - val_acc: 0.7690
    Epoch 555/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8809 - acc: 0.7815 - val_loss: 0.8999 - val_acc: 0.7710
    Epoch 556/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8807 - acc: 0.7827 - val_loss: 0.8971 - val_acc: 0.7670
    Epoch 557/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8810 - acc: 0.7813 - val_loss: 0.8964 - val_acc: 0.7720
    Epoch 558/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8806 - acc: 0.7815 - val_loss: 0.9139 - val_acc: 0.7570
    Epoch 559/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8803 - acc: 0.7831 - val_loss: 0.8958 - val_acc: 0.7720
    Epoch 560/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8803 - acc: 0.7825 - val_loss: 0.9025 - val_acc: 0.7760
    Epoch 561/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8799 - acc: 0.7827 - val_loss: 0.8957 - val_acc: 0.7680
    Epoch 562/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8796 - acc: 0.7824 - val_loss: 0.8983 - val_acc: 0.7640
    Epoch 563/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8794 - acc: 0.7823 - val_loss: 0.8961 - val_acc: 0.7670
    Epoch 564/1000
    7500/7500 [==============================] - ETA: 0s - loss: 0.8789 - acc: 0.781 - 0s 36us/step - loss: 0.8798 - acc: 0.7816 - val_loss: 0.8959 - val_acc: 0.7700
    Epoch 565/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8789 - acc: 0.7817 - val_loss: 0.9079 - val_acc: 0.7670
    Epoch 566/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8780 - acc: 0.7828 - val_loss: 0.8931 - val_acc: 0.7660
    Epoch 567/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8794 - acc: 0.7817 - val_loss: 0.9093 - val_acc: 0.7570
    Epoch 568/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8796 - acc: 0.7829 - val_loss: 0.9164 - val_acc: 0.7620
    Epoch 569/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8798 - acc: 0.7803 - val_loss: 0.8987 - val_acc: 0.7660
    Epoch 570/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8778 - acc: 0.7827 - val_loss: 0.9133 - val_acc: 0.7520
    Epoch 571/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8782 - acc: 0.7827 - val_loss: 0.8960 - val_acc: 0.7670
    Epoch 572/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8773 - acc: 0.7812 - val_loss: 0.9114 - val_acc: 0.7610
    Epoch 573/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8788 - acc: 0.7813 - val_loss: 0.8938 - val_acc: 0.7700
    Epoch 574/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8767 - acc: 0.7835 - val_loss: 0.9009 - val_acc: 0.7680
    Epoch 575/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8770 - acc: 0.7852 - val_loss: 0.8958 - val_acc: 0.7730
    Epoch 576/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8760 - acc: 0.7825 - val_loss: 0.8958 - val_acc: 0.7720
    Epoch 577/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8765 - acc: 0.7804 - val_loss: 0.8937 - val_acc: 0.7740
    Epoch 578/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8771 - acc: 0.7832 - val_loss: 0.8957 - val_acc: 0.7680
    Epoch 579/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8775 - acc: 0.7815 - val_loss: 0.8947 - val_acc: 0.7720
    Epoch 580/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8756 - acc: 0.7839 - val_loss: 0.8932 - val_acc: 0.7710
    Epoch 581/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8765 - acc: 0.7823 - val_loss: 0.8949 - val_acc: 0.7690
    Epoch 582/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8767 - acc: 0.7823 - val_loss: 0.8966 - val_acc: 0.7650
    Epoch 583/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8755 - acc: 0.7831 - val_loss: 0.8939 - val_acc: 0.7660
    Epoch 584/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8768 - acc: 0.7817 - val_loss: 0.8959 - val_acc: 0.7700
    Epoch 585/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8773 - acc: 0.7820 - val_loss: 0.8954 - val_acc: 0.7690
    Epoch 586/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8759 - acc: 0.7800 - val_loss: 0.9110 - val_acc: 0.7630
    Epoch 587/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8749 - acc: 0.7825 - val_loss: 0.8942 - val_acc: 0.7660
    Epoch 588/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8749 - acc: 0.7833 - val_loss: 0.8952 - val_acc: 0.7670
    Epoch 589/1000
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8751 - acc: 0.7833 - val_loss: 0.8930 - val_acc: 0.7680
    Epoch 590/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8733 - acc: 0.7859 - val_loss: 0.8997 - val_acc: 0.7570
    Epoch 591/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8744 - acc: 0.7843 - val_loss: 0.8956 - val_acc: 0.7630
    Epoch 592/1000
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8740 - acc: 0.7827 - val_loss: 0.8945 - val_acc: 0.7690
    Epoch 593/1000
    7500/7500 [==============================] - 0s 29us/step - loss: 0.8744 - acc: 0.7829 - val_loss: 0.8928 - val_acc: 0.7660
    Epoch 594/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8747 - acc: 0.7817 - val_loss: 0.8969 - val_acc: 0.7700
    Epoch 595/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8738 - acc: 0.7829 - val_loss: 0.8980 - val_acc: 0.7670
    Epoch 596/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8726 - acc: 0.7845 - val_loss: 0.9185 - val_acc: 0.7620
    Epoch 597/1000
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8742 - acc: 0.7841 - val_loss: 0.8928 - val_acc: 0.7720
    Epoch 598/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8733 - acc: 0.7853 - val_loss: 0.8896 - val_acc: 0.7700
    Epoch 599/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8725 - acc: 0.7813 - val_loss: 0.8941 - val_acc: 0.7650
    Epoch 600/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8729 - acc: 0.7840 - val_loss: 0.9022 - val_acc: 0.7620
    Epoch 601/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8732 - acc: 0.7815 - val_loss: 0.8955 - val_acc: 0.7600
    Epoch 602/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8727 - acc: 0.7836 - val_loss: 0.8885 - val_acc: 0.7710
    Epoch 603/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8718 - acc: 0.7837 - val_loss: 0.8904 - val_acc: 0.7700
    Epoch 604/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8715 - acc: 0.7837 - val_loss: 0.8936 - val_acc: 0.7660
    Epoch 605/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8722 - acc: 0.7815 - val_loss: 0.8932 - val_acc: 0.7660
    Epoch 606/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8735 - acc: 0.7813 - val_loss: 0.8924 - val_acc: 0.7670
    Epoch 607/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8708 - acc: 0.7855 - val_loss: 0.8929 - val_acc: 0.7710
    Epoch 608/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8717 - acc: 0.7835 - val_loss: 0.8895 - val_acc: 0.7740
    Epoch 609/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8707 - acc: 0.7851 - val_loss: 0.9019 - val_acc: 0.7520
    Epoch 610/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8706 - acc: 0.7836 - val_loss: 0.8904 - val_acc: 0.7680
    Epoch 611/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8717 - acc: 0.7833 - val_loss: 0.8919 - val_acc: 0.7690
    Epoch 612/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8708 - acc: 0.7847 - val_loss: 0.8945 - val_acc: 0.7730
    Epoch 613/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8701 - acc: 0.7823 - val_loss: 0.8916 - val_acc: 0.7700
    Epoch 614/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8695 - acc: 0.7828 - val_loss: 0.8978 - val_acc: 0.7550
    Epoch 615/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8715 - acc: 0.7839 - val_loss: 0.9058 - val_acc: 0.7530
    Epoch 616/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8698 - acc: 0.7824 - val_loss: 0.8964 - val_acc: 0.7680
    Epoch 617/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8692 - acc: 0.7851 - val_loss: 0.8976 - val_acc: 0.7690
    Epoch 618/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8707 - acc: 0.7844 - val_loss: 0.8921 - val_acc: 0.7620
    Epoch 619/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8695 - acc: 0.7840 - val_loss: 0.8903 - val_acc: 0.7720
    Epoch 620/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8695 - acc: 0.7839 - val_loss: 0.8968 - val_acc: 0.7590
    Epoch 621/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8697 - acc: 0.7833 - val_loss: 0.8901 - val_acc: 0.7650
    Epoch 622/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8685 - acc: 0.7851 - val_loss: 0.8876 - val_acc: 0.7730
    Epoch 623/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8696 - acc: 0.7837 - val_loss: 0.9169 - val_acc: 0.7610
    Epoch 624/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8693 - acc: 0.7831 - val_loss: 0.8896 - val_acc: 0.7770
    Epoch 625/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8682 - acc: 0.7828 - val_loss: 0.8882 - val_acc: 0.7670
    Epoch 626/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8690 - acc: 0.7803 - val_loss: 0.8892 - val_acc: 0.7680
    Epoch 627/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8687 - acc: 0.7824 - val_loss: 0.8875 - val_acc: 0.7770
    Epoch 628/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8687 - acc: 0.7839 - val_loss: 0.8971 - val_acc: 0.7640
    Epoch 629/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8680 - acc: 0.7848 - val_loss: 0.8876 - val_acc: 0.7740
    Epoch 630/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8674 - acc: 0.7847 - val_loss: 0.9123 - val_acc: 0.7540
    Epoch 631/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8684 - acc: 0.7851 - val_loss: 0.8878 - val_acc: 0.7690
    Epoch 632/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8683 - acc: 0.7845 - val_loss: 0.8944 - val_acc: 0.7520
    Epoch 633/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8670 - acc: 0.7845 - val_loss: 0.8923 - val_acc: 0.7690
    Epoch 634/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8683 - acc: 0.7837 - val_loss: 0.8988 - val_acc: 0.7670
    Epoch 635/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8667 - acc: 0.7852 - val_loss: 0.8988 - val_acc: 0.7700
    Epoch 636/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8675 - acc: 0.7851 - val_loss: 0.8952 - val_acc: 0.7700
    Epoch 637/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8669 - acc: 0.7843 - val_loss: 0.8854 - val_acc: 0.7690
    Epoch 638/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8669 - acc: 0.7835 - val_loss: 0.8869 - val_acc: 0.7720
    Epoch 639/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8668 - acc: 0.7824 - val_loss: 0.9022 - val_acc: 0.7690
    Epoch 640/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8677 - acc: 0.7831 - val_loss: 0.8940 - val_acc: 0.7730
    Epoch 641/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8654 - acc: 0.7859 - val_loss: 0.8909 - val_acc: 0.7730
    Epoch 642/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8671 - acc: 0.7841 - val_loss: 0.8867 - val_acc: 0.7720
    Epoch 643/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8649 - acc: 0.7823 - val_loss: 0.8942 - val_acc: 0.7660
    Epoch 644/1000
    7500/7500 [==============================] - 0s 41us/step - loss: 0.8665 - acc: 0.7848 - val_loss: 0.8929 - val_acc: 0.7740
    Epoch 645/1000
    7500/7500 [==============================] - 0s 41us/step - loss: 0.8648 - acc: 0.7843 - val_loss: 0.8940 - val_acc: 0.7730
    Epoch 646/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8660 - acc: 0.7836 - val_loss: 0.8881 - val_acc: 0.7660
    Epoch 647/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8643 - acc: 0.7864 - val_loss: 0.8970 - val_acc: 0.7670
    Epoch 648/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8653 - acc: 0.7841 - val_loss: 0.8851 - val_acc: 0.7760
    Epoch 649/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8649 - acc: 0.7836 - val_loss: 0.8896 - val_acc: 0.7760
    Epoch 650/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8640 - acc: 0.7841 - val_loss: 0.9073 - val_acc: 0.7550
    Epoch 651/1000
    7500/7500 [==============================] - 0s 44us/step - loss: 0.8668 - acc: 0.7856 - val_loss: 0.8923 - val_acc: 0.7630
    Epoch 652/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8658 - acc: 0.7857 - val_loss: 0.8906 - val_acc: 0.7760
    Epoch 653/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8642 - acc: 0.7855 - val_loss: 0.8862 - val_acc: 0.7560
    Epoch 654/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8643 - acc: 0.7841 - val_loss: 0.8937 - val_acc: 0.7660
    Epoch 655/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8647 - acc: 0.7861 - val_loss: 0.9076 - val_acc: 0.7620
    Epoch 656/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8655 - acc: 0.7853 - val_loss: 0.8893 - val_acc: 0.7700
    Epoch 657/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8645 - acc: 0.7832 - val_loss: 0.8870 - val_acc: 0.7620
    Epoch 658/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8646 - acc: 0.7843 - val_loss: 0.8907 - val_acc: 0.7620
    Epoch 659/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8649 - acc: 0.7856 - val_loss: 0.8835 - val_acc: 0.7760
    Epoch 660/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8632 - acc: 0.7845 - val_loss: 0.8857 - val_acc: 0.7690
    Epoch 661/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8629 - acc: 0.7832 - val_loss: 0.8880 - val_acc: 0.7660
    Epoch 662/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8625 - acc: 0.7867 - val_loss: 0.8946 - val_acc: 0.7670
    Epoch 663/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8632 - acc: 0.7867 - val_loss: 0.9199 - val_acc: 0.7470
    Epoch 664/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8648 - acc: 0.7844 - val_loss: 0.8890 - val_acc: 0.7710
    Epoch 665/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8626 - acc: 0.7867 - val_loss: 0.8937 - val_acc: 0.7620
    Epoch 666/1000
    7500/7500 [==============================] - 0s 33us/step - loss: 0.8642 - acc: 0.7847 - val_loss: 0.8906 - val_acc: 0.7690
    Epoch 667/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8622 - acc: 0.7860 - val_loss: 0.8846 - val_acc: 0.7710
    Epoch 668/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8621 - acc: 0.7855 - val_loss: 0.8870 - val_acc: 0.7670
    Epoch 669/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8630 - acc: 0.7855 - val_loss: 0.8827 - val_acc: 0.7710
    Epoch 670/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8619 - acc: 0.7847 - val_loss: 0.8839 - val_acc: 0.7690
    Epoch 671/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8638 - acc: 0.7860 - val_loss: 0.8872 - val_acc: 0.7560
    Epoch 672/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8615 - acc: 0.7848 - val_loss: 0.8840 - val_acc: 0.7770
    Epoch 673/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8606 - acc: 0.7865 - val_loss: 0.8842 - val_acc: 0.7670
    Epoch 674/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8600 - acc: 0.7875 - val_loss: 0.8842 - val_acc: 0.7740
    Epoch 675/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8627 - acc: 0.7840 - val_loss: 0.8854 - val_acc: 0.7720
    Epoch 676/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8617 - acc: 0.7847 - val_loss: 0.8840 - val_acc: 0.7750
    Epoch 677/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8607 - acc: 0.7855 - val_loss: 0.8852 - val_acc: 0.7750
    Epoch 678/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8610 - acc: 0.7844 - val_loss: 0.8845 - val_acc: 0.7740
    Epoch 679/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8604 - acc: 0.7869 - val_loss: 0.8938 - val_acc: 0.7700
    Epoch 680/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8603 - acc: 0.7865 - val_loss: 0.8844 - val_acc: 0.7720
    Epoch 681/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8603 - acc: 0.7857 - val_loss: 0.8933 - val_acc: 0.7670
    Epoch 682/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8608 - acc: 0.7860 - val_loss: 0.8809 - val_acc: 0.7740
    Epoch 683/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8609 - acc: 0.7847 - val_loss: 0.8915 - val_acc: 0.7640
    Epoch 684/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8605 - acc: 0.7871 - val_loss: 0.8904 - val_acc: 0.7600
    Epoch 685/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8605 - acc: 0.7867 - val_loss: 0.8816 - val_acc: 0.7760
    Epoch 686/1000
    7500/7500 [==============================] - 0s 33us/step - loss: 0.8610 - acc: 0.7833 - val_loss: 0.8809 - val_acc: 0.7740
    Epoch 687/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8585 - acc: 0.7853 - val_loss: 0.8882 - val_acc: 0.7700
    Epoch 688/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8602 - acc: 0.7852 - val_loss: 0.8810 - val_acc: 0.7760
    Epoch 689/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8584 - acc: 0.7859 - val_loss: 0.8847 - val_acc: 0.7660
    Epoch 690/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8604 - acc: 0.7852 - val_loss: 0.8929 - val_acc: 0.7670
    Epoch 691/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8613 - acc: 0.7852 - val_loss: 0.8826 - val_acc: 0.7730
    Epoch 692/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8590 - acc: 0.7857 - val_loss: 0.8836 - val_acc: 0.7750
    Epoch 693/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8594 - acc: 0.7865 - val_loss: 0.8780 - val_acc: 0.7750
    Epoch 694/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8584 - acc: 0.7837 - val_loss: 0.8789 - val_acc: 0.7720
    Epoch 695/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.8569 - acc: 0.7856 - val_loss: 0.9019 - val_acc: 0.7580
    Epoch 696/1000
    7500/7500 [==============================] - 0s 44us/step - loss: 0.8586 - acc: 0.7864 - val_loss: 0.8833 - val_acc: 0.7680
    Epoch 697/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8587 - acc: 0.7867 - val_loss: 0.8867 - val_acc: 0.7670
    Epoch 698/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8589 - acc: 0.7884 - val_loss: 0.8854 - val_acc: 0.7630
    Epoch 699/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8580 - acc: 0.7863 - val_loss: 0.8930 - val_acc: 0.7600
    Epoch 700/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8586 - acc: 0.7864 - val_loss: 0.8877 - val_acc: 0.7720
    Epoch 701/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8589 - acc: 0.7864 - val_loss: 0.8848 - val_acc: 0.7670
    Epoch 702/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8573 - acc: 0.7863 - val_loss: 0.8811 - val_acc: 0.7800
    Epoch 703/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8580 - acc: 0.7875 - val_loss: 0.8932 - val_acc: 0.7660
    Epoch 704/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8581 - acc: 0.7884 - val_loss: 0.8864 - val_acc: 0.7630
    Epoch 705/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8568 - acc: 0.7860 - val_loss: 0.8818 - val_acc: 0.7700
    Epoch 706/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8582 - acc: 0.7865 - val_loss: 0.8820 - val_acc: 0.7780
    Epoch 707/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8578 - acc: 0.7875 - val_loss: 0.8899 - val_acc: 0.7640
    Epoch 708/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8567 - acc: 0.7865 - val_loss: 0.8828 - val_acc: 0.7630
    Epoch 709/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8559 - acc: 0.7872 - val_loss: 0.8801 - val_acc: 0.7730
    Epoch 710/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.8576 - acc: 0.7848 - val_loss: 0.9000 - val_acc: 0.7660
    Epoch 711/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8566 - acc: 0.7859 - val_loss: 0.8860 - val_acc: 0.7610
    Epoch 712/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8575 - acc: 0.7876 - val_loss: 0.8777 - val_acc: 0.7790
    Epoch 713/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8551 - acc: 0.7885 - val_loss: 0.8820 - val_acc: 0.7680
    Epoch 714/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8577 - acc: 0.7881 - val_loss: 0.8791 - val_acc: 0.7750
    Epoch 715/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8567 - acc: 0.7861 - val_loss: 0.8865 - val_acc: 0.7650
    Epoch 716/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8570 - acc: 0.7872 - val_loss: 0.8798 - val_acc: 0.7790
    Epoch 717/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8555 - acc: 0.7875 - val_loss: 0.8794 - val_acc: 0.7690
    Epoch 718/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8556 - acc: 0.7888 - val_loss: 0.8802 - val_acc: 0.7760
    Epoch 719/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8554 - acc: 0.7879 - val_loss: 0.8878 - val_acc: 0.7720
    Epoch 720/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8542 - acc: 0.7892 - val_loss: 0.8827 - val_acc: 0.7630
    Epoch 721/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8553 - acc: 0.7869 - val_loss: 0.8850 - val_acc: 0.7740
    Epoch 722/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8555 - acc: 0.7856 - val_loss: 0.8881 - val_acc: 0.7750
    Epoch 723/1000
    7500/7500 [==============================] - 0s 24us/step - loss: 0.8531 - acc: 0.7888 - val_loss: 0.8868 - val_acc: 0.7720
    Epoch 724/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8547 - acc: 0.7872 - val_loss: 0.8804 - val_acc: 0.7740
    Epoch 725/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8552 - acc: 0.7867 - val_loss: 0.8799 - val_acc: 0.7660
    Epoch 726/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8550 - acc: 0.7875 - val_loss: 0.8773 - val_acc: 0.7790
    Epoch 727/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8541 - acc: 0.7867 - val_loss: 0.8778 - val_acc: 0.7790
    Epoch 728/1000
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8556 - acc: 0.7880 - val_loss: 0.8768 - val_acc: 0.7740
    Epoch 729/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8545 - acc: 0.7876 - val_loss: 0.8845 - val_acc: 0.7570
    Epoch 730/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8534 - acc: 0.7879 - val_loss: 0.8850 - val_acc: 0.7700
    Epoch 731/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8546 - acc: 0.7881 - val_loss: 0.8927 - val_acc: 0.7640
    Epoch 732/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8547 - acc: 0.7888 - val_loss: 0.8765 - val_acc: 0.7770
    Epoch 733/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8542 - acc: 0.7877 - val_loss: 0.8846 - val_acc: 0.7620
    Epoch 734/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8529 - acc: 0.7884 - val_loss: 0.8797 - val_acc: 0.7690
    Epoch 735/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8535 - acc: 0.7871 - val_loss: 0.8825 - val_acc: 0.7690
    Epoch 736/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8541 - acc: 0.7883 - val_loss: 0.9048 - val_acc: 0.7520
    Epoch 737/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8541 - acc: 0.7857 - val_loss: 0.8794 - val_acc: 0.7720
    Epoch 738/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8543 - acc: 0.7877 - val_loss: 0.8813 - val_acc: 0.7730
    Epoch 739/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8521 - acc: 0.7877 - val_loss: 0.8790 - val_acc: 0.7610
    Epoch 740/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8530 - acc: 0.7861 - val_loss: 0.8771 - val_acc: 0.7790
    Epoch 741/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8522 - acc: 0.7885 - val_loss: 0.8771 - val_acc: 0.7750
    Epoch 742/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8530 - acc: 0.7875 - val_loss: 0.8800 - val_acc: 0.7770
    Epoch 743/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8522 - acc: 0.7863 - val_loss: 0.8798 - val_acc: 0.7620
    Epoch 744/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8517 - acc: 0.7883 - val_loss: 0.8860 - val_acc: 0.7620
    Epoch 745/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8534 - acc: 0.7867 - val_loss: 0.8824 - val_acc: 0.7680
    Epoch 746/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8517 - acc: 0.7897 - val_loss: 0.8847 - val_acc: 0.7670
    Epoch 747/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8513 - acc: 0.7877 - val_loss: 0.8908 - val_acc: 0.7630
    Epoch 748/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8533 - acc: 0.7873 - val_loss: 0.8952 - val_acc: 0.7640
    Epoch 749/1000
    7500/7500 [==============================] - 0s 43us/step - loss: 0.8515 - acc: 0.7877 - val_loss: 0.9081 - val_acc: 0.7580
    Epoch 750/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8524 - acc: 0.7869 - val_loss: 0.8971 - val_acc: 0.7570
    Epoch 751/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8524 - acc: 0.7879 - val_loss: 0.8761 - val_acc: 0.7690
    Epoch 752/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8515 - acc: 0.7881 - val_loss: 0.8807 - val_acc: 0.7690
    Epoch 753/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8515 - acc: 0.7876 - val_loss: 0.8847 - val_acc: 0.7680
    Epoch 754/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8525 - acc: 0.7901 - val_loss: 0.8848 - val_acc: 0.7630
    Epoch 755/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8516 - acc: 0.7891 - val_loss: 0.8805 - val_acc: 0.7660
    Epoch 756/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8515 - acc: 0.7881 - val_loss: 0.8771 - val_acc: 0.7740
    Epoch 757/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8502 - acc: 0.7880 - val_loss: 0.8824 - val_acc: 0.7660
    Epoch 758/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8504 - acc: 0.7897 - val_loss: 0.8898 - val_acc: 0.7520
    Epoch 759/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8512 - acc: 0.7885 - val_loss: 0.8823 - val_acc: 0.7660
    Epoch 760/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8509 - acc: 0.7871 - val_loss: 0.8852 - val_acc: 0.7650
    Epoch 761/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8501 - acc: 0.7884 - val_loss: 0.8866 - val_acc: 0.7630
    Epoch 762/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8504 - acc: 0.7880 - val_loss: 0.9056 - val_acc: 0.7660
    Epoch 763/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8526 - acc: 0.7872 - val_loss: 0.8780 - val_acc: 0.7710
    Epoch 764/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8492 - acc: 0.7908 - val_loss: 0.9256 - val_acc: 0.7600
    Epoch 765/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8505 - acc: 0.7871 - val_loss: 0.8952 - val_acc: 0.7630
    Epoch 766/1000
    7500/7500 [==============================] - 0s 43us/step - loss: 0.8510 - acc: 0.7888 - val_loss: 0.8824 - val_acc: 0.7680
    Epoch 767/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8502 - acc: 0.7888 - val_loss: 0.8754 - val_acc: 0.7660
    Epoch 768/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8494 - acc: 0.7873 - val_loss: 0.8806 - val_acc: 0.7650
    Epoch 769/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.8499 - acc: 0.7888 - val_loss: 0.8784 - val_acc: 0.7610
    Epoch 770/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8493 - acc: 0.7896 - val_loss: 0.8749 - val_acc: 0.7730
    Epoch 771/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8489 - acc: 0.7883 - val_loss: 0.8865 - val_acc: 0.7570
    Epoch 772/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8503 - acc: 0.7861 - val_loss: 0.8771 - val_acc: 0.7730
    Epoch 773/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8514 - acc: 0.7860 - val_loss: 0.8727 - val_acc: 0.7780
    Epoch 774/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8473 - acc: 0.7901 - val_loss: 0.8729 - val_acc: 0.7810
    Epoch 775/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8500 - acc: 0.7881 - val_loss: 0.8779 - val_acc: 0.7670
    Epoch 776/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8501 - acc: 0.7884 - val_loss: 0.8944 - val_acc: 0.7730
    Epoch 777/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8512 - acc: 0.7875 - val_loss: 0.8777 - val_acc: 0.7730
    Epoch 778/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8491 - acc: 0.7883 - val_loss: 0.8754 - val_acc: 0.7670
    Epoch 779/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8485 - acc: 0.7908 - val_loss: 0.8877 - val_acc: 0.7600
    Epoch 780/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8483 - acc: 0.7869 - val_loss: 0.8789 - val_acc: 0.7720
    Epoch 781/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8484 - acc: 0.7879 - val_loss: 0.8857 - val_acc: 0.7590
    Epoch 782/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8477 - acc: 0.7885 - val_loss: 0.8737 - val_acc: 0.7710
    Epoch 783/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8476 - acc: 0.7881 - val_loss: 0.8722 - val_acc: 0.7660
    Epoch 784/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8476 - acc: 0.7900 - val_loss: 0.8985 - val_acc: 0.7710
    Epoch 785/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8482 - acc: 0.7865 - val_loss: 0.8861 - val_acc: 0.7640
    Epoch 786/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8490 - acc: 0.7897 - val_loss: 0.8863 - val_acc: 0.7690
    Epoch 787/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8481 - acc: 0.7872 - val_loss: 0.8863 - val_acc: 0.7790
    Epoch 788/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8472 - acc: 0.7884 - val_loss: 0.8747 - val_acc: 0.7760
    Epoch 789/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8478 - acc: 0.7921 - val_loss: 0.8806 - val_acc: 0.7730
    Epoch 790/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8463 - acc: 0.7892 - val_loss: 0.9155 - val_acc: 0.7620
    Epoch 791/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8494 - acc: 0.7888 - val_loss: 0.8986 - val_acc: 0.7580
    Epoch 792/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8471 - acc: 0.7880 - val_loss: 0.8724 - val_acc: 0.7720
    Epoch 793/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8464 - acc: 0.7911 - val_loss: 0.8766 - val_acc: 0.7700
    Epoch 794/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8464 - acc: 0.7901 - val_loss: 0.8803 - val_acc: 0.7680
    Epoch 795/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8463 - acc: 0.7919 - val_loss: 0.8756 - val_acc: 0.7760
    Epoch 796/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8457 - acc: 0.7896 - val_loss: 0.8704 - val_acc: 0.7780
    Epoch 797/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8467 - acc: 0.7897 - val_loss: 0.8741 - val_acc: 0.7680
    Epoch 798/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8451 - acc: 0.7889 - val_loss: 0.8812 - val_acc: 0.7730
    Epoch 799/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8475 - acc: 0.7888 - val_loss: 0.8841 - val_acc: 0.7680
    Epoch 800/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8459 - acc: 0.7901 - val_loss: 0.8758 - val_acc: 0.7640
    Epoch 801/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8454 - acc: 0.7911 - val_loss: 0.8855 - val_acc: 0.7680
    Epoch 802/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8478 - acc: 0.7895 - val_loss: 0.8714 - val_acc: 0.7760
    Epoch 803/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8447 - acc: 0.7931 - val_loss: 0.8870 - val_acc: 0.7630
    Epoch 804/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8463 - acc: 0.7917 - val_loss: 0.8808 - val_acc: 0.7760
    Epoch 805/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8469 - acc: 0.7899 - val_loss: 0.8759 - val_acc: 0.7670
    Epoch 806/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8457 - acc: 0.7880 - val_loss: 0.8780 - val_acc: 0.7730
    Epoch 807/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 0.8454 - acc: 0.7921 - val_loss: 0.8760 - val_acc: 0.7710
    Epoch 808/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8456 - acc: 0.7892 - val_loss: 0.8776 - val_acc: 0.7700
    Epoch 809/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8449 - acc: 0.7893 - val_loss: 0.8746 - val_acc: 0.7680
    Epoch 810/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8450 - acc: 0.7915 - val_loss: 0.8837 - val_acc: 0.7720
    Epoch 811/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8455 - acc: 0.7911 - val_loss: 0.8805 - val_acc: 0.7600
    Epoch 812/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8452 - acc: 0.7899 - val_loss: 0.8708 - val_acc: 0.7720
    Epoch 813/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8440 - acc: 0.7892 - val_loss: 0.8731 - val_acc: 0.7820
    Epoch 814/1000
    7500/7500 [==============================] - 0s 46us/step - loss: 0.8433 - acc: 0.7921 - val_loss: 0.8840 - val_acc: 0.7670
    Epoch 815/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8447 - acc: 0.7893 - val_loss: 0.8840 - val_acc: 0.7670
    Epoch 816/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8438 - acc: 0.7905 - val_loss: 0.8796 - val_acc: 0.7650
    Epoch 817/1000
    7500/7500 [==============================] - 0s 41us/step - loss: 0.8442 - acc: 0.7887 - val_loss: 0.8811 - val_acc: 0.7710
    Epoch 818/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8439 - acc: 0.7887 - val_loss: 0.8814 - val_acc: 0.7660
    Epoch 819/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8444 - acc: 0.7895 - val_loss: 0.9006 - val_acc: 0.7600
    Epoch 820/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8439 - acc: 0.7895 - val_loss: 0.8831 - val_acc: 0.7550
    Epoch 821/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8427 - acc: 0.7885 - val_loss: 0.8728 - val_acc: 0.7700
    Epoch 822/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8431 - acc: 0.7927 - val_loss: 0.8718 - val_acc: 0.7760
    Epoch 823/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8430 - acc: 0.7899 - val_loss: 0.8770 - val_acc: 0.7700
    Epoch 824/1000
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8447 - acc: 0.7880 - val_loss: 0.8745 - val_acc: 0.7650
    Epoch 825/1000
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8428 - acc: 0.7901 - val_loss: 0.8729 - val_acc: 0.7760
    Epoch 826/1000
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8425 - acc: 0.7915 - val_loss: 0.8753 - val_acc: 0.7740
    Epoch 827/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8437 - acc: 0.7893 - val_loss: 0.8756 - val_acc: 0.7630
    Epoch 828/1000
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8419 - acc: 0.7913 - val_loss: 0.8763 - val_acc: 0.7740
    Epoch 829/1000
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8435 - acc: 0.7900 - val_loss: 0.8850 - val_acc: 0.7780
    Epoch 830/1000
    7500/7500 [==============================] - 0s 24us/step - loss: 0.8432 - acc: 0.7911 - val_loss: 0.8700 - val_acc: 0.7760
    Epoch 831/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8421 - acc: 0.7883 - val_loss: 0.8707 - val_acc: 0.7790
    Epoch 832/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8446 - acc: 0.7892 - val_loss: 0.8699 - val_acc: 0.7750
    Epoch 833/1000
    7500/7500 [==============================] - 0s 23us/step - loss: 0.8420 - acc: 0.7919 - val_loss: 0.8727 - val_acc: 0.7700
    Epoch 834/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8420 - acc: 0.7923 - val_loss: 0.8736 - val_acc: 0.7640
    Epoch 835/1000
    7500/7500 [==============================] - 0s 33us/step - loss: 0.8415 - acc: 0.7896 - val_loss: 0.8838 - val_acc: 0.7550
    Epoch 836/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8415 - acc: 0.7917 - val_loss: 0.8744 - val_acc: 0.7670
    Epoch 837/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8430 - acc: 0.7892 - val_loss: 0.8892 - val_acc: 0.7550
    Epoch 838/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8431 - acc: 0.7903 - val_loss: 0.8729 - val_acc: 0.7710
    Epoch 839/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8410 - acc: 0.7931 - val_loss: 0.8717 - val_acc: 0.7700
    Epoch 840/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8413 - acc: 0.7903 - val_loss: 0.8742 - val_acc: 0.7740
    Epoch 841/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8417 - acc: 0.7908 - val_loss: 0.8823 - val_acc: 0.7750
    Epoch 842/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8423 - acc: 0.7908 - val_loss: 0.8700 - val_acc: 0.7700
    Epoch 843/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8403 - acc: 0.7907 - val_loss: 0.8698 - val_acc: 0.7720
    Epoch 844/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8405 - acc: 0.7913 - val_loss: 0.8691 - val_acc: 0.7740
    Epoch 845/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8429 - acc: 0.7896 - val_loss: 0.8698 - val_acc: 0.7740
    Epoch 846/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8391 - acc: 0.7912 - val_loss: 0.8722 - val_acc: 0.7700
    Epoch 847/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8410 - acc: 0.7901 - val_loss: 0.9618 - val_acc: 0.7380
    Epoch 848/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8452 - acc: 0.7873 - val_loss: 0.8821 - val_acc: 0.7790
    Epoch 849/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8439 - acc: 0.7900 - val_loss: 0.8696 - val_acc: 0.7820
    Epoch 850/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8402 - acc: 0.7912 - val_loss: 0.8697 - val_acc: 0.7610
    Epoch 851/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8392 - acc: 0.7917 - val_loss: 0.8730 - val_acc: 0.7640
    Epoch 852/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8392 - acc: 0.7928 - val_loss: 0.8805 - val_acc: 0.7570
    Epoch 853/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8410 - acc: 0.7911 - val_loss: 0.8692 - val_acc: 0.7690
    Epoch 854/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8393 - acc: 0.7912 - val_loss: 0.8701 - val_acc: 0.7720
    Epoch 855/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8404 - acc: 0.7904 - val_loss: 0.8670 - val_acc: 0.7790
    Epoch 856/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8417 - acc: 0.7921 - val_loss: 0.8851 - val_acc: 0.7690
    Epoch 857/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8400 - acc: 0.7905 - val_loss: 0.8999 - val_acc: 0.7610
    Epoch 858/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8409 - acc: 0.7907 - val_loss: 0.8827 - val_acc: 0.7700
    Epoch 859/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8418 - acc: 0.7880 - val_loss: 0.8833 - val_acc: 0.7710
    Epoch 860/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8380 - acc: 0.7939 - val_loss: 0.8869 - val_acc: 0.7600
    Epoch 861/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8395 - acc: 0.7927 - val_loss: 0.8714 - val_acc: 0.7600
    Epoch 862/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8386 - acc: 0.7903 - val_loss: 0.8788 - val_acc: 0.7680
    Epoch 863/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8409 - acc: 0.7909 - val_loss: 0.8762 - val_acc: 0.7700
    Epoch 864/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8391 - acc: 0.7908 - val_loss: 0.8687 - val_acc: 0.7720
    Epoch 865/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8390 - acc: 0.7929 - val_loss: 0.8948 - val_acc: 0.7550
    Epoch 866/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8384 - acc: 0.7925 - val_loss: 0.8860 - val_acc: 0.7680
    Epoch 867/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8376 - acc: 0.7923 - val_loss: 0.8703 - val_acc: 0.7740
    Epoch 868/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8389 - acc: 0.7901 - val_loss: 0.8712 - val_acc: 0.7650
    Epoch 869/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8403 - acc: 0.7921 - val_loss: 0.8751 - val_acc: 0.7640
    Epoch 870/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8391 - acc: 0.7933 - val_loss: 0.8676 - val_acc: 0.7740
    Epoch 871/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8390 - acc: 0.7920 - val_loss: 0.8739 - val_acc: 0.7710
    Epoch 872/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8379 - acc: 0.7924 - val_loss: 0.8740 - val_acc: 0.7750
    Epoch 873/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8370 - acc: 0.7909 - val_loss: 0.8694 - val_acc: 0.7780
    Epoch 874/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8384 - acc: 0.7924 - val_loss: 0.8706 - val_acc: 0.7780
    Epoch 875/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8360 - acc: 0.7919 - val_loss: 0.8774 - val_acc: 0.7750
    Epoch 876/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8364 - acc: 0.7953 - val_loss: 0.8750 - val_acc: 0.7700
    Epoch 877/1000
    7500/7500 [==============================] - 0s 47us/step - loss: 0.8387 - acc: 0.7908 - val_loss: 0.8785 - val_acc: 0.7660
    Epoch 878/1000
    7500/7500 [==============================] - 0s 53us/step - loss: 0.8378 - acc: 0.7920 - val_loss: 0.8692 - val_acc: 0.7620
    Epoch 879/1000
    7500/7500 [==============================] - 0s 45us/step - loss: 0.8357 - acc: 0.7912 - val_loss: 0.8671 - val_acc: 0.7770
    Epoch 880/1000
    7500/7500 [==============================] - 0s 52us/step - loss: 0.8378 - acc: 0.7921 - val_loss: 0.8758 - val_acc: 0.7720
    Epoch 881/1000
    7500/7500 [==============================] - 0s 45us/step - loss: 0.8380 - acc: 0.7924 - val_loss: 0.8648 - val_acc: 0.7760
    Epoch 882/1000
    7500/7500 [==============================] - 0s 47us/step - loss: 0.8372 - acc: 0.7929 - val_loss: 0.8806 - val_acc: 0.7570
    Epoch 883/1000
    7500/7500 [==============================] - 0s 45us/step - loss: 0.8369 - acc: 0.7915 - val_loss: 0.8671 - val_acc: 0.7760
    Epoch 884/1000
    7500/7500 [==============================] - 0s 49us/step - loss: 0.8381 - acc: 0.7924 - val_loss: 0.8780 - val_acc: 0.7590
    Epoch 885/1000
    7500/7500 [==============================] - 0s 47us/step - loss: 0.8364 - acc: 0.7929 - val_loss: 0.8660 - val_acc: 0.7780
    Epoch 886/1000
    7500/7500 [==============================] - 0s 53us/step - loss: 0.8360 - acc: 0.7935 - val_loss: 0.8719 - val_acc: 0.7640
    Epoch 887/1000
    7500/7500 [==============================] - 0s 45us/step - loss: 0.8375 - acc: 0.7937 - val_loss: 0.8741 - val_acc: 0.7720
    Epoch 888/1000
    7500/7500 [==============================] - 0s 51us/step - loss: 0.8369 - acc: 0.7924 - val_loss: 0.8723 - val_acc: 0.7700
    Epoch 889/1000
    7500/7500 [==============================] - 0s 43us/step - loss: 0.8362 - acc: 0.7937 - val_loss: 0.8802 - val_acc: 0.7640
    Epoch 890/1000
    7500/7500 [==============================] - 0s 46us/step - loss: 0.8361 - acc: 0.7883 - val_loss: 0.8674 - val_acc: 0.7750
    Epoch 891/1000
    7500/7500 [==============================] - 0s 46us/step - loss: 0.8349 - acc: 0.7940 - val_loss: 0.8714 - val_acc: 0.7690
    Epoch 892/1000
    7500/7500 [==============================] - 0s 45us/step - loss: 0.8356 - acc: 0.7939 - val_loss: 0.8732 - val_acc: 0.7630
    Epoch 893/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8354 - acc: 0.7935 - val_loss: 0.8716 - val_acc: 0.7700
    Epoch 894/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8361 - acc: 0.7916 - val_loss: 0.8665 - val_acc: 0.7740
    Epoch 895/1000
    7500/7500 [==============================] - 0s 43us/step - loss: 0.8353 - acc: 0.7949 - val_loss: 0.8713 - val_acc: 0.7790
    Epoch 896/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8348 - acc: 0.7917 - val_loss: 0.8746 - val_acc: 0.7700
    Epoch 897/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8351 - acc: 0.7912 - val_loss: 0.8771 - val_acc: 0.7750
    Epoch 898/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8370 - acc: 0.7927 - val_loss: 0.8718 - val_acc: 0.7730
    Epoch 899/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8361 - acc: 0.7931 - val_loss: 0.8860 - val_acc: 0.7580
    Epoch 900/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8353 - acc: 0.7915 - val_loss: 0.8719 - val_acc: 0.7750
    Epoch 901/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8356 - acc: 0.7935 - val_loss: 0.8811 - val_acc: 0.7600
    Epoch 902/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8350 - acc: 0.7932 - val_loss: 0.8670 - val_acc: 0.7700
    Epoch 903/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8361 - acc: 0.7917 - val_loss: 0.8775 - val_acc: 0.7670
    Epoch 904/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8366 - acc: 0.7927 - val_loss: 0.8785 - val_acc: 0.7630
    Epoch 905/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8349 - acc: 0.7931 - val_loss: 0.8660 - val_acc: 0.7820
    Epoch 906/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8348 - acc: 0.7925 - val_loss: 0.8749 - val_acc: 0.7630
    Epoch 907/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8355 - acc: 0.7924 - val_loss: 0.8714 - val_acc: 0.7740
    Epoch 908/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8357 - acc: 0.7931 - val_loss: 0.8642 - val_acc: 0.7800
    Epoch 909/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8336 - acc: 0.7924 - val_loss: 0.8703 - val_acc: 0.7630
    Epoch 910/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8331 - acc: 0.7953 - val_loss: 0.8805 - val_acc: 0.7660
    Epoch 911/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8329 - acc: 0.7939 - val_loss: 0.8785 - val_acc: 0.7590
    Epoch 912/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8348 - acc: 0.7917 - val_loss: 0.8721 - val_acc: 0.7800
    Epoch 913/1000
    7500/7500 [==============================] - 0s 44us/step - loss: 0.8345 - acc: 0.7949 - val_loss: 0.9010 - val_acc: 0.7670
    Epoch 914/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8361 - acc: 0.7913 - val_loss: 0.8663 - val_acc: 0.7790
    Epoch 915/1000
    7500/7500 [==============================] - 0s 46us/step - loss: 0.8332 - acc: 0.7947 - val_loss: 0.8676 - val_acc: 0.7700
    Epoch 916/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8328 - acc: 0.7931 - val_loss: 0.8873 - val_acc: 0.7640
    Epoch 917/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8343 - acc: 0.7940 - val_loss: 0.8788 - val_acc: 0.7690
    Epoch 918/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8333 - acc: 0.7933 - val_loss: 0.9082 - val_acc: 0.7560
    Epoch 919/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8334 - acc: 0.7937 - val_loss: 0.8792 - val_acc: 0.7650
    Epoch 920/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8334 - acc: 0.7905 - val_loss: 0.8708 - val_acc: 0.7700
    Epoch 921/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8318 - acc: 0.7947 - val_loss: 0.8702 - val_acc: 0.7780
    Epoch 922/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8331 - acc: 0.7961 - val_loss: 0.8727 - val_acc: 0.7710
    Epoch 923/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8334 - acc: 0.7940 - val_loss: 0.8657 - val_acc: 0.7670
    Epoch 924/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8320 - acc: 0.7917 - val_loss: 0.8748 - val_acc: 0.7620
    Epoch 925/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8348 - acc: 0.7956 - val_loss: 0.8673 - val_acc: 0.7710
    Epoch 926/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8332 - acc: 0.7928 - val_loss: 0.8731 - val_acc: 0.7700
    Epoch 927/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8332 - acc: 0.7948 - val_loss: 0.8649 - val_acc: 0.7750
    Epoch 928/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8340 - acc: 0.7935 - val_loss: 0.8692 - val_acc: 0.7730
    Epoch 929/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8319 - acc: 0.7949 - val_loss: 0.8743 - val_acc: 0.7750
    Epoch 930/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8353 - acc: 0.7935 - val_loss: 0.8815 - val_acc: 0.7590
    Epoch 931/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8325 - acc: 0.7924 - val_loss: 0.8768 - val_acc: 0.7660
    Epoch 932/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8309 - acc: 0.7928 - val_loss: 0.8755 - val_acc: 0.7800
    Epoch 933/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8327 - acc: 0.7940 - val_loss: 0.8750 - val_acc: 0.7760
    Epoch 934/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8318 - acc: 0.7943 - val_loss: 0.8758 - val_acc: 0.7630
    Epoch 935/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8316 - acc: 0.7937 - val_loss: 0.8810 - val_acc: 0.7650
    Epoch 936/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8310 - acc: 0.7932 - val_loss: 0.8650 - val_acc: 0.7790
    Epoch 937/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8313 - acc: 0.7931 - val_loss: 0.8739 - val_acc: 0.7790
    Epoch 938/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8301 - acc: 0.7935 - val_loss: 0.8809 - val_acc: 0.7660
    Epoch 939/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8326 - acc: 0.7952 - val_loss: 0.8729 - val_acc: 0.7660
    Epoch 940/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8318 - acc: 0.7915 - val_loss: 0.8750 - val_acc: 0.7770
    Epoch 941/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8314 - acc: 0.7937 - val_loss: 0.8660 - val_acc: 0.7780
    Epoch 942/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8322 - acc: 0.7944 - val_loss: 0.8653 - val_acc: 0.7800
    Epoch 943/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8303 - acc: 0.7969 - val_loss: 0.8620 - val_acc: 0.7780
    Epoch 944/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8307 - acc: 0.7940 - val_loss: 0.9117 - val_acc: 0.7530
    Epoch 945/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8319 - acc: 0.7949 - val_loss: 0.8722 - val_acc: 0.7640
    Epoch 946/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8318 - acc: 0.7956 - val_loss: 0.9060 - val_acc: 0.7530
    Epoch 947/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8312 - acc: 0.7940 - val_loss: 0.8882 - val_acc: 0.7560
    Epoch 948/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8293 - acc: 0.7953 - val_loss: 0.8652 - val_acc: 0.7710
    Epoch 949/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8320 - acc: 0.7960 - val_loss: 0.8838 - val_acc: 0.7720
    Epoch 950/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8297 - acc: 0.7943 - val_loss: 0.8648 - val_acc: 0.7820
    Epoch 951/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8288 - acc: 0.7957 - val_loss: 0.8646 - val_acc: 0.7750
    Epoch 952/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8293 - acc: 0.7955 - val_loss: 0.8815 - val_acc: 0.7700
    Epoch 953/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8303 - acc: 0.7967 - val_loss: 0.8624 - val_acc: 0.7730
    Epoch 954/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8296 - acc: 0.7955 - val_loss: 0.8666 - val_acc: 0.7700
    Epoch 955/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8304 - acc: 0.7936 - val_loss: 0.8820 - val_acc: 0.7740
    Epoch 956/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8294 - acc: 0.7951 - val_loss: 0.8776 - val_acc: 0.7760
    Epoch 957/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8311 - acc: 0.7921 - val_loss: 0.9091 - val_acc: 0.7580
    Epoch 958/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8301 - acc: 0.7952 - val_loss: 0.8609 - val_acc: 0.7770
    Epoch 959/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8293 - acc: 0.7951 - val_loss: 0.8673 - val_acc: 0.7700
    Epoch 960/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8315 - acc: 0.7940 - val_loss: 0.8695 - val_acc: 0.7740
    Epoch 961/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8308 - acc: 0.7948 - val_loss: 0.8662 - val_acc: 0.7700
    Epoch 962/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8289 - acc: 0.7959 - val_loss: 0.8629 - val_acc: 0.7800
    Epoch 963/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8269 - acc: 0.7956 - val_loss: 0.8621 - val_acc: 0.7790
    Epoch 964/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8309 - acc: 0.7948 - val_loss: 0.8687 - val_acc: 0.7730
    Epoch 965/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8297 - acc: 0.7956 - val_loss: 0.8656 - val_acc: 0.7730
    Epoch 966/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8275 - acc: 0.7956 - val_loss: 0.8643 - val_acc: 0.7700
    Epoch 967/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8281 - acc: 0.7937 - val_loss: 0.8645 - val_acc: 0.7780
    Epoch 968/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8269 - acc: 0.7956 - val_loss: 0.8661 - val_acc: 0.7810
    Epoch 969/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8286 - acc: 0.7937 - val_loss: 0.8648 - val_acc: 0.7770
    Epoch 970/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8291 - acc: 0.7952 - val_loss: 0.8656 - val_acc: 0.7810
    Epoch 971/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8297 - acc: 0.7952 - val_loss: 0.8651 - val_acc: 0.7770
    Epoch 972/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8281 - acc: 0.7944 - val_loss: 0.8798 - val_acc: 0.7690
    Epoch 973/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8282 - acc: 0.7987 - val_loss: 0.8627 - val_acc: 0.7750
    Epoch 974/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8275 - acc: 0.7961 - val_loss: 0.8654 - val_acc: 0.7710
    Epoch 975/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8281 - acc: 0.7956 - val_loss: 0.8720 - val_acc: 0.7670
    Epoch 976/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8328 - acc: 0.7943 - val_loss: 0.9056 - val_acc: 0.7610
    Epoch 977/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8312 - acc: 0.7955 - val_loss: 0.8746 - val_acc: 0.7730
    Epoch 978/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8295 - acc: 0.7947 - val_loss: 0.8683 - val_acc: 0.7670
    Epoch 979/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8262 - acc: 0.7971 - val_loss: 0.8672 - val_acc: 0.7750
    Epoch 980/1000
    7500/7500 [==============================] - 0s 40us/step - loss: 0.8269 - acc: 0.7943 - val_loss: 0.8624 - val_acc: 0.7740
    Epoch 981/1000
    7500/7500 [==============================] - 0s 42us/step - loss: 0.8274 - acc: 0.7972 - val_loss: 0.8839 - val_acc: 0.7690
    Epoch 982/1000
    7500/7500 [==============================] - 0s 39us/step - loss: 0.8274 - acc: 0.7949 - val_loss: 0.8687 - val_acc: 0.7670
    Epoch 983/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8260 - acc: 0.7971 - val_loss: 0.9091 - val_acc: 0.7640
    Epoch 984/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8273 - acc: 0.7960 - val_loss: 0.8981 - val_acc: 0.7720
    Epoch 985/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8323 - acc: 0.7923 - val_loss: 0.8765 - val_acc: 0.7760
    Epoch 986/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8276 - acc: 0.7931 - val_loss: 0.8668 - val_acc: 0.7680
    Epoch 987/1000
    7500/7500 [==============================] - 0s 38us/step - loss: 0.8286 - acc: 0.7952 - val_loss: 0.8701 - val_acc: 0.7690
    Epoch 988/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8278 - acc: 0.7963 - val_loss: 0.8693 - val_acc: 0.7620
    Epoch 989/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8265 - acc: 0.7953 - val_loss: 0.8649 - val_acc: 0.7690
    Epoch 990/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8269 - acc: 0.7981 - val_loss: 0.8839 - val_acc: 0.7730
    Epoch 991/1000
    7500/7500 [==============================] - 0s 35us/step - loss: 0.8277 - acc: 0.7940 - val_loss: 0.8716 - val_acc: 0.7770
    Epoch 992/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8268 - acc: 0.7971 - val_loss: 0.8685 - val_acc: 0.7830
    Epoch 993/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8255 - acc: 0.7952 - val_loss: 0.8832 - val_acc: 0.7540
    Epoch 994/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8265 - acc: 0.7960 - val_loss: 0.8651 - val_acc: 0.7650
    Epoch 995/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8261 - acc: 0.7964 - val_loss: 0.8932 - val_acc: 0.7690
    Epoch 996/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8298 - acc: 0.7944 - val_loss: 0.8595 - val_acc: 0.7830
    Epoch 997/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8243 - acc: 0.7968 - val_loss: 0.8693 - val_acc: 0.7730
    Epoch 998/1000
    7500/7500 [==============================] - 0s 36us/step - loss: 0.8247 - acc: 0.7968 - val_loss: 0.8596 - val_acc: 0.7790
    Epoch 999/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8254 - acc: 0.7988 - val_loss: 0.8680 - val_acc: 0.7660
    Epoch 1000/1000
    7500/7500 [==============================] - 0s 37us/step - loss: 0.8262 - acc: 0.7965 - val_loss: 0.8655 - val_acc: 0.7780



```python
L1_model_dict = L1_model.history
plt.clf()

acc_values = L1_model_dict['acc'] 
val_acc_values = L1_model_dict['val_acc']

epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'g', label='Training acc L1')
plt.plot(epochs, val_acc_values, 'g,', label='Validation acc L1')
plt.title('Training & validation accuracy L2 vs regular')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


![png](index_files/index_60_0.png)



```python
results_train = model.evaluate(train_final, label_train_final)

results_test = model.evaluate(X_test, y_test)
```

    7500/7500 [==============================] - 0s 24us/step
    1500/1500 [==============================] - 0s 26us/step



```python
results_train
```




    [0.8237653533299764, 0.7967999999682108]




```python
results_test
```




    [0.966706668694814, 0.7499999998410543]



This is about the best result you've achieved so far, but you were training for quite a while! Next, experiment with dropout regularization to see if it offers any advantages.

## Dropout Regularization


```python
# ⏰ This cell may take about a minute to run
random.seed(123)
model = models.Sequential()
model.add(layers.Dropout(0.3, input_shape=(2000,)))
model.add(layers.Dense(50, activation='relu')) #2 hidden layers
model.add(layers.Dropout(0.3))
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

dropout_model = model.fit(train_final,
                    label_train_final,
                    epochs=200,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/200
    7500/7500 [==============================] - 1s 71us/step - loss: 2.0228 - acc: 0.1372 - val_loss: 1.9610 - val_acc: 0.1380
    Epoch 2/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.9760 - acc: 0.1439 - val_loss: 1.9430 - val_acc: 0.1630
    Epoch 3/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.9560 - acc: 0.1548 - val_loss: 1.9328 - val_acc: 0.1790
    Epoch 4/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.9462 - acc: 0.1639 - val_loss: 1.9249 - val_acc: 0.1890
    Epoch 5/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.9412 - acc: 0.1661 - val_loss: 1.9189 - val_acc: 0.2040
    Epoch 6/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.9261 - acc: 0.1800 - val_loss: 1.9124 - val_acc: 0.2090
    Epoch 7/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.9231 - acc: 0.1896 - val_loss: 1.9062 - val_acc: 0.2310
    Epoch 8/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.9152 - acc: 0.1916 - val_loss: 1.8993 - val_acc: 0.2430
    Epoch 9/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.9105 - acc: 0.1981 - val_loss: 1.8928 - val_acc: 0.2540
    Epoch 10/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.9032 - acc: 0.2036 - val_loss: 1.8854 - val_acc: 0.2590
    Epoch 11/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.8927 - acc: 0.2129 - val_loss: 1.8773 - val_acc: 0.2640
    Epoch 12/200
    7500/7500 [==============================] - 0s 28us/step - loss: 1.8881 - acc: 0.2221 - val_loss: 1.8688 - val_acc: 0.2660
    Epoch 13/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.8835 - acc: 0.2249 - val_loss: 1.8599 - val_acc: 0.2720
    Epoch 14/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.8740 - acc: 0.2337 - val_loss: 1.8489 - val_acc: 0.2840
    Epoch 15/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.8637 - acc: 0.2427 - val_loss: 1.8371 - val_acc: 0.2930
    Epoch 16/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.8544 - acc: 0.2432 - val_loss: 1.8235 - val_acc: 0.3030
    Epoch 17/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.8401 - acc: 0.2505 - val_loss: 1.8080 - val_acc: 0.3100
    Epoch 18/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.8334 - acc: 0.2677 - val_loss: 1.7913 - val_acc: 0.3120
    Epoch 19/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.8201 - acc: 0.2657 - val_loss: 1.7722 - val_acc: 0.3190
    Epoch 20/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.8009 - acc: 0.2712 - val_loss: 1.7512 - val_acc: 0.3240
    Epoch 21/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.7911 - acc: 0.2812 - val_loss: 1.7285 - val_acc: 0.3310
    Epoch 22/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.7735 - acc: 0.2939 - val_loss: 1.7043 - val_acc: 0.3420
    Epoch 23/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.7500 - acc: 0.3079 - val_loss: 1.6776 - val_acc: 0.3510
    Epoch 24/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.7402 - acc: 0.3083 - val_loss: 1.6486 - val_acc: 0.3800
    Epoch 25/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.7199 - acc: 0.3201 - val_loss: 1.6207 - val_acc: 0.3930
    Epoch 26/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.6974 - acc: 0.3253 - val_loss: 1.5931 - val_acc: 0.4070
    Epoch 27/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.6781 - acc: 0.3377 - val_loss: 1.5651 - val_acc: 0.4270
    Epoch 28/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.6711 - acc: 0.3471 - val_loss: 1.5406 - val_acc: 0.4440
    Epoch 29/200
    7500/7500 [==============================] - 0s 28us/step - loss: 1.6474 - acc: 0.3469 - val_loss: 1.5145 - val_acc: 0.4510
    Epoch 30/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.6281 - acc: 0.3521 - val_loss: 1.4876 - val_acc: 0.4610
    Epoch 31/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.6150 - acc: 0.3639 - val_loss: 1.4634 - val_acc: 0.4750
    Epoch 32/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.5985 - acc: 0.3748 - val_loss: 1.4395 - val_acc: 0.4940
    Epoch 33/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.5763 - acc: 0.3808 - val_loss: 1.4134 - val_acc: 0.5050
    Epoch 34/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.5539 - acc: 0.3969 - val_loss: 1.3892 - val_acc: 0.5280
    Epoch 35/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.5548 - acc: 0.3883 - val_loss: 1.3709 - val_acc: 0.5280
    Epoch 36/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.5287 - acc: 0.4059 - val_loss: 1.3478 - val_acc: 0.5470
    Epoch 37/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.5163 - acc: 0.4099 - val_loss: 1.3288 - val_acc: 0.5560
    Epoch 38/200
    7500/7500 [==============================] - 0s 28us/step - loss: 1.5083 - acc: 0.4137 - val_loss: 1.3094 - val_acc: 0.5720
    Epoch 39/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.4847 - acc: 0.4319 - val_loss: 1.2877 - val_acc: 0.5770
    Epoch 40/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.4855 - acc: 0.4204 - val_loss: 1.2714 - val_acc: 0.5920
    Epoch 41/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.4642 - acc: 0.4336 - val_loss: 1.2518 - val_acc: 0.6070
    Epoch 42/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.4565 - acc: 0.4380 - val_loss: 1.2329 - val_acc: 0.6190
    Epoch 43/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.4303 - acc: 0.4479 - val_loss: 1.2134 - val_acc: 0.6380
    Epoch 44/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.4283 - acc: 0.4561 - val_loss: 1.1985 - val_acc: 0.6450
    Epoch 45/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.3998 - acc: 0.4632 - val_loss: 1.1791 - val_acc: 0.6420
    Epoch 46/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.3886 - acc: 0.4664 - val_loss: 1.1616 - val_acc: 0.6460
    Epoch 47/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.3729 - acc: 0.4784 - val_loss: 1.1441 - val_acc: 0.6540
    Epoch 48/200
    7500/7500 [==============================] - 0s 28us/step - loss: 1.3671 - acc: 0.4807 - val_loss: 1.1300 - val_acc: 0.6640
    Epoch 49/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.3640 - acc: 0.4808 - val_loss: 1.1187 - val_acc: 0.6570
    Epoch 50/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.3373 - acc: 0.4841 - val_loss: 1.0995 - val_acc: 0.6660
    Epoch 51/200
    7500/7500 [==============================] - 0s 28us/step - loss: 1.3246 - acc: 0.4999 - val_loss: 1.0832 - val_acc: 0.6730
    Epoch 52/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.3099 - acc: 0.5103 - val_loss: 1.0693 - val_acc: 0.6760
    Epoch 53/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.3033 - acc: 0.5064 - val_loss: 1.0562 - val_acc: 0.6850
    Epoch 54/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2944 - acc: 0.5116 - val_loss: 1.0446 - val_acc: 0.6820
    Epoch 55/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2866 - acc: 0.5151 - val_loss: 1.0344 - val_acc: 0.6850
    Epoch 56/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2655 - acc: 0.5233 - val_loss: 1.0180 - val_acc: 0.6880
    Epoch 57/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2563 - acc: 0.5239 - val_loss: 1.0041 - val_acc: 0.6920
    Epoch 58/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2481 - acc: 0.5256 - val_loss: 0.9925 - val_acc: 0.6950
    Epoch 59/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2476 - acc: 0.5280 - val_loss: 0.9800 - val_acc: 0.6940
    Epoch 60/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2354 - acc: 0.5300 - val_loss: 0.9684 - val_acc: 0.7050
    Epoch 61/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.2150 - acc: 0.5400 - val_loss: 0.9589 - val_acc: 0.7120
    Epoch 62/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.2109 - acc: 0.5513 - val_loss: 0.9459 - val_acc: 0.7030
    Epoch 63/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.2099 - acc: 0.5472 - val_loss: 0.9360 - val_acc: 0.7070
    Epoch 64/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.1975 - acc: 0.5485 - val_loss: 0.9275 - val_acc: 0.7140
    Epoch 65/200
    7500/7500 [==============================] - 0s 28us/step - loss: 1.1849 - acc: 0.5585 - val_loss: 0.9184 - val_acc: 0.7120
    Epoch 66/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.1837 - acc: 0.5669 - val_loss: 0.9071 - val_acc: 0.7230
    Epoch 67/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.1751 - acc: 0.5639 - val_loss: 0.8986 - val_acc: 0.7200
    Epoch 68/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.1549 - acc: 0.5747 - val_loss: 0.8873 - val_acc: 0.7260
    Epoch 69/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.1635 - acc: 0.5651 - val_loss: 0.8807 - val_acc: 0.7260
    Epoch 70/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.1557 - acc: 0.5672 - val_loss: 0.8725 - val_acc: 0.7280
    Epoch 71/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.1337 - acc: 0.5803 - val_loss: 0.8605 - val_acc: 0.7310
    Epoch 72/200
    7500/7500 [==============================] - 0s 30us/step - loss: 1.1170 - acc: 0.5859 - val_loss: 0.8531 - val_acc: 0.7290
    Epoch 73/200
    7500/7500 [==============================] - 0s 32us/step - loss: 1.1179 - acc: 0.5797 - val_loss: 0.8437 - val_acc: 0.7330
    Epoch 74/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.1314 - acc: 0.5715 - val_loss: 0.8400 - val_acc: 0.7340
    Epoch 75/200
    7500/7500 [==============================] - 0s 32us/step - loss: 1.0935 - acc: 0.5948 - val_loss: 0.8310 - val_acc: 0.7370
    Epoch 76/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.0940 - acc: 0.5903 - val_loss: 0.8213 - val_acc: 0.7360
    Epoch 77/200
    7500/7500 [==============================] - 0s 30us/step - loss: 1.0864 - acc: 0.5919 - val_loss: 0.8140 - val_acc: 0.7370
    Epoch 78/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0872 - acc: 0.5936 - val_loss: 0.8073 - val_acc: 0.7370
    Epoch 79/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.0767 - acc: 0.5975 - val_loss: 0.7990 - val_acc: 0.7370
    Epoch 80/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0606 - acc: 0.6045 - val_loss: 0.7912 - val_acc: 0.7390
    Epoch 81/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0702 - acc: 0.6021 - val_loss: 0.7851 - val_acc: 0.7400
    Epoch 82/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0764 - acc: 0.5968 - val_loss: 0.7811 - val_acc: 0.7430
    Epoch 83/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0551 - acc: 0.6103 - val_loss: 0.7777 - val_acc: 0.7410
    Epoch 84/200
    7500/7500 [==============================] - 0s 29us/step - loss: 1.0540 - acc: 0.6044 - val_loss: 0.7700 - val_acc: 0.7450
    Epoch 85/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0394 - acc: 0.6139 - val_loss: 0.7644 - val_acc: 0.7470
    Epoch 86/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.0397 - acc: 0.6067 - val_loss: 0.7589 - val_acc: 0.7470
    Epoch 87/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0272 - acc: 0.6219 - val_loss: 0.7505 - val_acc: 0.7510
    Epoch 88/200
    7500/7500 [==============================] - 0s 27us/step - loss: 1.0158 - acc: 0.6245 - val_loss: 0.7446 - val_acc: 0.7460
    Epoch 89/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.0275 - acc: 0.6136 - val_loss: 0.7412 - val_acc: 0.7500
    Epoch 90/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.0228 - acc: 0.6173 - val_loss: 0.7377 - val_acc: 0.7490
    Epoch 91/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.0199 - acc: 0.6184 - val_loss: 0.7353 - val_acc: 0.7500
    Epoch 92/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.0134 - acc: 0.6215 - val_loss: 0.7303 - val_acc: 0.7530
    Epoch 93/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.0174 - acc: 0.6208 - val_loss: 0.7271 - val_acc: 0.7500
    Epoch 94/200
    7500/7500 [==============================] - 0s 26us/step - loss: 1.0023 - acc: 0.6265 - val_loss: 0.7241 - val_acc: 0.7500
    Epoch 95/200
    7500/7500 [==============================] - 0s 25us/step - loss: 1.0029 - acc: 0.6220 - val_loss: 0.7190 - val_acc: 0.7540
    Epoch 96/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9879 - acc: 0.6327 - val_loss: 0.7127 - val_acc: 0.7560
    Epoch 97/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9865 - acc: 0.6252 - val_loss: 0.7100 - val_acc: 0.7550
    Epoch 98/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9628 - acc: 0.6372 - val_loss: 0.7027 - val_acc: 0.7590
    Epoch 99/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9743 - acc: 0.6404 - val_loss: 0.7010 - val_acc: 0.7580
    Epoch 100/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.9698 - acc: 0.6383 - val_loss: 0.6961 - val_acc: 0.7580
    Epoch 101/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9758 - acc: 0.6328 - val_loss: 0.6926 - val_acc: 0.7590
    Epoch 102/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9662 - acc: 0.6391 - val_loss: 0.6881 - val_acc: 0.7580
    Epoch 103/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9672 - acc: 0.6409 - val_loss: 0.6854 - val_acc: 0.7550
    Epoch 104/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9480 - acc: 0.6501 - val_loss: 0.6818 - val_acc: 0.7640
    Epoch 105/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9476 - acc: 0.6475 - val_loss: 0.6776 - val_acc: 0.7610
    Epoch 106/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9526 - acc: 0.6485 - val_loss: 0.6744 - val_acc: 0.7600
    Epoch 107/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9478 - acc: 0.6456 - val_loss: 0.6734 - val_acc: 0.7620
    Epoch 108/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9373 - acc: 0.6532 - val_loss: 0.6690 - val_acc: 0.7600
    Epoch 109/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9404 - acc: 0.6469 - val_loss: 0.6653 - val_acc: 0.7620
    Epoch 110/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.9366 - acc: 0.6524 - val_loss: 0.6629 - val_acc: 0.7620
    Epoch 111/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.9141 - acc: 0.6569 - val_loss: 0.6577 - val_acc: 0.7650
    Epoch 112/200
    7500/7500 [==============================] - 0s 32us/step - loss: 0.9396 - acc: 0.6484 - val_loss: 0.6591 - val_acc: 0.7620
    Epoch 113/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9192 - acc: 0.6569 - val_loss: 0.6553 - val_acc: 0.7630
    Epoch 114/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.9250 - acc: 0.6605 - val_loss: 0.6542 - val_acc: 0.7620
    Epoch 115/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.9224 - acc: 0.6533 - val_loss: 0.6496 - val_acc: 0.7640
    Epoch 116/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9200 - acc: 0.6568 - val_loss: 0.6510 - val_acc: 0.7640
    Epoch 117/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9003 - acc: 0.6680 - val_loss: 0.6470 - val_acc: 0.7610
    Epoch 118/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.9010 - acc: 0.6671 - val_loss: 0.6434 - val_acc: 0.7660
    Epoch 119/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.9034 - acc: 0.6628 - val_loss: 0.6428 - val_acc: 0.7640
    Epoch 120/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.9110 - acc: 0.6584 - val_loss: 0.6407 - val_acc: 0.7640
    Epoch 121/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8934 - acc: 0.6695 - val_loss: 0.6380 - val_acc: 0.7670
    Epoch 122/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.8921 - acc: 0.6660 - val_loss: 0.6374 - val_acc: 0.7640
    Epoch 123/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8894 - acc: 0.6664 - val_loss: 0.6322 - val_acc: 0.7660
    Epoch 124/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8856 - acc: 0.6701 - val_loss: 0.6285 - val_acc: 0.7670
    Epoch 125/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.8889 - acc: 0.6688 - val_loss: 0.6283 - val_acc: 0.7700
    Epoch 126/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8769 - acc: 0.6671 - val_loss: 0.6276 - val_acc: 0.7680
    Epoch 127/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.8794 - acc: 0.6720 - val_loss: 0.6248 - val_acc: 0.7670
    Epoch 128/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8593 - acc: 0.6759 - val_loss: 0.6234 - val_acc: 0.7690
    Epoch 129/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8801 - acc: 0.6691 - val_loss: 0.6223 - val_acc: 0.7690
    Epoch 130/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8658 - acc: 0.6731 - val_loss: 0.6190 - val_acc: 0.7720
    Epoch 131/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8869 - acc: 0.6708 - val_loss: 0.6189 - val_acc: 0.7750
    Epoch 132/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8563 - acc: 0.6824 - val_loss: 0.6160 - val_acc: 0.7750
    Epoch 133/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8640 - acc: 0.6839 - val_loss: 0.6151 - val_acc: 0.7740
    Epoch 134/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.8484 - acc: 0.6855 - val_loss: 0.6113 - val_acc: 0.7710
    Epoch 135/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8609 - acc: 0.6819 - val_loss: 0.6112 - val_acc: 0.7730
    Epoch 136/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.8558 - acc: 0.6849 - val_loss: 0.6074 - val_acc: 0.7730
    Epoch 137/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8475 - acc: 0.6873 - val_loss: 0.6076 - val_acc: 0.7700
    Epoch 138/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.8470 - acc: 0.6843 - val_loss: 0.6069 - val_acc: 0.7720
    Epoch 139/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.8506 - acc: 0.6827 - val_loss: 0.6039 - val_acc: 0.7750
    Epoch 140/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.8442 - acc: 0.6788 - val_loss: 0.6027 - val_acc: 0.7720
    Epoch 141/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8418 - acc: 0.6872 - val_loss: 0.6037 - val_acc: 0.7760
    Epoch 142/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8406 - acc: 0.6839 - val_loss: 0.6015 - val_acc: 0.7760
    Epoch 143/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8358 - acc: 0.6884 - val_loss: 0.5979 - val_acc: 0.7790
    Epoch 144/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8416 - acc: 0.6879 - val_loss: 0.5957 - val_acc: 0.7770
    Epoch 145/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8329 - acc: 0.6801 - val_loss: 0.5942 - val_acc: 0.7750
    Epoch 146/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8179 - acc: 0.6956 - val_loss: 0.5947 - val_acc: 0.7760
    Epoch 147/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8390 - acc: 0.6831 - val_loss: 0.5946 - val_acc: 0.7770
    Epoch 148/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8243 - acc: 0.6896 - val_loss: 0.5922 - val_acc: 0.7760
    Epoch 149/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.8181 - acc: 0.6913 - val_loss: 0.5918 - val_acc: 0.7810
    Epoch 150/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8336 - acc: 0.6841 - val_loss: 0.5914 - val_acc: 0.7810
    Epoch 151/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8093 - acc: 0.6941 - val_loss: 0.5911 - val_acc: 0.7800
    Epoch 152/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8186 - acc: 0.6928 - val_loss: 0.5864 - val_acc: 0.7850
    Epoch 153/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8137 - acc: 0.7011 - val_loss: 0.5850 - val_acc: 0.7760
    Epoch 154/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8236 - acc: 0.6940 - val_loss: 0.5852 - val_acc: 0.7820
    Epoch 155/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8231 - acc: 0.6897 - val_loss: 0.5852 - val_acc: 0.7830
    Epoch 156/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8060 - acc: 0.7013 - val_loss: 0.5845 - val_acc: 0.7790
    Epoch 157/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7927 - acc: 0.7033 - val_loss: 0.5826 - val_acc: 0.7810
    Epoch 158/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7906 - acc: 0.7007 - val_loss: 0.5800 - val_acc: 0.7840
    Epoch 159/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.8047 - acc: 0.6956 - val_loss: 0.5789 - val_acc: 0.7820
    Epoch 160/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.8041 - acc: 0.7039 - val_loss: 0.5785 - val_acc: 0.7840
    Epoch 161/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7914 - acc: 0.7107 - val_loss: 0.5778 - val_acc: 0.7770
    Epoch 162/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7874 - acc: 0.7096 - val_loss: 0.5766 - val_acc: 0.7780
    Epoch 163/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7860 - acc: 0.7115 - val_loss: 0.5753 - val_acc: 0.7880
    Epoch 164/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7962 - acc: 0.7027 - val_loss: 0.5772 - val_acc: 0.7800
    Epoch 165/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7840 - acc: 0.7115 - val_loss: 0.5736 - val_acc: 0.7850
    Epoch 166/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.7821 - acc: 0.7073 - val_loss: 0.5730 - val_acc: 0.7820
    Epoch 167/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7884 - acc: 0.7077 - val_loss: 0.5700 - val_acc: 0.7840
    Epoch 168/200
    7500/7500 [==============================] - 0s 30us/step - loss: 0.7944 - acc: 0.7095 - val_loss: 0.5711 - val_acc: 0.7820
    Epoch 169/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7728 - acc: 0.7117 - val_loss: 0.5682 - val_acc: 0.7810
    Epoch 170/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.7821 - acc: 0.7048 - val_loss: 0.5666 - val_acc: 0.7840
    Epoch 171/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7830 - acc: 0.7076 - val_loss: 0.5673 - val_acc: 0.7840
    Epoch 172/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.7835 - acc: 0.7121 - val_loss: 0.5684 - val_acc: 0.7830
    Epoch 173/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7745 - acc: 0.7136 - val_loss: 0.5668 - val_acc: 0.7850
    Epoch 174/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7770 - acc: 0.7079 - val_loss: 0.5686 - val_acc: 0.7810
    Epoch 175/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.7860 - acc: 0.7053 - val_loss: 0.5658 - val_acc: 0.7860
    Epoch 176/200
    7500/7500 [==============================] - 0s 31us/step - loss: 0.7738 - acc: 0.7079 - val_loss: 0.5644 - val_acc: 0.7880
    Epoch 177/200
    7500/7500 [==============================] - 0s 31us/step - loss: 0.7666 - acc: 0.7068 - val_loss: 0.5638 - val_acc: 0.7820
    Epoch 178/200
    7500/7500 [==============================] - 0s 30us/step - loss: 0.7707 - acc: 0.7079 - val_loss: 0.5627 - val_acc: 0.7840
    Epoch 179/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7840 - acc: 0.7064 - val_loss: 0.5618 - val_acc: 0.7870
    Epoch 180/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7650 - acc: 0.7151 - val_loss: 0.5598 - val_acc: 0.7850
    Epoch 181/200
    7500/7500 [==============================] - 0s 29us/step - loss: 0.7661 - acc: 0.7088 - val_loss: 0.5599 - val_acc: 0.7860
    Epoch 182/200
    7500/7500 [==============================] - 0s 30us/step - loss: 0.7444 - acc: 0.7239 - val_loss: 0.5568 - val_acc: 0.7870
    Epoch 183/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7523 - acc: 0.7149 - val_loss: 0.5578 - val_acc: 0.7830
    Epoch 184/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7616 - acc: 0.7185 - val_loss: 0.5593 - val_acc: 0.7850
    Epoch 185/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7566 - acc: 0.7108 - val_loss: 0.5588 - val_acc: 0.7860
    Epoch 186/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7467 - acc: 0.7197 - val_loss: 0.5590 - val_acc: 0.7830
    Epoch 187/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7537 - acc: 0.7157 - val_loss: 0.5566 - val_acc: 0.7800
    Epoch 188/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7494 - acc: 0.7199 - val_loss: 0.5566 - val_acc: 0.7830
    Epoch 189/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7526 - acc: 0.7203 - val_loss: 0.5554 - val_acc: 0.7880
    Epoch 190/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7362 - acc: 0.7256 - val_loss: 0.5528 - val_acc: 0.7860
    Epoch 191/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7464 - acc: 0.7181 - val_loss: 0.5523 - val_acc: 0.7850
    Epoch 192/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7515 - acc: 0.7141 - val_loss: 0.5511 - val_acc: 0.7850
    Epoch 193/200
    7500/7500 [==============================] - 0s 30us/step - loss: 0.7432 - acc: 0.7191 - val_loss: 0.5511 - val_acc: 0.7880
    Epoch 194/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7388 - acc: 0.7232 - val_loss: 0.5523 - val_acc: 0.7850
    Epoch 195/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.7412 - acc: 0.7212 - val_loss: 0.5543 - val_acc: 0.7840
    Epoch 196/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7288 - acc: 0.7279 - val_loss: 0.5510 - val_acc: 0.7830
    Epoch 197/200
    7500/7500 [==============================] - 0s 28us/step - loss: 0.7335 - acc: 0.7285 - val_loss: 0.5509 - val_acc: 0.7820
    Epoch 198/200
    7500/7500 [==============================] - 0s 25us/step - loss: 0.7255 - acc: 0.7292 - val_loss: 0.5480 - val_acc: 0.7840
    Epoch 199/200
    7500/7500 [==============================] - 0s 27us/step - loss: 0.7341 - acc: 0.7236 - val_loss: 0.5490 - val_acc: 0.7830
    Epoch 200/200
    7500/7500 [==============================] - 0s 26us/step - loss: 0.7398 - acc: 0.7244 - val_loss: 0.5489 - val_acc: 0.7840



```python
results_train = model.evaluate(train_final, label_train_final)
results_test = model.evaluate(X_test, y_test)
```

    7500/7500 [==============================] - 0s 24us/step
    1500/1500 [==============================] - 0s 26us/step



```python
results_train
```




    [0.44953240927060445, 0.8355999999682109]




```python
results_test
```




    [0.6567809325853984, 0.745333333492279]



You can see here that the validation performance has improved again! the variance did become higher again compared to L1-regularization.

## Bigger Data?

In the lecture, one of the solutions to high variance was just getting more data. You actually *have* more data, but took a subset of 10,000 units before. Let's now quadruple your data set, and see what happens. Note that you are really just lucky here, and getting more data isn't always possible, but this is a useful exercise in order to understand the power of big data sets.


```python
df = pd.read_csv('Bank_complaints.csv')
random.seed(123)
df = df.sample(40000)
df.index = range(40000)
product = df["Product"]
complaints = df["Consumer complaint narrative"]

#one-hot encoding of the complaints
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(complaints)
sequences = tokenizer.texts_to_sequences(complaints)
one_hot_results= tokenizer.texts_to_matrix(complaints, mode='binary')
word_index = tokenizer.word_index
np.shape(one_hot_results)

#one-hot encoding of products
le = preprocessing.LabelEncoder()
le.fit(product)
list(le.classes_)
product_cat = le.transform(product) 
product_onehot = to_categorical(product_cat)

# train test split
test_index = random.sample(range(1,40000), 4000)
test = one_hot_results[test_index]
train = np.delete(one_hot_results, test_index, 0)
label_test = product_onehot[test_index]
label_train = np.delete(product_onehot, test_index, 0)

#Validation set
random.seed(123)
val = train[:3000]
train_final = train[3000:]
label_val = label_train[:3000]
label_train_final = label_train[3000:]
```


```python
# ⏰ This cell may take several minutes to run
random.seed(123)
model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(2000,))) #2 hidden layers
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

moredata_model = model.fit(train_final,
                    label_train_final,
                    epochs=120,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 33000 samples, validate on 3000 samples
    Epoch 1/120
    33000/33000 [==============================] - 1s 25us/step - loss: 1.9131 - acc: 0.1977 - val_loss: 1.8734 - val_acc: 0.2517
    Epoch 2/120
    33000/33000 [==============================] - 1s 16us/step - loss: 1.8204 - acc: 0.3034 - val_loss: 1.7551 - val_acc: 0.3397
    Epoch 3/120
    33000/33000 [==============================] - 1s 16us/step - loss: 1.6686 - acc: 0.4072 - val_loss: 1.5741 - val_acc: 0.4647
    Epoch 4/120
    33000/33000 [==============================] - 1s 15us/step - loss: 1.4662 - acc: 0.5248 - val_loss: 1.3619 - val_acc: 0.5560
    Epoch 5/120
    33000/33000 [==============================] - 1s 15us/step - loss: 1.2557 - acc: 0.6060 - val_loss: 1.1666 - val_acc: 0.6303
    Epoch 6/120
    33000/33000 [==============================] - 0s 15us/step - loss: 1.0768 - acc: 0.6660 - val_loss: 1.0120 - val_acc: 0.6777
    Epoch 7/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.9451 - acc: 0.7012 - val_loss: 0.9037 - val_acc: 0.7047
    Epoch 8/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.8536 - acc: 0.7191 - val_loss: 0.8281 - val_acc: 0.7210
    Epoch 9/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.7894 - acc: 0.7321 - val_loss: 0.7750 - val_acc: 0.7300
    Epoch 10/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.7434 - acc: 0.7419 - val_loss: 0.7363 - val_acc: 0.7397
    Epoch 11/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.7086 - acc: 0.7492 - val_loss: 0.7075 - val_acc: 0.7450
    Epoch 12/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.6815 - acc: 0.7555 - val_loss: 0.6846 - val_acc: 0.7507
    Epoch 13/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.6596 - acc: 0.7612 - val_loss: 0.6661 - val_acc: 0.7587
    Epoch 14/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.6413 - acc: 0.7678 - val_loss: 0.6501 - val_acc: 0.7623
    Epoch 15/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.6259 - acc: 0.7724 - val_loss: 0.6383 - val_acc: 0.7653
    Epoch 16/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.6119 - acc: 0.7780 - val_loss: 0.6275 - val_acc: 0.7700
    Epoch 17/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.5999 - acc: 0.7809 - val_loss: 0.6168 - val_acc: 0.7720
    Epoch 18/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.5891 - acc: 0.7845 - val_loss: 0.6079 - val_acc: 0.7757
    Epoch 19/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.5792 - acc: 0.7892 - val_loss: 0.6013 - val_acc: 0.7753
    Epoch 20/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.5700 - acc: 0.7920 - val_loss: 0.5924 - val_acc: 0.7813
    Epoch 21/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.5614 - acc: 0.7954 - val_loss: 0.5879 - val_acc: 0.7807
    Epoch 22/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5535 - acc: 0.7989 - val_loss: 0.5832 - val_acc: 0.7817
    Epoch 23/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5459 - acc: 0.8010 - val_loss: 0.5766 - val_acc: 0.7847
    Epoch 24/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5391 - acc: 0.8044 - val_loss: 0.5735 - val_acc: 0.7850
    Epoch 25/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5326 - acc: 0.8076 - val_loss: 0.5674 - val_acc: 0.7937
    Epoch 26/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.5264 - acc: 0.8092 - val_loss: 0.5622 - val_acc: 0.7920
    Epoch 27/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5198 - acc: 0.8114 - val_loss: 0.5599 - val_acc: 0.7977
    Epoch 28/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5144 - acc: 0.8140 - val_loss: 0.5571 - val_acc: 0.8000
    Epoch 29/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5087 - acc: 0.8162 - val_loss: 0.5509 - val_acc: 0.8000
    Epoch 30/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.5033 - acc: 0.8180 - val_loss: 0.5483 - val_acc: 0.8020
    Epoch 31/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4982 - acc: 0.8205 - val_loss: 0.5443 - val_acc: 0.8023
    Epoch 32/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4934 - acc: 0.8222 - val_loss: 0.5435 - val_acc: 0.8027
    Epoch 33/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.4885 - acc: 0.8253 - val_loss: 0.5426 - val_acc: 0.8033
    Epoch 34/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.4840 - acc: 0.8256 - val_loss: 0.5386 - val_acc: 0.8080
    Epoch 35/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4799 - acc: 0.8278 - val_loss: 0.5341 - val_acc: 0.8093
    Epoch 36/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4755 - acc: 0.8305 - val_loss: 0.5322 - val_acc: 0.8100
    Epoch 37/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.4713 - acc: 0.8308 - val_loss: 0.5297 - val_acc: 0.8117
    Epoch 38/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4674 - acc: 0.8319 - val_loss: 0.5273 - val_acc: 0.8123
    Epoch 39/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4632 - acc: 0.8339 - val_loss: 0.5265 - val_acc: 0.8103
    Epoch 40/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4599 - acc: 0.8355 - val_loss: 0.5236 - val_acc: 0.8103
    Epoch 41/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4558 - acc: 0.8370 - val_loss: 0.5241 - val_acc: 0.8103
    Epoch 42/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4522 - acc: 0.8383 - val_loss: 0.5210 - val_acc: 0.8120
    Epoch 43/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4487 - acc: 0.8403 - val_loss: 0.5223 - val_acc: 0.8143
    Epoch 44/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.4453 - acc: 0.8405 - val_loss: 0.5187 - val_acc: 0.8180
    Epoch 45/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4420 - acc: 0.8427 - val_loss: 0.5206 - val_acc: 0.8153
    Epoch 46/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4388 - acc: 0.8437 - val_loss: 0.5186 - val_acc: 0.8120
    Epoch 47/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4358 - acc: 0.8442 - val_loss: 0.5154 - val_acc: 0.8133
    Epoch 48/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4324 - acc: 0.8458 - val_loss: 0.5156 - val_acc: 0.8130
    Epoch 49/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4296 - acc: 0.8474 - val_loss: 0.5147 - val_acc: 0.8150
    Epoch 50/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4266 - acc: 0.8484 - val_loss: 0.5136 - val_acc: 0.8117
    Epoch 51/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4235 - acc: 0.8492 - val_loss: 0.5142 - val_acc: 0.8167
    Epoch 52/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.4206 - acc: 0.8502 - val_loss: 0.5135 - val_acc: 0.8133
    Epoch 53/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4181 - acc: 0.8508 - val_loss: 0.5154 - val_acc: 0.8163
    Epoch 54/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4152 - acc: 0.8521 - val_loss: 0.5109 - val_acc: 0.8140
    Epoch 55/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4125 - acc: 0.8531 - val_loss: 0.5124 - val_acc: 0.8160
    Epoch 56/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4100 - acc: 0.8537 - val_loss: 0.5126 - val_acc: 0.8163
    Epoch 57/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4076 - acc: 0.8543 - val_loss: 0.5112 - val_acc: 0.8180
    Epoch 58/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.4052 - acc: 0.8555 - val_loss: 0.5120 - val_acc: 0.8113
    Epoch 59/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4028 - acc: 0.8556 - val_loss: 0.5111 - val_acc: 0.8130
    Epoch 60/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.4000 - acc: 0.8580 - val_loss: 0.5105 - val_acc: 0.8183
    Epoch 61/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3977 - acc: 0.8590 - val_loss: 0.5116 - val_acc: 0.8163
    Epoch 62/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3953 - acc: 0.8602 - val_loss: 0.5123 - val_acc: 0.8180
    Epoch 63/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3931 - acc: 0.8606 - val_loss: 0.5089 - val_acc: 0.8157
    Epoch 64/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3908 - acc: 0.8617 - val_loss: 0.5120 - val_acc: 0.8113
    Epoch 65/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3888 - acc: 0.8624 - val_loss: 0.5128 - val_acc: 0.8157
    Epoch 66/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3863 - acc: 0.8633 - val_loss: 0.5122 - val_acc: 0.8167
    Epoch 67/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3844 - acc: 0.8638 - val_loss: 0.5100 - val_acc: 0.8157
    Epoch 68/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3821 - acc: 0.8646 - val_loss: 0.5113 - val_acc: 0.8160
    Epoch 69/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3801 - acc: 0.8665 - val_loss: 0.5136 - val_acc: 0.8120
    Epoch 70/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3779 - acc: 0.8661 - val_loss: 0.5121 - val_acc: 0.8163
    Epoch 71/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3760 - acc: 0.8688 - val_loss: 0.5113 - val_acc: 0.8117
    Epoch 72/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3741 - acc: 0.8685 - val_loss: 0.5115 - val_acc: 0.8163
    Epoch 73/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3720 - acc: 0.8689 - val_loss: 0.5121 - val_acc: 0.8167
    Epoch 74/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3702 - acc: 0.8699 - val_loss: 0.5157 - val_acc: 0.8160
    Epoch 75/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3681 - acc: 0.8702 - val_loss: 0.5137 - val_acc: 0.8160
    Epoch 76/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3661 - acc: 0.8725 - val_loss: 0.5126 - val_acc: 0.8143
    Epoch 77/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3641 - acc: 0.8718 - val_loss: 0.5138 - val_acc: 0.8147
    Epoch 78/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3627 - acc: 0.8725 - val_loss: 0.5194 - val_acc: 0.8160
    Epoch 79/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3610 - acc: 0.8739 - val_loss: 0.5152 - val_acc: 0.8117
    Epoch 80/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3589 - acc: 0.8742 - val_loss: 0.5166 - val_acc: 0.8170
    Epoch 81/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3571 - acc: 0.8754 - val_loss: 0.5157 - val_acc: 0.8147
    Epoch 82/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3557 - acc: 0.8765 - val_loss: 0.5159 - val_acc: 0.8150
    Epoch 83/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3536 - acc: 0.8771 - val_loss: 0.5180 - val_acc: 0.8157
    Epoch 84/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3520 - acc: 0.8768 - val_loss: 0.5189 - val_acc: 0.8140
    Epoch 85/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3501 - acc: 0.8779 - val_loss: 0.5177 - val_acc: 0.8160
    Epoch 86/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3487 - acc: 0.8785 - val_loss: 0.5218 - val_acc: 0.8167
    Epoch 87/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3468 - acc: 0.8794 - val_loss: 0.5212 - val_acc: 0.8137
    Epoch 88/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.3456 - acc: 0.8793 - val_loss: 0.5198 - val_acc: 0.8153
    Epoch 89/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.3438 - acc: 0.8801 - val_loss: 0.5210 - val_acc: 0.8143
    Epoch 90/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3421 - acc: 0.8802 - val_loss: 0.5235 - val_acc: 0.8127
    Epoch 91/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3406 - acc: 0.8813 - val_loss: 0.5213 - val_acc: 0.8143
    Epoch 92/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3386 - acc: 0.8818 - val_loss: 0.5223 - val_acc: 0.8153
    Epoch 93/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.3374 - acc: 0.8827 - val_loss: 0.5232 - val_acc: 0.8137
    Epoch 94/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3358 - acc: 0.8840 - val_loss: 0.5240 - val_acc: 0.8150
    Epoch 95/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3342 - acc: 0.8832 - val_loss: 0.5284 - val_acc: 0.8160
    Epoch 96/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3328 - acc: 0.8852 - val_loss: 0.5263 - val_acc: 0.8160
    Epoch 97/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3312 - acc: 0.8856 - val_loss: 0.5260 - val_acc: 0.8137
    Epoch 98/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3297 - acc: 0.8869 - val_loss: 0.5322 - val_acc: 0.8117
    Epoch 99/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3280 - acc: 0.8863 - val_loss: 0.5297 - val_acc: 0.8140
    Epoch 100/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3267 - acc: 0.8873 - val_loss: 0.5302 - val_acc: 0.8127
    Epoch 101/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3257 - acc: 0.8878 - val_loss: 0.5295 - val_acc: 0.8133
    Epoch 102/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3239 - acc: 0.8889 - val_loss: 0.5335 - val_acc: 0.8143
    Epoch 103/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3223 - acc: 0.8888 - val_loss: 0.5320 - val_acc: 0.8153
    Epoch 104/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.3212 - acc: 0.8890 - val_loss: 0.5335 - val_acc: 0.8130
    Epoch 105/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3196 - acc: 0.8891 - val_loss: 0.5339 - val_acc: 0.8150
    Epoch 106/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3184 - acc: 0.8903 - val_loss: 0.5370 - val_acc: 0.8143
    Epoch 107/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3171 - acc: 0.8912 - val_loss: 0.5352 - val_acc: 0.8147
    Epoch 108/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3152 - acc: 0.8909 - val_loss: 0.5379 - val_acc: 0.8127
    Epoch 109/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3144 - acc: 0.8923 - val_loss: 0.5363 - val_acc: 0.8137
    Epoch 110/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3130 - acc: 0.8931 - val_loss: 0.5379 - val_acc: 0.8133
    Epoch 111/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3114 - acc: 0.8927 - val_loss: 0.5388 - val_acc: 0.8153
    Epoch 112/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3100 - acc: 0.8924 - val_loss: 0.5392 - val_acc: 0.8147
    Epoch 113/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.3091 - acc: 0.8947 - val_loss: 0.5406 - val_acc: 0.8137
    Epoch 114/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3078 - acc: 0.8943 - val_loss: 0.5422 - val_acc: 0.8157
    Epoch 115/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3061 - acc: 0.8951 - val_loss: 0.5433 - val_acc: 0.8123
    Epoch 116/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3052 - acc: 0.8956 - val_loss: 0.5432 - val_acc: 0.8130
    Epoch 117/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3035 - acc: 0.8950 - val_loss: 0.5483 - val_acc: 0.8090
    Epoch 118/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.3024 - acc: 0.8965 - val_loss: 0.5461 - val_acc: 0.8117
    Epoch 119/120
    33000/33000 [==============================] - 1s 16us/step - loss: 0.3013 - acc: 0.8966 - val_loss: 0.5459 - val_acc: 0.8127
    Epoch 120/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.2998 - acc: 0.8972 - val_loss: 0.5460 - val_acc: 0.8150



```python
results_train = model.evaluate(train_final, label_train_final)
results_test = model.evaluate(test, label_test)
```

    33000/33000 [==============================] - 1s 21us/step
    4000/4000 [==============================] - 0s 22us/step



```python
results_train
```




    [0.29492314792401864, 0.8997272727272727]




```python
results_test
```




    [0.5750258494615554, 0.805]



With the same amount of epochs, you were able to get a fairly similar validation accuracy of 89.67 (compared to 88.55 in obtained in the first model in this lab). your test set accuracy went up from 75.8 to a staggering 80.225% though, without any other regularization technique. You can still consider early stopping, L1, L2 and dropout here. It's clear that having more data has a strong impact on model performance!

## Additional Resources

* https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb
* https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
* https://catalog.data.gov/dataset/consumer-complaint-database

## Summary  

In this lesson, you not only built an initial deep-learning model, you then used a validation set to tune your model using various types of regularization.
