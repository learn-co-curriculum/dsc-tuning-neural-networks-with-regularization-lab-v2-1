
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
```


```python
# __SOLUTION__ 
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

    Using TensorFlow backend.


## Load the Data

As with the previous lab, the data is stored in a file **Bank_complaints.csv**. Load and preview the dataset.


```python
#Your code here; load and preview the dataset
```


```python
# __SOLUTION__ 
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
```


```python
# __SOLUTION__ 
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
```


```python
# __SOLUTION__ 
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
#Then transform these integer values into a matrix of binary flags
```


```python
# __SOLUTION__ 
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
#Yyour code here
X_train = 
X_test = 
y_train = 
y_test = 
```


```python
# __SOLUTION__ 
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
#Just run this block of code 
random.seed(123)
val = X_train[:1000]
train_final = X_train[1000:]
label_val = y_train[:1000]
label_train_final = y_train[1000:]
```


```python
# __SOLUTION__ 
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
model = 
```


```python
# __SOLUTION__ 
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
```


```python
# __SOLUTION__ 
#Your code here
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Training the Model

Ok, now for the resource intensive part: time to train your model! Note that this is where you also introduce the validation data to the model.


```python
#Code provided; note the extra validation parameter passed.
model_val = model.fit(train_final,
                    label_train_final,
                    epochs=120,
                    batch_size=256,
                    validation_data=(val, label_val))
```


```python
# __SOLUTION__ 
model_val = model.fit(train_final,
                    label_train_final,
                    epochs=120,
                    batch_size=256,
                    validation_data=(val, label_val))
```

    Train on 7500 samples, validate on 1000 samples
    Epoch 1/120
    7500/7500 [==============================] - 0s 27us/step - loss: 1.9514 - acc: 0.1745 - val_loss: 1.9359 - val_acc: 0.1830
    Epoch 2/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.9169 - acc: 0.2111 - val_loss: 1.9126 - val_acc: 0.2050
    Epoch 3/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.8930 - acc: 0.2324 - val_loss: 1.8926 - val_acc: 0.2180
    Epoch 4/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8708 - acc: 0.2373 - val_loss: 1.8714 - val_acc: 0.2310
    Epoch 5/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8476 - acc: 0.2524 - val_loss: 1.8479 - val_acc: 0.2490
    Epoch 6/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8218 - acc: 0.2756 - val_loss: 1.8210 - val_acc: 0.2620
    Epoch 7/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7923 - acc: 0.2989 - val_loss: 1.7903 - val_acc: 0.2910
    Epoch 8/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.7596 - acc: 0.3248 - val_loss: 1.7559 - val_acc: 0.3230
    Epoch 9/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.7238 - acc: 0.3541 - val_loss: 1.7183 - val_acc: 0.3620
    Epoch 10/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.6848 - acc: 0.3815 - val_loss: 1.6776 - val_acc: 0.3870
    Epoch 11/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.6430 - acc: 0.4085 - val_loss: 1.6340 - val_acc: 0.4240
    Epoch 12/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.5994 - acc: 0.4385 - val_loss: 1.5889 - val_acc: 0.4580
    Epoch 13/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5538 - acc: 0.4657 - val_loss: 1.5431 - val_acc: 0.4690
    Epoch 14/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5073 - acc: 0.4891 - val_loss: 1.4943 - val_acc: 0.5080
    Epoch 15/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.4601 - acc: 0.5173 - val_loss: 1.4468 - val_acc: 0.5250
    Epoch 16/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.4131 - acc: 0.5435 - val_loss: 1.3994 - val_acc: 0.5480
    Epoch 17/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.3661 - acc: 0.5649 - val_loss: 1.3508 - val_acc: 0.5810
    Epoch 18/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.3202 - acc: 0.5924 - val_loss: 1.3052 - val_acc: 0.6070
    Epoch 19/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.2752 - acc: 0.6124 - val_loss: 1.2608 - val_acc: 0.6250
    Epoch 20/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2312 - acc: 0.6311 - val_loss: 1.2171 - val_acc: 0.6380
    Epoch 21/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1889 - acc: 0.6415 - val_loss: 1.1770 - val_acc: 0.6490
    Epoch 22/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1487 - acc: 0.6516 - val_loss: 1.1366 - val_acc: 0.6560
    Epoch 23/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1101 - acc: 0.6641 - val_loss: 1.0992 - val_acc: 0.6710
    Epoch 24/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.0739 - acc: 0.6700 - val_loss: 1.0648 - val_acc: 0.6780
    Epoch 25/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0395 - acc: 0.6804 - val_loss: 1.0302 - val_acc: 0.6760
    Epoch 26/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0079 - acc: 0.6857 - val_loss: 1.0025 - val_acc: 0.6850
    Epoch 27/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.9780 - acc: 0.6916 - val_loss: 0.9728 - val_acc: 0.6970
    Epoch 28/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9500 - acc: 0.6979 - val_loss: 0.9496 - val_acc: 0.6960
    Epoch 29/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9246 - acc: 0.7008 - val_loss: 0.9242 - val_acc: 0.7020
    Epoch 30/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9006 - acc: 0.7061 - val_loss: 0.9030 - val_acc: 0.7040
    Epoch 31/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8786 - acc: 0.7119 - val_loss: 0.8859 - val_acc: 0.7030
    Epoch 32/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8584 - acc: 0.7183 - val_loss: 0.8655 - val_acc: 0.7100
    Epoch 33/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8393 - acc: 0.7253 - val_loss: 0.8500 - val_acc: 0.7100
    Epoch 34/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.8217 - acc: 0.7251 - val_loss: 0.8324 - val_acc: 0.7090
    Epoch 35/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8058 - acc: 0.7284 - val_loss: 0.8197 - val_acc: 0.7180
    Epoch 36/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.7898 - acc: 0.7341 - val_loss: 0.8065 - val_acc: 0.7210
    Epoch 37/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.7755 - acc: 0.7388 - val_loss: 0.7952 - val_acc: 0.7160
    Epoch 38/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.7624 - acc: 0.7396 - val_loss: 0.7843 - val_acc: 0.7250
    Epoch 39/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.7497 - acc: 0.7428 - val_loss: 0.7726 - val_acc: 0.7210
    Epoch 40/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.7375 - acc: 0.7452 - val_loss: 0.7640 - val_acc: 0.7220
    Epoch 41/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.7265 - acc: 0.7493 - val_loss: 0.7567 - val_acc: 0.7290
    Epoch 42/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.7159 - acc: 0.7511 - val_loss: 0.7469 - val_acc: 0.7340
    Epoch 43/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.7058 - acc: 0.7555 - val_loss: 0.7439 - val_acc: 0.7290
    Epoch 44/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6959 - acc: 0.7556 - val_loss: 0.7340 - val_acc: 0.7370
    Epoch 45/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.6867 - acc: 0.7604 - val_loss: 0.7274 - val_acc: 0.7360
    Epoch 46/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6777 - acc: 0.7628 - val_loss: 0.7205 - val_acc: 0.7360
    Epoch 47/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6697 - acc: 0.7640 - val_loss: 0.7165 - val_acc: 0.7320
    Epoch 48/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.6615 - acc: 0.7675 - val_loss: 0.7095 - val_acc: 0.7350
    Epoch 49/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6538 - acc: 0.7696 - val_loss: 0.7073 - val_acc: 0.7330
    Epoch 50/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.6465 - acc: 0.7720 - val_loss: 0.6989 - val_acc: 0.7400
    Epoch 51/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.6387 - acc: 0.7740 - val_loss: 0.6991 - val_acc: 0.7330
    Epoch 52/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6318 - acc: 0.7777 - val_loss: 0.6924 - val_acc: 0.7460
    Epoch 53/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6250 - acc: 0.7792 - val_loss: 0.6886 - val_acc: 0.7410
    Epoch 54/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6180 - acc: 0.7819 - val_loss: 0.6893 - val_acc: 0.7490
    Epoch 55/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6125 - acc: 0.7841 - val_loss: 0.6825 - val_acc: 0.7470
    Epoch 56/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6057 - acc: 0.7868 - val_loss: 0.6789 - val_acc: 0.7510
    Epoch 57/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5994 - acc: 0.7868 - val_loss: 0.6753 - val_acc: 0.7470
    Epoch 58/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5939 - acc: 0.7909 - val_loss: 0.6722 - val_acc: 0.7450
    Epoch 59/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5884 - acc: 0.7945 - val_loss: 0.6700 - val_acc: 0.7470
    Epoch 60/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5822 - acc: 0.7947 - val_loss: 0.6670 - val_acc: 0.7500
    Epoch 61/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5774 - acc: 0.7991 - val_loss: 0.6633 - val_acc: 0.7540
    Epoch 62/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5715 - acc: 0.7999 - val_loss: 0.6635 - val_acc: 0.7430
    Epoch 63/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5666 - acc: 0.8029 - val_loss: 0.6595 - val_acc: 0.7590
    Epoch 64/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5614 - acc: 0.8044 - val_loss: 0.6573 - val_acc: 0.7470
    Epoch 65/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5563 - acc: 0.8077 - val_loss: 0.6573 - val_acc: 0.7520
    Epoch 66/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5516 - acc: 0.8100 - val_loss: 0.6548 - val_acc: 0.7480
    Epoch 67/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5466 - acc: 0.8112 - val_loss: 0.6536 - val_acc: 0.7560
    Epoch 68/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5422 - acc: 0.8120 - val_loss: 0.6502 - val_acc: 0.7570
    Epoch 69/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5375 - acc: 0.8145 - val_loss: 0.6478 - val_acc: 0.7570
    Epoch 70/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5329 - acc: 0.8177 - val_loss: 0.6481 - val_acc: 0.7540
    Epoch 71/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5285 - acc: 0.8161 - val_loss: 0.6466 - val_acc: 0.7640
    Epoch 72/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5241 - acc: 0.8195 - val_loss: 0.6462 - val_acc: 0.7570
    Epoch 73/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5199 - acc: 0.8211 - val_loss: 0.6436 - val_acc: 0.7530
    Epoch 74/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5153 - acc: 0.8205 - val_loss: 0.6432 - val_acc: 0.7550
    Epoch 75/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5114 - acc: 0.8253 - val_loss: 0.6422 - val_acc: 0.7510
    Epoch 76/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5075 - acc: 0.8253 - val_loss: 0.6403 - val_acc: 0.7560
    Epoch 77/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.5034 - acc: 0.8279 - val_loss: 0.6394 - val_acc: 0.7670
    Epoch 78/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4995 - acc: 0.8284 - val_loss: 0.6408 - val_acc: 0.7600
    Epoch 79/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4958 - acc: 0.8303 - val_loss: 0.6377 - val_acc: 0.7550
    Epoch 80/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4920 - acc: 0.8304 - val_loss: 0.6370 - val_acc: 0.7520
    Epoch 81/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4881 - acc: 0.8327 - val_loss: 0.6366 - val_acc: 0.7640
    Epoch 82/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4839 - acc: 0.8325 - val_loss: 0.6400 - val_acc: 0.7520
    Epoch 83/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4805 - acc: 0.8359 - val_loss: 0.6393 - val_acc: 0.7530
    Epoch 84/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4764 - acc: 0.8380 - val_loss: 0.6343 - val_acc: 0.7540
    Epoch 85/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4728 - acc: 0.8387 - val_loss: 0.6361 - val_acc: 0.7570
    Epoch 86/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4694 - acc: 0.8403 - val_loss: 0.6342 - val_acc: 0.7530
    Epoch 87/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4658 - acc: 0.8431 - val_loss: 0.6344 - val_acc: 0.7530
    Epoch 88/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4626 - acc: 0.8455 - val_loss: 0.6334 - val_acc: 0.7580
    Epoch 89/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4593 - acc: 0.8444 - val_loss: 0.6332 - val_acc: 0.7500
    Epoch 90/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4553 - acc: 0.8452 - val_loss: 0.6325 - val_acc: 0.7540
    Epoch 91/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4528 - acc: 0.8485 - val_loss: 0.6327 - val_acc: 0.7540
    Epoch 92/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.4492 - acc: 0.8500 - val_loss: 0.6318 - val_acc: 0.7610
    Epoch 93/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4460 - acc: 0.8515 - val_loss: 0.6313 - val_acc: 0.7530
    Epoch 94/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4425 - acc: 0.8531 - val_loss: 0.6350 - val_acc: 0.7600
    Epoch 95/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4395 - acc: 0.8551 - val_loss: 0.6317 - val_acc: 0.7620
    Epoch 96/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4362 - acc: 0.8552 - val_loss: 0.6311 - val_acc: 0.7600
    Epoch 97/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4328 - acc: 0.8573 - val_loss: 0.6335 - val_acc: 0.7470
    Epoch 98/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4298 - acc: 0.8600 - val_loss: 0.6299 - val_acc: 0.7570
    Epoch 99/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4269 - acc: 0.8591 - val_loss: 0.6291 - val_acc: 0.7600
    Epoch 100/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4236 - acc: 0.8605 - val_loss: 0.6304 - val_acc: 0.7580
    Epoch 101/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4209 - acc: 0.8617 - val_loss: 0.6300 - val_acc: 0.7620
    Epoch 102/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4177 - acc: 0.8615 - val_loss: 0.6301 - val_acc: 0.7580
    Epoch 103/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4147 - acc: 0.8621 - val_loss: 0.6297 - val_acc: 0.7570
    Epoch 104/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4119 - acc: 0.8648 - val_loss: 0.6324 - val_acc: 0.7550
    Epoch 105/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4092 - acc: 0.8663 - val_loss: 0.6308 - val_acc: 0.7580
    Epoch 106/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4058 - acc: 0.8675 - val_loss: 0.6314 - val_acc: 0.7550
    Epoch 107/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.4036 - acc: 0.8683 - val_loss: 0.6309 - val_acc: 0.7540
    Epoch 108/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.4003 - acc: 0.8709 - val_loss: 0.6307 - val_acc: 0.7580
    Epoch 109/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3976 - acc: 0.8696 - val_loss: 0.6299 - val_acc: 0.7550
    Epoch 110/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3949 - acc: 0.8711 - val_loss: 0.6336 - val_acc: 0.7540
    Epoch 111/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3924 - acc: 0.8731 - val_loss: 0.6305 - val_acc: 0.7600
    Epoch 112/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3895 - acc: 0.8743 - val_loss: 0.6313 - val_acc: 0.7580
    Epoch 113/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3868 - acc: 0.8775 - val_loss: 0.6312 - val_acc: 0.7610
    Epoch 114/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3844 - acc: 0.8748 - val_loss: 0.6301 - val_acc: 0.7600
    Epoch 115/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3811 - acc: 0.8781 - val_loss: 0.6331 - val_acc: 0.7540
    Epoch 116/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.3789 - acc: 0.8791 - val_loss: 0.6310 - val_acc: 0.7550
    Epoch 117/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.3762 - acc: 0.8805 - val_loss: 0.6303 - val_acc: 0.7560
    Epoch 118/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.3741 - acc: 0.8808 - val_loss: 0.6322 - val_acc: 0.7560
    Epoch 119/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3714 - acc: 0.8812 - val_loss: 0.6334 - val_acc: 0.7570
    Epoch 120/120
    7500/7500 [==============================] - 0s 12us/step - loss: 0.3689 - acc: 0.8832 - val_loss: 0.6336 - val_acc: 0.7550


## Retrieving Performance Results: the `history` dictionary

The dictionary `history` contains four entries this time: one per metric that was being monitored during training and during validation.


```python
model_val_dict = model_val.history
model_val_dict.keys()
```


```python
results_train = model.evaluate(train_final, label_train_final)
```


```python
results_test = model.evaluate(X_test, y_test)
```


```python
results_train
```


```python
results_test
```


```python
# __SOLUTION__ 
model_val_dict = model_val.history
model_val_dict.keys()
```




    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])




```python
# __SOLUTION__ 
results_train = model.evaluate(train_final, label_train_final)
```

    7500/7500 [==============================] - 0s 20us/step



```python
# __SOLUTION__ 
results_test = model.evaluate(X_test, y_test)
```

    1500/1500 [==============================] - 0s 23us/step



```python
# __SOLUTION__ 
results_train
```




    [0.36572940867741904, 0.8845333333015442]




```python
# __SOLUTION__ 
results_test
```




    [0.6589062994321188, 0.7606666666666667]



To interpret these results, run the cell below:


```python
model.metrics_names
```


```python
# __SOLUTION__
model.metrics_names
```




    ['loss', 'acc']



The first element of the list returned by `model.evaluate` is the loss, and the second is the accuracy score. 

Note that the result you obtained here isn't exactly the same as before. This is because the training set is slightly different! You removed 1000 instances for validation!

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


```python
# __SOLUTION__ 
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


![png](index_files/index_56_0.png)



```python
# __SOLUTION__ 
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


![png](index_files/index_57_0.png)


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


```python
# __SOLUTION__ 
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
    7500/7500 [==============================] - 0s 26us/step - loss: 1.9477 - acc: 0.1487 - val_loss: 1.9288 - val_acc: 0.1730
    Epoch 2/60
    7500/7500 [==============================] - 0s 15us/step - loss: 1.9137 - acc: 0.1896 - val_loss: 1.9019 - val_acc: 0.2030
    Epoch 3/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8861 - acc: 0.2344 - val_loss: 1.8738 - val_acc: 0.2360
    Epoch 4/60
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8564 - acc: 0.2651 - val_loss: 1.8427 - val_acc: 0.2690
    Epoch 5/60
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8229 - acc: 0.2975 - val_loss: 1.8066 - val_acc: 0.3010
    Epoch 6/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7838 - acc: 0.3292 - val_loss: 1.7642 - val_acc: 0.3310
    Epoch 7/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7382 - acc: 0.3668 - val_loss: 1.7151 - val_acc: 0.3730
    Epoch 8/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6871 - acc: 0.4040 - val_loss: 1.6609 - val_acc: 0.4060
    Epoch 9/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6319 - acc: 0.4411 - val_loss: 1.6053 - val_acc: 0.4250
    Epoch 10/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5736 - acc: 0.4669 - val_loss: 1.5483 - val_acc: 0.4550
    Epoch 11/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5148 - acc: 0.4999 - val_loss: 1.4912 - val_acc: 0.4890
    Epoch 12/60
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4563 - acc: 0.5285 - val_loss: 1.4372 - val_acc: 0.5040
    Epoch 13/60
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3985 - acc: 0.5540 - val_loss: 1.3799 - val_acc: 0.5370
    Epoch 14/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3418 - acc: 0.5745 - val_loss: 1.3280 - val_acc: 0.5550
    Epoch 15/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2879 - acc: 0.5955 - val_loss: 1.2769 - val_acc: 0.5880
    Epoch 16/60
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2362 - acc: 0.6115 - val_loss: 1.2305 - val_acc: 0.5990
    Epoch 17/60
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1878 - acc: 0.6275 - val_loss: 1.1868 - val_acc: 0.6190
    Epoch 18/60
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1427 - acc: 0.6423 - val_loss: 1.1440 - val_acc: 0.6370
    Epoch 19/60
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1005 - acc: 0.6547 - val_loss: 1.1053 - val_acc: 0.6490
    Epoch 20/60
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0616 - acc: 0.6681 - val_loss: 1.0697 - val_acc: 0.6660
    Epoch 21/60
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0256 - acc: 0.6763 - val_loss: 1.0378 - val_acc: 0.6680
    Epoch 22/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9917 - acc: 0.6852 - val_loss: 1.0092 - val_acc: 0.6670
    Epoch 23/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9610 - acc: 0.6929 - val_loss: 0.9807 - val_acc: 0.6700
    Epoch 24/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9326 - acc: 0.6987 - val_loss: 0.9550 - val_acc: 0.6860
    Epoch 25/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9063 - acc: 0.7063 - val_loss: 0.9300 - val_acc: 0.6900
    Epoch 26/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8818 - acc: 0.7128 - val_loss: 0.9089 - val_acc: 0.6960
    Epoch 27/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8596 - acc: 0.7163 - val_loss: 0.8916 - val_acc: 0.7180
    Epoch 28/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8389 - acc: 0.7217 - val_loss: 0.8714 - val_acc: 0.7210
    Epoch 29/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8198 - acc: 0.7272 - val_loss: 0.8551 - val_acc: 0.7090
    Epoch 30/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8018 - acc: 0.7305 - val_loss: 0.8391 - val_acc: 0.7140
    Epoch 31/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.7849 - acc: 0.7347 - val_loss: 0.8255 - val_acc: 0.7150
    Epoch 32/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.7693 - acc: 0.7383 - val_loss: 0.8122 - val_acc: 0.7230
    Epoch 33/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.7547 - acc: 0.7441 - val_loss: 0.8042 - val_acc: 0.7300
    Epoch 34/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.7411 - acc: 0.7457 - val_loss: 0.7905 - val_acc: 0.7260
    Epoch 35/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.7281 - acc: 0.7516 - val_loss: 0.7811 - val_acc: 0.7380
    Epoch 36/60
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7156 - acc: 0.7549 - val_loss: 0.7718 - val_acc: 0.7330
    Epoch 37/60
    7500/7500 [==============================] - 0s 16us/step - loss: 0.7042 - acc: 0.7557 - val_loss: 0.7637 - val_acc: 0.7350
    Epoch 38/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6929 - acc: 0.7611 - val_loss: 0.7548 - val_acc: 0.7410
    Epoch 39/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6825 - acc: 0.7636 - val_loss: 0.7510 - val_acc: 0.7380
    Epoch 40/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6728 - acc: 0.7679 - val_loss: 0.7407 - val_acc: 0.7500
    Epoch 41/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6634 - acc: 0.7728 - val_loss: 0.7348 - val_acc: 0.7500
    Epoch 42/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6549 - acc: 0.7723 - val_loss: 0.7281 - val_acc: 0.7500
    Epoch 43/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6459 - acc: 0.7769 - val_loss: 0.7219 - val_acc: 0.7510
    Epoch 44/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6375 - acc: 0.7796 - val_loss: 0.7186 - val_acc: 0.7510
    Epoch 45/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6295 - acc: 0.7784 - val_loss: 0.7114 - val_acc: 0.7520
    Epoch 46/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6218 - acc: 0.7841 - val_loss: 0.7102 - val_acc: 0.7570
    Epoch 47/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.6151 - acc: 0.7847 - val_loss: 0.7059 - val_acc: 0.7480
    Epoch 48/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.6078 - acc: 0.7883 - val_loss: 0.6986 - val_acc: 0.7550
    Epoch 49/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.6011 - acc: 0.7909 - val_loss: 0.6945 - val_acc: 0.7570
    Epoch 50/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5940 - acc: 0.7921 - val_loss: 0.6902 - val_acc: 0.7560
    Epoch 51/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5881 - acc: 0.7952 - val_loss: 0.6890 - val_acc: 0.7540
    Epoch 52/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5815 - acc: 0.7957 - val_loss: 0.6852 - val_acc: 0.7550
    Epoch 53/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5756 - acc: 0.8000 - val_loss: 0.6827 - val_acc: 0.7580
    Epoch 54/60
    7500/7500 [==============================] - 0s 15us/step - loss: 0.5697 - acc: 0.8016 - val_loss: 0.6794 - val_acc: 0.7570
    Epoch 55/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5643 - acc: 0.8045 - val_loss: 0.6773 - val_acc: 0.7570
    Epoch 56/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5586 - acc: 0.8064 - val_loss: 0.6736 - val_acc: 0.7650
    Epoch 57/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5530 - acc: 0.8097 - val_loss: 0.6716 - val_acc: 0.7610
    Epoch 58/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5477 - acc: 0.8120 - val_loss: 0.6688 - val_acc: 0.7580
    Epoch 59/60
    7500/7500 [==============================] - 0s 14us/step - loss: 0.5427 - acc: 0.8117 - val_loss: 0.6664 - val_acc: 0.7640
    Epoch 60/60
    7500/7500 [==============================] - 0s 13us/step - loss: 0.5373 - acc: 0.8131 - val_loss: 0.6646 - val_acc: 0.7610


Now, you can use the test set to make label predictions


```python
results_train = model.evaluate(train_final, label_train_final)
```


```python
results_test = model.evaluate(X_test, y_test)
```


```python
results_train
```


```python
results_test
```


```python
# __SOLUTION__ 
results_train = model.evaluate(train_final, label_train_final)
```

    7500/7500 [==============================] - 0s 19us/step



```python
# __SOLUTION__ 
results_test = model.evaluate(X_test, y_test)
```

    1500/1500 [==============================] - 0s 20us/step



```python
# __SOLUTION__ 
results_train
```




    [0.5326700656572978, 0.8154666666666667]




```python
# __SOLUTION__ 
results_test
```




    [0.6795797921816508, 0.7439999996821086]



We've significantly reduced the variance, so this is already pretty good! your test set accuracy is slightly worse, but this model will definitely be more robust than the 120 epochs model you originally fit.

Now, take a look at how regularization techniques can further improve your model performance.

## L2 Regularization

First, take a look at L2 regularization. Keras makes L2 regularization easy. Simply add the `kernel_regularizer=keras.regularizers.l2(lambda_coeff)` parameter to any model layer. The `lambda_coeff` parameter determines the strength of the regularization you wish to perform.


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


```python
L2_model_dict = L2_model.history
L2_model_dict.keys()
```


```python
# __SOLUTION__ 
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
    7500/7500 [==============================] - 0s 32us/step - loss: 2.5921 - acc: 0.1567 - val_loss: 2.5850 - val_acc: 0.1640
    Epoch 2/120
    7500/7500 [==============================] - 0s 16us/step - loss: 2.5698 - acc: 0.1959 - val_loss: 2.5664 - val_acc: 0.1860
    Epoch 3/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.5494 - acc: 0.2169 - val_loss: 2.5476 - val_acc: 0.2060
    Epoch 4/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.5274 - acc: 0.2321 - val_loss: 2.5265 - val_acc: 0.2270
    Epoch 5/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.5025 - acc: 0.2508 - val_loss: 2.5023 - val_acc: 0.2480
    Epoch 6/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.4744 - acc: 0.2737 - val_loss: 2.4744 - val_acc: 0.2640
    Epoch 7/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.4428 - acc: 0.2993 - val_loss: 2.4420 - val_acc: 0.2910
    Epoch 8/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.4078 - acc: 0.3293 - val_loss: 2.4061 - val_acc: 0.3070
    Epoch 9/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.3691 - acc: 0.3585 - val_loss: 2.3662 - val_acc: 0.3300
    Epoch 10/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.3269 - acc: 0.3832 - val_loss: 2.3229 - val_acc: 0.3610
    Epoch 11/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.2823 - acc: 0.4083 - val_loss: 2.2776 - val_acc: 0.3860
    Epoch 12/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.2358 - acc: 0.4289 - val_loss: 2.2319 - val_acc: 0.4130
    Epoch 13/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.1894 - acc: 0.4473 - val_loss: 2.1851 - val_acc: 0.4380
    Epoch 14/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.1425 - acc: 0.4705 - val_loss: 2.1399 - val_acc: 0.4530
    Epoch 15/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.0965 - acc: 0.4948 - val_loss: 2.0944 - val_acc: 0.4840
    Epoch 16/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.0513 - acc: 0.5165 - val_loss: 2.0505 - val_acc: 0.4970
    Epoch 17/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.0068 - acc: 0.5387 - val_loss: 2.0073 - val_acc: 0.5120
    Epoch 18/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.9637 - acc: 0.5597 - val_loss: 1.9657 - val_acc: 0.5370
    Epoch 19/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.9221 - acc: 0.5777 - val_loss: 1.9262 - val_acc: 0.5540
    Epoch 20/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8817 - acc: 0.5941 - val_loss: 1.8866 - val_acc: 0.5850
    Epoch 21/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.8430 - acc: 0.6116 - val_loss: 1.8501 - val_acc: 0.5970
    Epoch 22/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8053 - acc: 0.6243 - val_loss: 1.8134 - val_acc: 0.5990
    Epoch 23/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7693 - acc: 0.6371 - val_loss: 1.7793 - val_acc: 0.6250
    Epoch 24/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.7344 - acc: 0.6539 - val_loss: 1.7453 - val_acc: 0.6310
    Epoch 25/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7014 - acc: 0.6613 - val_loss: 1.7131 - val_acc: 0.6450
    Epoch 26/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6695 - acc: 0.6723 - val_loss: 1.6831 - val_acc: 0.6480
    Epoch 27/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6394 - acc: 0.6787 - val_loss: 1.6539 - val_acc: 0.6590
    Epoch 28/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6101 - acc: 0.6903 - val_loss: 1.6272 - val_acc: 0.6600
    Epoch 29/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5826 - acc: 0.6960 - val_loss: 1.6014 - val_acc: 0.6670
    Epoch 30/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.5569 - acc: 0.7019 - val_loss: 1.5764 - val_acc: 0.6710
    Epoch 31/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5321 - acc: 0.7067 - val_loss: 1.5536 - val_acc: 0.6770
    Epoch 32/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5088 - acc: 0.7143 - val_loss: 1.5317 - val_acc: 0.6760
    Epoch 33/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.4862 - acc: 0.7193 - val_loss: 1.5165 - val_acc: 0.6820
    Epoch 34/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.4657 - acc: 0.7224 - val_loss: 1.4921 - val_acc: 0.6860
    Epoch 35/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4454 - acc: 0.7251 - val_loss: 1.4767 - val_acc: 0.6880
    Epoch 36/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4263 - acc: 0.7303 - val_loss: 1.4572 - val_acc: 0.6850
    Epoch 37/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.4088 - acc: 0.7332 - val_loss: 1.4415 - val_acc: 0.6940
    Epoch 38/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3916 - acc: 0.7379 - val_loss: 1.4261 - val_acc: 0.6990
    Epoch 39/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3758 - acc: 0.7403 - val_loss: 1.4127 - val_acc: 0.7050
    Epoch 40/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.3605 - acc: 0.7451 - val_loss: 1.3989 - val_acc: 0.7050
    Epoch 41/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3461 - acc: 0.7489 - val_loss: 1.3854 - val_acc: 0.7040
    Epoch 42/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3322 - acc: 0.7516 - val_loss: 1.3743 - val_acc: 0.7120
    Epoch 43/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3192 - acc: 0.7556 - val_loss: 1.3621 - val_acc: 0.7090
    Epoch 44/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.3067 - acc: 0.7563 - val_loss: 1.3517 - val_acc: 0.7110
    Epoch 45/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.2944 - acc: 0.7587 - val_loss: 1.3443 - val_acc: 0.7200
    Epoch 46/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.2832 - acc: 0.7615 - val_loss: 1.3308 - val_acc: 0.7220
    Epoch 47/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2715 - acc: 0.7657 - val_loss: 1.3224 - val_acc: 0.7270
    Epoch 48/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2611 - acc: 0.7664 - val_loss: 1.3139 - val_acc: 0.7270
    Epoch 49/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2507 - acc: 0.7707 - val_loss: 1.3056 - val_acc: 0.7300
    Epoch 50/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.2405 - acc: 0.7720 - val_loss: 1.2965 - val_acc: 0.7330
    Epoch 51/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.2312 - acc: 0.7773 - val_loss: 1.2895 - val_acc: 0.7330
    Epoch 52/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2217 - acc: 0.7768 - val_loss: 1.2821 - val_acc: 0.7300
    Epoch 53/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2132 - acc: 0.7785 - val_loss: 1.2746 - val_acc: 0.7310
    Epoch 54/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.2043 - acc: 0.7797 - val_loss: 1.2684 - val_acc: 0.7420
    Epoch 55/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.1956 - acc: 0.7817 - val_loss: 1.2611 - val_acc: 0.7390
    Epoch 56/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.1874 - acc: 0.7831 - val_loss: 1.2570 - val_acc: 0.7420
    Epoch 57/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1799 - acc: 0.7864 - val_loss: 1.2488 - val_acc: 0.7390
    Epoch 58/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1715 - acc: 0.7901 - val_loss: 1.2442 - val_acc: 0.7400
    Epoch 59/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1640 - acc: 0.7888 - val_loss: 1.2379 - val_acc: 0.7380
    Epoch 60/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1564 - acc: 0.7945 - val_loss: 1.2324 - val_acc: 0.7480
    Epoch 61/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1495 - acc: 0.7951 - val_loss: 1.2262 - val_acc: 0.7440
    Epoch 62/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1423 - acc: 0.7980 - val_loss: 1.2207 - val_acc: 0.7520
    Epoch 63/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1354 - acc: 0.7992 - val_loss: 1.2156 - val_acc: 0.7550
    Epoch 64/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.1284 - acc: 0.8000 - val_loss: 1.2109 - val_acc: 0.7530
    Epoch 65/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1222 - acc: 0.8048 - val_loss: 1.2067 - val_acc: 0.7490
    Epoch 66/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1155 - acc: 0.8059 - val_loss: 1.2018 - val_acc: 0.7490
    Epoch 67/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1086 - acc: 0.8048 - val_loss: 1.1960 - val_acc: 0.7540
    Epoch 68/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1024 - acc: 0.8084 - val_loss: 1.1917 - val_acc: 0.7570
    Epoch 69/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0962 - acc: 0.8092 - val_loss: 1.1876 - val_acc: 0.7600
    Epoch 70/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0903 - acc: 0.8099 - val_loss: 1.1832 - val_acc: 0.7590
    Epoch 71/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0843 - acc: 0.8116 - val_loss: 1.1802 - val_acc: 0.7510
    Epoch 72/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0786 - acc: 0.8147 - val_loss: 1.1753 - val_acc: 0.7600
    Epoch 73/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.0728 - acc: 0.8160 - val_loss: 1.1714 - val_acc: 0.7580
    Epoch 74/120
    7500/7500 [==============================] - 0s 12us/step - loss: 1.0669 - acc: 0.8191 - val_loss: 1.1673 - val_acc: 0.7600
    Epoch 75/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0615 - acc: 0.8205 - val_loss: 1.1637 - val_acc: 0.7580
    Epoch 76/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0557 - acc: 0.8200 - val_loss: 1.1594 - val_acc: 0.7620
    Epoch 77/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0501 - acc: 0.8220 - val_loss: 1.1574 - val_acc: 0.7580
    Epoch 78/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0449 - acc: 0.8260 - val_loss: 1.1525 - val_acc: 0.7650
    Epoch 79/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0398 - acc: 0.8255 - val_loss: 1.1509 - val_acc: 0.7590
    Epoch 80/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0343 - acc: 0.8261 - val_loss: 1.1460 - val_acc: 0.7620
    Epoch 81/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0293 - acc: 0.8253 - val_loss: 1.1417 - val_acc: 0.7690
    Epoch 82/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0241 - acc: 0.8292 - val_loss: 1.1390 - val_acc: 0.7650
    Epoch 83/120
    7500/7500 [==============================] - 0s 13us/step - loss: 1.0190 - acc: 0.8311 - val_loss: 1.1356 - val_acc: 0.7640
    Epoch 84/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0144 - acc: 0.8303 - val_loss: 1.1334 - val_acc: 0.7620
    Epoch 85/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0092 - acc: 0.8321 - val_loss: 1.1299 - val_acc: 0.7690
    Epoch 86/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0045 - acc: 0.8329 - val_loss: 1.1262 - val_acc: 0.7660
    Epoch 87/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9997 - acc: 0.8355 - val_loss: 1.1237 - val_acc: 0.7660
    Epoch 88/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9950 - acc: 0.8363 - val_loss: 1.1208 - val_acc: 0.7730
    Epoch 89/120
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9906 - acc: 0.8373 - val_loss: 1.1182 - val_acc: 0.7690
    Epoch 90/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9862 - acc: 0.8375 - val_loss: 1.1134 - val_acc: 0.7750
    Epoch 91/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9811 - acc: 0.8396 - val_loss: 1.1114 - val_acc: 0.7720
    Epoch 92/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9768 - acc: 0.8395 - val_loss: 1.1088 - val_acc: 0.7690
    Epoch 93/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9722 - acc: 0.8420 - val_loss: 1.1057 - val_acc: 0.7770
    Epoch 94/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9678 - acc: 0.8416 - val_loss: 1.1051 - val_acc: 0.7740
    Epoch 95/120
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9636 - acc: 0.8431 - val_loss: 1.1009 - val_acc: 0.7710
    Epoch 96/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9591 - acc: 0.8447 - val_loss: 1.0981 - val_acc: 0.7700
    Epoch 97/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9551 - acc: 0.8452 - val_loss: 1.0963 - val_acc: 0.7770
    Epoch 98/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9508 - acc: 0.8488 - val_loss: 1.0923 - val_acc: 0.7720
    Epoch 99/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9467 - acc: 0.8480 - val_loss: 1.0897 - val_acc: 0.7750
    Epoch 100/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9422 - acc: 0.8501 - val_loss: 1.0871 - val_acc: 0.7760
    Epoch 101/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9379 - acc: 0.8507 - val_loss: 1.0872 - val_acc: 0.7760
    Epoch 102/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9342 - acc: 0.8500 - val_loss: 1.0831 - val_acc: 0.7730
    Epoch 103/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9300 - acc: 0.8536 - val_loss: 1.0797 - val_acc: 0.7790
    Epoch 104/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9260 - acc: 0.8548 - val_loss: 1.0781 - val_acc: 0.7780
    Epoch 105/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9222 - acc: 0.8539 - val_loss: 1.0767 - val_acc: 0.7740
    Epoch 106/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9179 - acc: 0.8545 - val_loss: 1.0732 - val_acc: 0.7800
    Epoch 107/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9142 - acc: 0.8577 - val_loss: 1.0698 - val_acc: 0.7820
    Epoch 108/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9104 - acc: 0.8580 - val_loss: 1.0676 - val_acc: 0.7790
    Epoch 109/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9066 - acc: 0.8607 - val_loss: 1.0660 - val_acc: 0.7770
    Epoch 110/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.9029 - acc: 0.8601 - val_loss: 1.0667 - val_acc: 0.7800
    Epoch 111/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8993 - acc: 0.8624 - val_loss: 1.0622 - val_acc: 0.7770
    Epoch 112/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8950 - acc: 0.8620 - val_loss: 1.0609 - val_acc: 0.7820
    Epoch 113/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8915 - acc: 0.8652 - val_loss: 1.0580 - val_acc: 0.7790
    Epoch 114/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8878 - acc: 0.8652 - val_loss: 1.0567 - val_acc: 0.7760
    Epoch 115/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8844 - acc: 0.8677 - val_loss: 1.0534 - val_acc: 0.7840
    Epoch 116/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8806 - acc: 0.8643 - val_loss: 1.0511 - val_acc: 0.7870
    Epoch 117/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8772 - acc: 0.8692 - val_loss: 1.0499 - val_acc: 0.7790
    Epoch 118/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8740 - acc: 0.8697 - val_loss: 1.0493 - val_acc: 0.7790
    Epoch 119/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8700 - acc: 0.8712 - val_loss: 1.0465 - val_acc: 0.7870
    Epoch 120/120
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8666 - acc: 0.8723 - val_loss: 1.0436 - val_acc: 0.7890



```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
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


![png](index_files/index_80_0.png)


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


```python
# __SOLUTION__ 
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
    7500/7500 [==============================] - 0s 33us/step - loss: 15.9796 - acc: 0.1627 - val_loss: 15.5710 - val_acc: 0.1460
    Epoch 2/120
    7500/7500 [==============================] - 0s 15us/step - loss: 15.2173 - acc: 0.1788 - val_loss: 14.8255 - val_acc: 0.1790
    Epoch 3/120
    7500/7500 [==============================] - 0s 14us/step - loss: 14.4813 - acc: 0.2037 - val_loss: 14.1016 - val_acc: 0.2020
    Epoch 4/120
    7500/7500 [==============================] - 0s 14us/step - loss: 13.7662 - acc: 0.2288 - val_loss: 13.3963 - val_acc: 0.2230
    Epoch 5/120
    7500/7500 [==============================] - 0s 14us/step - loss: 13.0705 - acc: 0.2579 - val_loss: 12.7102 - val_acc: 0.2730
    Epoch 6/120
    7500/7500 [==============================] - 0s 14us/step - loss: 12.3944 - acc: 0.2924 - val_loss: 12.0434 - val_acc: 0.3070
    Epoch 7/120
    7500/7500 [==============================] - 0s 14us/step - loss: 11.7371 - acc: 0.3287 - val_loss: 11.3953 - val_acc: 0.3420
    Epoch 8/120
    7500/7500 [==============================] - 0s 16us/step - loss: 11.0991 - acc: 0.3684 - val_loss: 10.7670 - val_acc: 0.3910
    Epoch 9/120
    7500/7500 [==============================] - 0s 15us/step - loss: 10.4811 - acc: 0.4105 - val_loss: 10.1588 - val_acc: 0.4520
    Epoch 10/120
    7500/7500 [==============================] - 0s 15us/step - loss: 9.8834 - acc: 0.4467 - val_loss: 9.5716 - val_acc: 0.4770
    Epoch 11/120
    7500/7500 [==============================] - 0s 14us/step - loss: 9.3070 - acc: 0.4689 - val_loss: 9.0056 - val_acc: 0.5190
    Epoch 12/120
    7500/7500 [==============================] - 0s 14us/step - loss: 8.7520 - acc: 0.4991 - val_loss: 8.4611 - val_acc: 0.5370
    Epoch 13/120
    7500/7500 [==============================] - 0s 15us/step - loss: 8.2183 - acc: 0.5209 - val_loss: 7.9383 - val_acc: 0.5560
    Epoch 14/120
    7500/7500 [==============================] - 0s 16us/step - loss: 7.7066 - acc: 0.5416 - val_loss: 7.4384 - val_acc: 0.5620
    Epoch 15/120
    7500/7500 [==============================] - 0s 15us/step - loss: 7.2177 - acc: 0.5536 - val_loss: 6.9622 - val_acc: 0.5660
    Epoch 16/120
    7500/7500 [==============================] - 0s 15us/step - loss: 6.7523 - acc: 0.5657 - val_loss: 6.5078 - val_acc: 0.5730
    Epoch 17/120
    7500/7500 [==============================] - 0s 15us/step - loss: 6.3090 - acc: 0.5799 - val_loss: 6.0762 - val_acc: 0.5770
    Epoch 18/120
    7500/7500 [==============================] - 0s 14us/step - loss: 5.8880 - acc: 0.5887 - val_loss: 5.6693 - val_acc: 0.5940
    Epoch 19/120
    7500/7500 [==============================] - 0s 15us/step - loss: 5.4904 - acc: 0.6005 - val_loss: 5.2825 - val_acc: 0.5960
    Epoch 20/120
    7500/7500 [==============================] - 0s 17us/step - loss: 5.1151 - acc: 0.6148 - val_loss: 4.9184 - val_acc: 0.6120
    Epoch 21/120
    7500/7500 [==============================] - 0s 17us/step - loss: 4.7622 - acc: 0.6176 - val_loss: 4.5774 - val_acc: 0.6200
    Epoch 22/120
    7500/7500 [==============================] - 0s 16us/step - loss: 4.4311 - acc: 0.6248 - val_loss: 4.2584 - val_acc: 0.6300
    Epoch 23/120
    7500/7500 [==============================] - 0s 16us/step - loss: 4.1228 - acc: 0.6341 - val_loss: 3.9611 - val_acc: 0.6410
    Epoch 24/120
    7500/7500 [==============================] - 0s 14us/step - loss: 3.8381 - acc: 0.6408 - val_loss: 3.6884 - val_acc: 0.6330
    Epoch 25/120
    7500/7500 [==============================] - 0s 14us/step - loss: 3.5757 - acc: 0.6425 - val_loss: 3.4383 - val_acc: 0.6430
    Epoch 26/120
    7500/7500 [==============================] - 0s 15us/step - loss: 3.3357 - acc: 0.6475 - val_loss: 3.2104 - val_acc: 0.6490
    Epoch 27/120
    7500/7500 [==============================] - 0s 14us/step - loss: 3.1180 - acc: 0.6479 - val_loss: 3.0026 - val_acc: 0.6590
    Epoch 28/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.9218 - acc: 0.6563 - val_loss: 2.8177 - val_acc: 0.6630
    Epoch 29/120
    7500/7500 [==============================] - 0s 16us/step - loss: 2.7474 - acc: 0.6557 - val_loss: 2.6530 - val_acc: 0.6740
    Epoch 30/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.5935 - acc: 0.6636 - val_loss: 2.5095 - val_acc: 0.6690
    Epoch 31/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.4601 - acc: 0.6637 - val_loss: 2.3879 - val_acc: 0.6700
    Epoch 32/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.3467 - acc: 0.6652 - val_loss: 2.2822 - val_acc: 0.6720
    Epoch 33/120
    7500/7500 [==============================] - 0s 16us/step - loss: 2.2526 - acc: 0.6667 - val_loss: 2.1983 - val_acc: 0.6660
    Epoch 34/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.1769 - acc: 0.6665 - val_loss: 2.1302 - val_acc: 0.6690
    Epoch 35/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.1182 - acc: 0.6691 - val_loss: 2.0786 - val_acc: 0.6680
    Epoch 36/120
    7500/7500 [==============================] - 0s 14us/step - loss: 2.0744 - acc: 0.6689 - val_loss: 2.0416 - val_acc: 0.6700
    Epoch 37/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.0423 - acc: 0.6695 - val_loss: 2.0136 - val_acc: 0.6700
    Epoch 38/120
    7500/7500 [==============================] - 0s 15us/step - loss: 2.0163 - acc: 0.6708 - val_loss: 1.9868 - val_acc: 0.6810
    Epoch 39/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.9932 - acc: 0.6729 - val_loss: 1.9660 - val_acc: 0.6770
    Epoch 40/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.9723 - acc: 0.6716 - val_loss: 1.9448 - val_acc: 0.6760
    Epoch 41/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.9523 - acc: 0.6743 - val_loss: 1.9250 - val_acc: 0.6800
    Epoch 42/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.9340 - acc: 0.6725 - val_loss: 1.9054 - val_acc: 0.6850
    Epoch 43/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.9168 - acc: 0.6760 - val_loss: 1.8926 - val_acc: 0.6850
    Epoch 44/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.9007 - acc: 0.6763 - val_loss: 1.8718 - val_acc: 0.6820
    Epoch 45/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8842 - acc: 0.6779 - val_loss: 1.8568 - val_acc: 0.6880
    Epoch 46/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8687 - acc: 0.6792 - val_loss: 1.8380 - val_acc: 0.6900
    Epoch 47/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8541 - acc: 0.6820 - val_loss: 1.8240 - val_acc: 0.6930
    Epoch 48/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8395 - acc: 0.6808 - val_loss: 1.8081 - val_acc: 0.6940
    Epoch 49/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8263 - acc: 0.6837 - val_loss: 1.7973 - val_acc: 0.6930
    Epoch 50/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.8124 - acc: 0.6852 - val_loss: 1.7852 - val_acc: 0.6930
    Epoch 51/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7997 - acc: 0.6860 - val_loss: 1.7679 - val_acc: 0.6960
    Epoch 52/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7872 - acc: 0.6863 - val_loss: 1.7584 - val_acc: 0.7020
    Epoch 53/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.7748 - acc: 0.6876 - val_loss: 1.7445 - val_acc: 0.7010
    Epoch 54/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.7633 - acc: 0.6892 - val_loss: 1.7318 - val_acc: 0.6980
    Epoch 55/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.7519 - acc: 0.6905 - val_loss: 1.7211 - val_acc: 0.7000
    Epoch 56/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7405 - acc: 0.6901 - val_loss: 1.7119 - val_acc: 0.6920
    Epoch 57/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7300 - acc: 0.6935 - val_loss: 1.6975 - val_acc: 0.7000
    Epoch 58/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.7188 - acc: 0.6941 - val_loss: 1.6861 - val_acc: 0.7000
    Epoch 59/120
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7082 - acc: 0.6948 - val_loss: 1.6760 - val_acc: 0.6990
    Epoch 60/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6980 - acc: 0.6952 - val_loss: 1.6681 - val_acc: 0.7000
    Epoch 61/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6876 - acc: 0.6956 - val_loss: 1.6627 - val_acc: 0.6980
    Epoch 62/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6784 - acc: 0.6976 - val_loss: 1.6480 - val_acc: 0.7080
    Epoch 63/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6691 - acc: 0.6964 - val_loss: 1.6410 - val_acc: 0.7030
    Epoch 64/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6591 - acc: 0.6989 - val_loss: 1.6289 - val_acc: 0.7040
    Epoch 65/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6500 - acc: 0.6987 - val_loss: 1.6179 - val_acc: 0.7060
    Epoch 66/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6411 - acc: 0.6993 - val_loss: 1.6120 - val_acc: 0.7050
    Epoch 67/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6325 - acc: 0.7028 - val_loss: 1.6042 - val_acc: 0.7170
    Epoch 68/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.6234 - acc: 0.7023 - val_loss: 1.5936 - val_acc: 0.7080
    Epoch 69/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6148 - acc: 0.7020 - val_loss: 1.5839 - val_acc: 0.7100
    Epoch 70/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6065 - acc: 0.7033 - val_loss: 1.5748 - val_acc: 0.7090
    Epoch 71/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5987 - acc: 0.7041 - val_loss: 1.5686 - val_acc: 0.7110
    Epoch 72/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5907 - acc: 0.7036 - val_loss: 1.5585 - val_acc: 0.7120
    Epoch 73/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.5819 - acc: 0.7048 - val_loss: 1.5523 - val_acc: 0.7130
    Epoch 74/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5745 - acc: 0.7059 - val_loss: 1.5458 - val_acc: 0.7130
    Epoch 75/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5669 - acc: 0.7059 - val_loss: 1.5359 - val_acc: 0.7070
    Epoch 76/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.5592 - acc: 0.7052 - val_loss: 1.5363 - val_acc: 0.7090
    Epoch 77/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5518 - acc: 0.7072 - val_loss: 1.5207 - val_acc: 0.7170
    Epoch 78/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5437 - acc: 0.7084 - val_loss: 1.5138 - val_acc: 0.7160
    Epoch 79/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5364 - acc: 0.7081 - val_loss: 1.5059 - val_acc: 0.7100
    Epoch 80/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5292 - acc: 0.7068 - val_loss: 1.4993 - val_acc: 0.7150
    Epoch 81/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5219 - acc: 0.7085 - val_loss: 1.4942 - val_acc: 0.7140
    Epoch 82/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5149 - acc: 0.7099 - val_loss: 1.4855 - val_acc: 0.7160
    Epoch 83/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5077 - acc: 0.7101 - val_loss: 1.4859 - val_acc: 0.7100
    Epoch 84/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.5013 - acc: 0.7104 - val_loss: 1.4709 - val_acc: 0.7140
    Epoch 85/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4942 - acc: 0.7123 - val_loss: 1.4629 - val_acc: 0.7170
    Epoch 86/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.4873 - acc: 0.7116 - val_loss: 1.4567 - val_acc: 0.7160
    Epoch 87/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4807 - acc: 0.7128 - val_loss: 1.4498 - val_acc: 0.7180
    Epoch 88/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4741 - acc: 0.7113 - val_loss: 1.4458 - val_acc: 0.7200
    Epoch 89/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4674 - acc: 0.7145 - val_loss: 1.4378 - val_acc: 0.7220
    Epoch 90/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4610 - acc: 0.7141 - val_loss: 1.4309 - val_acc: 0.7160
    Epoch 91/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4543 - acc: 0.7139 - val_loss: 1.4271 - val_acc: 0.7180
    Epoch 92/120
    7500/7500 [==============================] - 0s 17us/step - loss: 1.4481 - acc: 0.7151 - val_loss: 1.4241 - val_acc: 0.7150
    Epoch 93/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4426 - acc: 0.7149 - val_loss: 1.4142 - val_acc: 0.7200
    Epoch 94/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.4363 - acc: 0.7157 - val_loss: 1.4057 - val_acc: 0.7220
    Epoch 95/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4297 - acc: 0.7175 - val_loss: 1.4016 - val_acc: 0.7220
    Epoch 96/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4235 - acc: 0.7157 - val_loss: 1.3969 - val_acc: 0.7170
    Epoch 97/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4183 - acc: 0.7176 - val_loss: 1.3934 - val_acc: 0.7170
    Epoch 98/120
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4127 - acc: 0.7171 - val_loss: 1.3846 - val_acc: 0.7210
    Epoch 99/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4071 - acc: 0.7184 - val_loss: 1.3792 - val_acc: 0.7230
    Epoch 100/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4010 - acc: 0.7185 - val_loss: 1.3727 - val_acc: 0.7250
    Epoch 101/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3953 - acc: 0.7199 - val_loss: 1.3676 - val_acc: 0.7230
    Epoch 102/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3899 - acc: 0.7199 - val_loss: 1.3630 - val_acc: 0.7250
    Epoch 103/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3839 - acc: 0.7191 - val_loss: 1.3587 - val_acc: 0.7270
    Epoch 104/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3786 - acc: 0.7184 - val_loss: 1.3498 - val_acc: 0.7260
    Epoch 105/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3733 - acc: 0.7207 - val_loss: 1.3476 - val_acc: 0.7230
    Epoch 106/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3682 - acc: 0.7211 - val_loss: 1.3406 - val_acc: 0.7310
    Epoch 107/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3630 - acc: 0.7220 - val_loss: 1.3410 - val_acc: 0.7220
    Epoch 108/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3579 - acc: 0.7213 - val_loss: 1.3310 - val_acc: 0.7290
    Epoch 109/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3522 - acc: 0.7229 - val_loss: 1.3241 - val_acc: 0.7270
    Epoch 110/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3476 - acc: 0.7211 - val_loss: 1.3213 - val_acc: 0.7310
    Epoch 111/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3434 - acc: 0.7211 - val_loss: 1.3167 - val_acc: 0.7290
    Epoch 112/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3370 - acc: 0.7232 - val_loss: 1.3130 - val_acc: 0.7290
    Epoch 113/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3324 - acc: 0.7257 - val_loss: 1.3118 - val_acc: 0.7250
    Epoch 114/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3285 - acc: 0.7253 - val_loss: 1.3017 - val_acc: 0.7300
    Epoch 115/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3228 - acc: 0.7252 - val_loss: 1.2974 - val_acc: 0.7270
    Epoch 116/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3179 - acc: 0.7237 - val_loss: 1.2956 - val_acc: 0.7230
    Epoch 117/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3138 - acc: 0.7245 - val_loss: 1.2914 - val_acc: 0.7280
    Epoch 118/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.3091 - acc: 0.7272 - val_loss: 1.2828 - val_acc: 0.7370
    Epoch 119/120
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3041 - acc: 0.7252 - val_loss: 1.2797 - val_acc: 0.7320
    Epoch 120/120
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2998 - acc: 0.7260 - val_loss: 1.2743 - val_acc: 0.7330



```python
# __SOLUTION__ 
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


![png](index_files/index_87_0.png)


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


```python
results_train = model.evaluate(train_final, label_train_final)

results_test = model.evaluate(X_test, y_test)
```


```python
results_train
```


```python
results_test
```


```python
# __SOLUTION__ 
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
    7500/7500 [==============================] - 0s 37us/step - loss: 16.0274 - acc: 0.1509 - val_loss: 15.6090 - val_acc: 0.1830
    Epoch 2/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 15.2518 - acc: 0.2101 - val_loss: 14.8593 - val_acc: 0.2300
    Epoch 3/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 14.5123 - acc: 0.2427 - val_loss: 14.1349 - val_acc: 0.2490
    Epoch 4/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 13.7952 - acc: 0.2671 - val_loss: 13.4313 - val_acc: 0.2580
    Epoch 5/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 13.0985 - acc: 0.2865 - val_loss: 12.7468 - val_acc: 0.2730
    Epoch 6/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 12.4211 - acc: 0.3085 - val_loss: 12.0808 - val_acc: 0.2840
    Epoch 7/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 11.7624 - acc: 0.3340 - val_loss: 11.4348 - val_acc: 0.2970
    Epoch 8/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 11.1228 - acc: 0.3593 - val_loss: 10.8059 - val_acc: 0.3240
    Epoch 9/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 10.5030 - acc: 0.3837 - val_loss: 10.1971 - val_acc: 0.3580
    Epoch 10/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 9.9031 - acc: 0.4139 - val_loss: 9.6082 - val_acc: 0.3940
    Epoch 11/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 9.3244 - acc: 0.4337 - val_loss: 9.0404 - val_acc: 0.4400
    Epoch 12/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 8.7676 - acc: 0.4657 - val_loss: 8.4948 - val_acc: 0.4680
    Epoch 13/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 8.2335 - acc: 0.4843 - val_loss: 7.9730 - val_acc: 0.4900
    Epoch 14/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 7.7227 - acc: 0.5056 - val_loss: 7.4752 - val_acc: 0.4910
    Epoch 15/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 7.2350 - acc: 0.5224 - val_loss: 6.9996 - val_acc: 0.5020
    Epoch 16/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 6.7700 - acc: 0.5429 - val_loss: 6.5474 - val_acc: 0.5060
    Epoch 17/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 6.3278 - acc: 0.5532 - val_loss: 6.1164 - val_acc: 0.5560
    Epoch 18/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 5.9088 - acc: 0.5732 - val_loss: 5.7112 - val_acc: 0.5460
    Epoch 19/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 5.5130 - acc: 0.5853 - val_loss: 5.3278 - val_acc: 0.5680
    Epoch 20/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 5.1403 - acc: 0.5957 - val_loss: 4.9652 - val_acc: 0.5870
    Epoch 21/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 4.7907 - acc: 0.6111 - val_loss: 4.6281 - val_acc: 0.5900
    Epoch 22/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 4.4631 - acc: 0.6163 - val_loss: 4.3137 - val_acc: 0.5730
    Epoch 23/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 4.1577 - acc: 0.6201 - val_loss: 4.0188 - val_acc: 0.5940
    Epoch 24/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 3.8753 - acc: 0.6313 - val_loss: 3.7495 - val_acc: 0.5960
    Epoch 25/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 3.6158 - acc: 0.6339 - val_loss: 3.4981 - val_acc: 0.6240
    Epoch 26/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 3.3785 - acc: 0.6432 - val_loss: 3.2716 - val_acc: 0.6390
    Epoch 27/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 3.1627 - acc: 0.6496 - val_loss: 3.0665 - val_acc: 0.6400
    Epoch 28/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.9672 - acc: 0.6548 - val_loss: 2.8825 - val_acc: 0.6340
    Epoch 29/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.7928 - acc: 0.6549 - val_loss: 2.7185 - val_acc: 0.6480
    Epoch 30/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.6407 - acc: 0.6569 - val_loss: 2.5759 - val_acc: 0.6400
    Epoch 31/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.5090 - acc: 0.6616 - val_loss: 2.4563 - val_acc: 0.6420
    Epoch 32/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.3976 - acc: 0.6577 - val_loss: 2.3529 - val_acc: 0.6500
    Epoch 33/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.3048 - acc: 0.6617 - val_loss: 2.2690 - val_acc: 0.6450
    Epoch 34/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 2.2299 - acc: 0.6616 - val_loss: 2.2018 - val_acc: 0.6640
    Epoch 35/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 2.1711 - acc: 0.6623 - val_loss: 2.1522 - val_acc: 0.6480
    Epoch 36/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 2.1270 - acc: 0.6639 - val_loss: 2.1113 - val_acc: 0.6610
    Epoch 37/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 2.0934 - acc: 0.6657 - val_loss: 2.0827 - val_acc: 0.6550
    Epoch 38/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 2.0668 - acc: 0.6652 - val_loss: 2.0577 - val_acc: 0.6630
    Epoch 39/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.0430 - acc: 0.6667 - val_loss: 2.0327 - val_acc: 0.6780
    Epoch 40/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 2.0210 - acc: 0.6716 - val_loss: 2.0104 - val_acc: 0.6790
    Epoch 41/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 2.0009 - acc: 0.6721 - val_loss: 1.9945 - val_acc: 0.6690
    Epoch 42/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9827 - acc: 0.6737 - val_loss: 1.9740 - val_acc: 0.6780
    Epoch 43/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9646 - acc: 0.6764 - val_loss: 1.9564 - val_acc: 0.6740
    Epoch 44/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9483 - acc: 0.6763 - val_loss: 1.9425 - val_acc: 0.6740
    Epoch 45/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.9323 - acc: 0.6760 - val_loss: 1.9231 - val_acc: 0.6810
    Epoch 46/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.9165 - acc: 0.6761 - val_loss: 1.9063 - val_acc: 0.6880
    Epoch 47/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.9018 - acc: 0.6781 - val_loss: 1.8916 - val_acc: 0.6890
    Epoch 48/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8874 - acc: 0.6787 - val_loss: 1.8810 - val_acc: 0.6800
    Epoch 49/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8742 - acc: 0.6789 - val_loss: 1.8635 - val_acc: 0.6910
    Epoch 50/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8612 - acc: 0.6833 - val_loss: 1.8499 - val_acc: 0.6830
    Epoch 51/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.8478 - acc: 0.6837 - val_loss: 1.8381 - val_acc: 0.6880
    Epoch 52/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.8357 - acc: 0.6820 - val_loss: 1.8259 - val_acc: 0.6890
    Epoch 53/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8236 - acc: 0.6825 - val_loss: 1.8141 - val_acc: 0.6870
    Epoch 54/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8113 - acc: 0.6867 - val_loss: 1.8037 - val_acc: 0.6860
    Epoch 55/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.8005 - acc: 0.6852 - val_loss: 1.7891 - val_acc: 0.6870
    Epoch 56/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7887 - acc: 0.6875 - val_loss: 1.7787 - val_acc: 0.6920
    Epoch 57/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7780 - acc: 0.6849 - val_loss: 1.7696 - val_acc: 0.6870
    Epoch 58/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7675 - acc: 0.6893 - val_loss: 1.7603 - val_acc: 0.6880
    Epoch 59/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7569 - acc: 0.6909 - val_loss: 1.7458 - val_acc: 0.6850
    Epoch 60/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7468 - acc: 0.6927 - val_loss: 1.7368 - val_acc: 0.6850
    Epoch 61/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7360 - acc: 0.6923 - val_loss: 1.7292 - val_acc: 0.6890
    Epoch 62/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.7264 - acc: 0.6931 - val_loss: 1.7187 - val_acc: 0.6900
    Epoch 63/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7168 - acc: 0.6924 - val_loss: 1.7071 - val_acc: 0.6930
    Epoch 64/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.7075 - acc: 0.6932 - val_loss: 1.7007 - val_acc: 0.6910
    Epoch 65/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6982 - acc: 0.6949 - val_loss: 1.6887 - val_acc: 0.6940
    Epoch 66/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6890 - acc: 0.6944 - val_loss: 1.6850 - val_acc: 0.6950
    Epoch 67/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6805 - acc: 0.6969 - val_loss: 1.6675 - val_acc: 0.7010
    Epoch 68/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6709 - acc: 0.6961 - val_loss: 1.6610 - val_acc: 0.6980
    Epoch 69/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6625 - acc: 0.6960 - val_loss: 1.6548 - val_acc: 0.6980
    Epoch 70/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6538 - acc: 0.7004 - val_loss: 1.6428 - val_acc: 0.7100
    Epoch 71/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.6455 - acc: 0.6969 - val_loss: 1.6360 - val_acc: 0.7050
    Epoch 72/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6366 - acc: 0.6983 - val_loss: 1.6260 - val_acc: 0.7040
    Epoch 73/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6284 - acc: 0.6993 - val_loss: 1.6161 - val_acc: 0.7040
    Epoch 74/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6203 - acc: 0.7012 - val_loss: 1.6088 - val_acc: 0.7070
    Epoch 75/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.6121 - acc: 0.7016 - val_loss: 1.6031 - val_acc: 0.7050
    Epoch 76/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.6049 - acc: 0.7003 - val_loss: 1.5935 - val_acc: 0.7100
    Epoch 77/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5963 - acc: 0.7032 - val_loss: 1.5849 - val_acc: 0.7070
    Epoch 78/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5885 - acc: 0.7021 - val_loss: 1.5746 - val_acc: 0.7170
    Epoch 79/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 1.5813 - acc: 0.7031 - val_loss: 1.5697 - val_acc: 0.7130
    Epoch 80/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5737 - acc: 0.7048 - val_loss: 1.5646 - val_acc: 0.7030
    Epoch 81/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5659 - acc: 0.7052 - val_loss: 1.5547 - val_acc: 0.7070
    Epoch 82/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5584 - acc: 0.7051 - val_loss: 1.5467 - val_acc: 0.7120
    Epoch 83/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5520 - acc: 0.7048 - val_loss: 1.5426 - val_acc: 0.7030
    Epoch 84/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 1.5440 - acc: 0.7052 - val_loss: 1.5317 - val_acc: 0.7130
    Epoch 85/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 1.5370 - acc: 0.7064 - val_loss: 1.5255 - val_acc: 0.7070
    Epoch 86/1000
    7500/7500 [==============================] - 0s 20us/step - loss: 1.5303 - acc: 0.7077 - val_loss: 1.5259 - val_acc: 0.7090
    Epoch 87/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5231 - acc: 0.7056 - val_loss: 1.5132 - val_acc: 0.7060
    Epoch 88/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5157 - acc: 0.7064 - val_loss: 1.5036 - val_acc: 0.7170
    Epoch 89/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5090 - acc: 0.7081 - val_loss: 1.5010 - val_acc: 0.7130
    Epoch 90/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.5030 - acc: 0.7080 - val_loss: 1.4887 - val_acc: 0.7150
    Epoch 91/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4958 - acc: 0.7105 - val_loss: 1.4841 - val_acc: 0.7170
    Epoch 92/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4891 - acc: 0.7085 - val_loss: 1.4806 - val_acc: 0.7110
    Epoch 93/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4827 - acc: 0.7085 - val_loss: 1.4709 - val_acc: 0.7140
    Epoch 94/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4757 - acc: 0.7107 - val_loss: 1.4671 - val_acc: 0.7070
    Epoch 95/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4700 - acc: 0.7097 - val_loss: 1.4557 - val_acc: 0.7170
    Epoch 96/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4632 - acc: 0.7111 - val_loss: 1.4505 - val_acc: 0.7200
    Epoch 97/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4567 - acc: 0.7119 - val_loss: 1.4511 - val_acc: 0.7120
    Epoch 98/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4507 - acc: 0.7112 - val_loss: 1.4411 - val_acc: 0.7170
    Epoch 99/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4447 - acc: 0.7121 - val_loss: 1.4307 - val_acc: 0.7210
    Epoch 100/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4383 - acc: 0.7117 - val_loss: 1.4264 - val_acc: 0.7180
    Epoch 101/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.4328 - acc: 0.7127 - val_loss: 1.4187 - val_acc: 0.7240
    Epoch 102/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4257 - acc: 0.7139 - val_loss: 1.4140 - val_acc: 0.7170
    Epoch 103/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4198 - acc: 0.7145 - val_loss: 1.4146 - val_acc: 0.7110
    Epoch 104/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4148 - acc: 0.7153 - val_loss: 1.4035 - val_acc: 0.7160
    Epoch 105/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4089 - acc: 0.7151 - val_loss: 1.4049 - val_acc: 0.7120
    Epoch 106/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.4031 - acc: 0.7168 - val_loss: 1.3912 - val_acc: 0.7140
    Epoch 107/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3970 - acc: 0.7160 - val_loss: 1.3837 - val_acc: 0.7230
    Epoch 108/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3913 - acc: 0.7165 - val_loss: 1.3815 - val_acc: 0.7180
    Epoch 109/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3857 - acc: 0.7172 - val_loss: 1.3785 - val_acc: 0.7170
    Epoch 110/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3801 - acc: 0.7177 - val_loss: 1.3696 - val_acc: 0.7190
    Epoch 111/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3747 - acc: 0.7179 - val_loss: 1.3609 - val_acc: 0.7210
    Epoch 112/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3687 - acc: 0.7180 - val_loss: 1.3592 - val_acc: 0.7190
    Epoch 113/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3636 - acc: 0.7189 - val_loss: 1.3552 - val_acc: 0.7210
    Epoch 114/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3587 - acc: 0.7188 - val_loss: 1.3471 - val_acc: 0.7220
    Epoch 115/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3543 - acc: 0.7197 - val_loss: 1.3407 - val_acc: 0.7240
    Epoch 116/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3481 - acc: 0.7200 - val_loss: 1.3418 - val_acc: 0.7160
    Epoch 117/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3433 - acc: 0.7196 - val_loss: 1.3327 - val_acc: 0.7220
    Epoch 118/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3378 - acc: 0.7215 - val_loss: 1.3257 - val_acc: 0.7260
    Epoch 119/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3329 - acc: 0.7207 - val_loss: 1.3204 - val_acc: 0.7230
    Epoch 120/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3283 - acc: 0.7215 - val_loss: 1.3173 - val_acc: 0.7190
    Epoch 121/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3232 - acc: 0.7237 - val_loss: 1.3200 - val_acc: 0.7200
    Epoch 122/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.3188 - acc: 0.7220 - val_loss: 1.3041 - val_acc: 0.7260
    Epoch 123/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3132 - acc: 0.7235 - val_loss: 1.3030 - val_acc: 0.7290
    Epoch 124/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3092 - acc: 0.7241 - val_loss: 1.3025 - val_acc: 0.7200
    Epoch 125/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.3046 - acc: 0.7224 - val_loss: 1.2919 - val_acc: 0.7300
    Epoch 126/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2991 - acc: 0.7236 - val_loss: 1.2861 - val_acc: 0.7320
    Epoch 127/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2944 - acc: 0.7257 - val_loss: 1.2816 - val_acc: 0.7290
    Epoch 128/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2902 - acc: 0.7265 - val_loss: 1.2780 - val_acc: 0.7270
    Epoch 129/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2859 - acc: 0.7255 - val_loss: 1.2773 - val_acc: 0.7240
    Epoch 130/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2815 - acc: 0.7275 - val_loss: 1.2704 - val_acc: 0.7300
    Epoch 131/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2766 - acc: 0.7277 - val_loss: 1.2657 - val_acc: 0.7290
    Epoch 132/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2726 - acc: 0.7253 - val_loss: 1.2640 - val_acc: 0.7280
    Epoch 133/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2684 - acc: 0.7264 - val_loss: 1.2599 - val_acc: 0.7230
    Epoch 134/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2638 - acc: 0.7279 - val_loss: 1.2520 - val_acc: 0.7330
    Epoch 135/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2600 - acc: 0.7287 - val_loss: 1.2499 - val_acc: 0.7330
    Epoch 136/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2560 - acc: 0.7291 - val_loss: 1.2467 - val_acc: 0.7310
    Epoch 137/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.2518 - acc: 0.7265 - val_loss: 1.2443 - val_acc: 0.7250
    Epoch 138/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2476 - acc: 0.7281 - val_loss: 1.2414 - val_acc: 0.7240
    Epoch 139/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2438 - acc: 0.7284 - val_loss: 1.2326 - val_acc: 0.7340
    Epoch 140/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 1.2397 - acc: 0.7299 - val_loss: 1.2344 - val_acc: 0.7280
    Epoch 141/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2371 - acc: 0.7292 - val_loss: 1.2240 - val_acc: 0.7350
    Epoch 142/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2318 - acc: 0.7312 - val_loss: 1.2215 - val_acc: 0.7320
    Epoch 143/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2285 - acc: 0.7304 - val_loss: 1.2174 - val_acc: 0.7320
    Epoch 144/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2243 - acc: 0.7296 - val_loss: 1.2122 - val_acc: 0.7320
    Epoch 145/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2201 - acc: 0.7308 - val_loss: 1.2065 - val_acc: 0.7370
    Epoch 146/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2166 - acc: 0.7328 - val_loss: 1.2052 - val_acc: 0.7360
    Epoch 147/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2131 - acc: 0.7332 - val_loss: 1.2050 - val_acc: 0.7400
    Epoch 148/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.2096 - acc: 0.7341 - val_loss: 1.1995 - val_acc: 0.7370
    Epoch 149/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2055 - acc: 0.7323 - val_loss: 1.2025 - val_acc: 0.7310
    Epoch 150/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.2028 - acc: 0.7315 - val_loss: 1.1933 - val_acc: 0.7340
    Epoch 151/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1985 - acc: 0.7331 - val_loss: 1.1869 - val_acc: 0.7330
    Epoch 152/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1953 - acc: 0.7348 - val_loss: 1.1856 - val_acc: 0.7330
    Epoch 153/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1920 - acc: 0.7343 - val_loss: 1.1816 - val_acc: 0.7340
    Epoch 154/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1888 - acc: 0.7355 - val_loss: 1.1795 - val_acc: 0.7460
    Epoch 155/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1853 - acc: 0.7353 - val_loss: 1.1754 - val_acc: 0.7370
    Epoch 156/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1825 - acc: 0.7352 - val_loss: 1.1697 - val_acc: 0.7390
    Epoch 157/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1790 - acc: 0.7360 - val_loss: 1.1748 - val_acc: 0.7370
    Epoch 158/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1761 - acc: 0.7379 - val_loss: 1.1683 - val_acc: 0.7310
    Epoch 159/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1725 - acc: 0.7371 - val_loss: 1.1677 - val_acc: 0.7340
    Epoch 160/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1692 - acc: 0.7352 - val_loss: 1.1580 - val_acc: 0.7470
    Epoch 161/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1657 - acc: 0.7377 - val_loss: 1.1575 - val_acc: 0.7440
    Epoch 162/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1632 - acc: 0.7364 - val_loss: 1.1519 - val_acc: 0.7400
    Epoch 163/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1608 - acc: 0.7363 - val_loss: 1.1519 - val_acc: 0.7430
    Epoch 164/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1581 - acc: 0.7372 - val_loss: 1.1500 - val_acc: 0.7410
    Epoch 165/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1543 - acc: 0.7385 - val_loss: 1.1563 - val_acc: 0.7330
    Epoch 166/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1527 - acc: 0.7396 - val_loss: 1.1532 - val_acc: 0.7380
    Epoch 167/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1492 - acc: 0.7407 - val_loss: 1.1402 - val_acc: 0.7440
    Epoch 168/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1462 - acc: 0.7393 - val_loss: 1.1360 - val_acc: 0.7430
    Epoch 169/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1437 - acc: 0.7379 - val_loss: 1.1340 - val_acc: 0.7440
    Epoch 170/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1410 - acc: 0.7393 - val_loss: 1.1346 - val_acc: 0.7470
    Epoch 171/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1379 - acc: 0.7409 - val_loss: 1.1311 - val_acc: 0.7480
    Epoch 172/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1359 - acc: 0.7389 - val_loss: 1.1364 - val_acc: 0.7370
    Epoch 173/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1336 - acc: 0.7391 - val_loss: 1.1329 - val_acc: 0.7370
    Epoch 174/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1309 - acc: 0.7405 - val_loss: 1.1215 - val_acc: 0.7450
    Epoch 175/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1282 - acc: 0.7399 - val_loss: 1.1222 - val_acc: 0.7490
    Epoch 176/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1264 - acc: 0.7408 - val_loss: 1.1158 - val_acc: 0.7470
    Epoch 177/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1225 - acc: 0.7411 - val_loss: 1.1146 - val_acc: 0.7470
    Epoch 178/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1208 - acc: 0.7415 - val_loss: 1.1158 - val_acc: 0.7450
    Epoch 179/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.1191 - acc: 0.7419 - val_loss: 1.1092 - val_acc: 0.7450
    Epoch 180/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1167 - acc: 0.7417 - val_loss: 1.1075 - val_acc: 0.7480
    Epoch 181/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1144 - acc: 0.7413 - val_loss: 1.1059 - val_acc: 0.7470
    Epoch 182/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1125 - acc: 0.7393 - val_loss: 1.1027 - val_acc: 0.7500
    Epoch 183/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1102 - acc: 0.7432 - val_loss: 1.1031 - val_acc: 0.7470
    Epoch 184/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1079 - acc: 0.7419 - val_loss: 1.1001 - val_acc: 0.7460
    Epoch 185/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1053 - acc: 0.7436 - val_loss: 1.0979 - val_acc: 0.7430
    Epoch 186/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.1033 - acc: 0.7420 - val_loss: 1.0990 - val_acc: 0.7430
    Epoch 187/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.1017 - acc: 0.7425 - val_loss: 1.0945 - val_acc: 0.7470
    Epoch 188/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0993 - acc: 0.7433 - val_loss: 1.0955 - val_acc: 0.7430
    Epoch 189/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0978 - acc: 0.7456 - val_loss: 1.0904 - val_acc: 0.7490
    Epoch 190/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0956 - acc: 0.7436 - val_loss: 1.0900 - val_acc: 0.7500
    Epoch 191/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0937 - acc: 0.7453 - val_loss: 1.0884 - val_acc: 0.7480
    Epoch 192/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0915 - acc: 0.7431 - val_loss: 1.0847 - val_acc: 0.7490
    Epoch 193/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0896 - acc: 0.7453 - val_loss: 1.0851 - val_acc: 0.7480
    Epoch 194/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0878 - acc: 0.7461 - val_loss: 1.0794 - val_acc: 0.7510
    Epoch 195/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0858 - acc: 0.7444 - val_loss: 1.0851 - val_acc: 0.7490
    Epoch 196/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0839 - acc: 0.7449 - val_loss: 1.0758 - val_acc: 0.7520
    Epoch 197/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0816 - acc: 0.7459 - val_loss: 1.0757 - val_acc: 0.7480
    Epoch 198/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0805 - acc: 0.7459 - val_loss: 1.0711 - val_acc: 0.7520
    Epoch 199/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0782 - acc: 0.7463 - val_loss: 1.0737 - val_acc: 0.7500
    Epoch 200/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0765 - acc: 0.7461 - val_loss: 1.0855 - val_acc: 0.7440
    Epoch 201/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0759 - acc: 0.7460 - val_loss: 1.0684 - val_acc: 0.7490
    Epoch 202/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0739 - acc: 0.7449 - val_loss: 1.0654 - val_acc: 0.7520
    Epoch 203/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0714 - acc: 0.7460 - val_loss: 1.0711 - val_acc: 0.7470
    Epoch 204/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0696 - acc: 0.7503 - val_loss: 1.0688 - val_acc: 0.7470
    Epoch 205/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0681 - acc: 0.7467 - val_loss: 1.0636 - val_acc: 0.7540
    Epoch 206/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0670 - acc: 0.7461 - val_loss: 1.0600 - val_acc: 0.7490
    Epoch 207/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0649 - acc: 0.7473 - val_loss: 1.0572 - val_acc: 0.7520
    Epoch 208/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0634 - acc: 0.7479 - val_loss: 1.0567 - val_acc: 0.7510
    Epoch 209/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0619 - acc: 0.7491 - val_loss: 1.0575 - val_acc: 0.7530
    Epoch 210/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0603 - acc: 0.7484 - val_loss: 1.0538 - val_acc: 0.7530
    Epoch 211/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0589 - acc: 0.7489 - val_loss: 1.0652 - val_acc: 0.7480
    Epoch 212/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0573 - acc: 0.7476 - val_loss: 1.0600 - val_acc: 0.7450
    Epoch 213/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0557 - acc: 0.7489 - val_loss: 1.0511 - val_acc: 0.7520
    Epoch 214/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0545 - acc: 0.7480 - val_loss: 1.0473 - val_acc: 0.7530
    Epoch 215/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0523 - acc: 0.7483 - val_loss: 1.0483 - val_acc: 0.7540
    Epoch 216/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0514 - acc: 0.7500 - val_loss: 1.0438 - val_acc: 0.7520
    Epoch 217/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0493 - acc: 0.7501 - val_loss: 1.0430 - val_acc: 0.7530
    Epoch 218/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0480 - acc: 0.7504 - val_loss: 1.0422 - val_acc: 0.7600
    Epoch 219/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0460 - acc: 0.7493 - val_loss: 1.0397 - val_acc: 0.7600
    Epoch 220/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0455 - acc: 0.7507 - val_loss: 1.0403 - val_acc: 0.7550
    Epoch 221/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0435 - acc: 0.7496 - val_loss: 1.0394 - val_acc: 0.7520
    Epoch 222/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0420 - acc: 0.7523 - val_loss: 1.0384 - val_acc: 0.7540
    Epoch 223/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0417 - acc: 0.7489 - val_loss: 1.0436 - val_acc: 0.7480
    Epoch 224/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0402 - acc: 0.7507 - val_loss: 1.0336 - val_acc: 0.7550
    Epoch 225/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0377 - acc: 0.7507 - val_loss: 1.0374 - val_acc: 0.7530
    Epoch 226/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0377 - acc: 0.7496 - val_loss: 1.0327 - val_acc: 0.7550
    Epoch 227/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0360 - acc: 0.7513 - val_loss: 1.0290 - val_acc: 0.7530
    Epoch 228/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0337 - acc: 0.7513 - val_loss: 1.0371 - val_acc: 0.7540
    Epoch 229/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 1.0329 - acc: 0.7523 - val_loss: 1.0270 - val_acc: 0.7550
    Epoch 230/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0317 - acc: 0.7521 - val_loss: 1.0322 - val_acc: 0.7540
    Epoch 231/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 1.0303 - acc: 0.7525 - val_loss: 1.0251 - val_acc: 0.7560
    Epoch 232/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0290 - acc: 0.7536 - val_loss: 1.0287 - val_acc: 0.7580
    Epoch 233/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0275 - acc: 0.7508 - val_loss: 1.0264 - val_acc: 0.7560
    Epoch 234/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0265 - acc: 0.7515 - val_loss: 1.0202 - val_acc: 0.7560
    Epoch 235/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0255 - acc: 0.7537 - val_loss: 1.0272 - val_acc: 0.7570
    Epoch 236/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0243 - acc: 0.7528 - val_loss: 1.0214 - val_acc: 0.7580
    Epoch 237/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0230 - acc: 0.7536 - val_loss: 1.0391 - val_acc: 0.7470
    Epoch 238/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0215 - acc: 0.7547 - val_loss: 1.0196 - val_acc: 0.7570
    Epoch 239/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0198 - acc: 0.7532 - val_loss: 1.0173 - val_acc: 0.7560
    Epoch 240/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0193 - acc: 0.7533 - val_loss: 1.0183 - val_acc: 0.7550
    Epoch 241/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0178 - acc: 0.7544 - val_loss: 1.0149 - val_acc: 0.7560
    Epoch 242/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0165 - acc: 0.7539 - val_loss: 1.0228 - val_acc: 0.7560
    Epoch 243/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0159 - acc: 0.7527 - val_loss: 1.0305 - val_acc: 0.7500
    Epoch 244/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0149 - acc: 0.7539 - val_loss: 1.0104 - val_acc: 0.7580
    Epoch 245/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0137 - acc: 0.7545 - val_loss: 1.0124 - val_acc: 0.7610
    Epoch 246/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0125 - acc: 0.7567 - val_loss: 1.0154 - val_acc: 0.7560
    Epoch 247/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0108 - acc: 0.7533 - val_loss: 1.0093 - val_acc: 0.7580
    Epoch 248/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0102 - acc: 0.7564 - val_loss: 1.0059 - val_acc: 0.7620
    Epoch 249/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0087 - acc: 0.7561 - val_loss: 1.0064 - val_acc: 0.7640
    Epoch 250/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0075 - acc: 0.7559 - val_loss: 1.0074 - val_acc: 0.7610
    Epoch 251/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0065 - acc: 0.7544 - val_loss: 1.0026 - val_acc: 0.7590
    Epoch 252/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0049 - acc: 0.7565 - val_loss: 1.0053 - val_acc: 0.7630
    Epoch 253/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0037 - acc: 0.7552 - val_loss: 1.0023 - val_acc: 0.7610
    Epoch 254/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0033 - acc: 0.7571 - val_loss: 1.0067 - val_acc: 0.7530
    Epoch 255/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0027 - acc: 0.7568 - val_loss: 1.0027 - val_acc: 0.7620
    Epoch 256/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 1.0013 - acc: 0.7583 - val_loss: 1.0114 - val_acc: 0.7560
    Epoch 257/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 1.0005 - acc: 0.7559 - val_loss: 0.9976 - val_acc: 0.7630
    Epoch 258/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9993 - acc: 0.7568 - val_loss: 0.9981 - val_acc: 0.7620
    Epoch 259/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9981 - acc: 0.7585 - val_loss: 0.9982 - val_acc: 0.7600
    Epoch 260/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9966 - acc: 0.7585 - val_loss: 0.9961 - val_acc: 0.7640
    Epoch 261/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9951 - acc: 0.7572 - val_loss: 0.9945 - val_acc: 0.7620
    Epoch 262/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9944 - acc: 0.7587 - val_loss: 0.9972 - val_acc: 0.7630
    Epoch 263/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9936 - acc: 0.7583 - val_loss: 1.0038 - val_acc: 0.7530
    Epoch 264/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9937 - acc: 0.7579 - val_loss: 0.9954 - val_acc: 0.7600
    Epoch 265/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9921 - acc: 0.7599 - val_loss: 0.9925 - val_acc: 0.7640
    Epoch 266/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9904 - acc: 0.7587 - val_loss: 0.9892 - val_acc: 0.7630
    Epoch 267/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9890 - acc: 0.7589 - val_loss: 0.9886 - val_acc: 0.7630
    Epoch 268/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9886 - acc: 0.7593 - val_loss: 0.9899 - val_acc: 0.7630
    Epoch 269/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9870 - acc: 0.7591 - val_loss: 0.9884 - val_acc: 0.7600
    Epoch 270/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9873 - acc: 0.7567 - val_loss: 0.9869 - val_acc: 0.7620
    Epoch 271/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9859 - acc: 0.7608 - val_loss: 0.9889 - val_acc: 0.7580
    Epoch 272/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9854 - acc: 0.7591 - val_loss: 0.9876 - val_acc: 0.7600
    Epoch 273/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9840 - acc: 0.7593 - val_loss: 0.9839 - val_acc: 0.7630
    Epoch 274/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9829 - acc: 0.7587 - val_loss: 0.9840 - val_acc: 0.7590
    Epoch 275/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9826 - acc: 0.7600 - val_loss: 0.9834 - val_acc: 0.7600
    Epoch 276/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9815 - acc: 0.7576 - val_loss: 0.9802 - val_acc: 0.7590
    Epoch 277/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9800 - acc: 0.7596 - val_loss: 0.9797 - val_acc: 0.7650
    Epoch 278/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9797 - acc: 0.7609 - val_loss: 0.9843 - val_acc: 0.7680
    Epoch 279/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9787 - acc: 0.7607 - val_loss: 0.9786 - val_acc: 0.7660
    Epoch 280/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9775 - acc: 0.7607 - val_loss: 0.9792 - val_acc: 0.7580
    Epoch 281/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9768 - acc: 0.7607 - val_loss: 0.9832 - val_acc: 0.7650
    Epoch 282/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9766 - acc: 0.7604 - val_loss: 0.9901 - val_acc: 0.7560
    Epoch 283/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9757 - acc: 0.7619 - val_loss: 0.9756 - val_acc: 0.7670
    Epoch 284/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9741 - acc: 0.7616 - val_loss: 0.9787 - val_acc: 0.7580
    Epoch 285/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9742 - acc: 0.7616 - val_loss: 0.9818 - val_acc: 0.7620
    Epoch 286/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9728 - acc: 0.7613 - val_loss: 0.9741 - val_acc: 0.7630
    Epoch 287/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9717 - acc: 0.7620 - val_loss: 0.9887 - val_acc: 0.7570
    Epoch 288/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9717 - acc: 0.7621 - val_loss: 0.9731 - val_acc: 0.7620
    Epoch 289/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9699 - acc: 0.7619 - val_loss: 0.9739 - val_acc: 0.7640
    Epoch 290/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9694 - acc: 0.7629 - val_loss: 0.9686 - val_acc: 0.7610
    Epoch 291/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9685 - acc: 0.7625 - val_loss: 0.9755 - val_acc: 0.7640
    Epoch 292/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9682 - acc: 0.7611 - val_loss: 0.9714 - val_acc: 0.7620
    Epoch 293/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9677 - acc: 0.7631 - val_loss: 0.9687 - val_acc: 0.7690
    Epoch 294/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9665 - acc: 0.7613 - val_loss: 0.9722 - val_acc: 0.7620
    Epoch 295/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9654 - acc: 0.7612 - val_loss: 0.9918 - val_acc: 0.7530
    Epoch 296/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9660 - acc: 0.7624 - val_loss: 0.9679 - val_acc: 0.7620
    Epoch 297/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9638 - acc: 0.7637 - val_loss: 0.9671 - val_acc: 0.7600
    Epoch 298/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9632 - acc: 0.7616 - val_loss: 0.9697 - val_acc: 0.7610
    Epoch 299/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9638 - acc: 0.7624 - val_loss: 0.9655 - val_acc: 0.7620
    Epoch 300/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9618 - acc: 0.7628 - val_loss: 0.9784 - val_acc: 0.7590
    Epoch 301/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9608 - acc: 0.7652 - val_loss: 0.9702 - val_acc: 0.7570
    Epoch 302/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9606 - acc: 0.7624 - val_loss: 0.9627 - val_acc: 0.7640
    Epoch 303/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9595 - acc: 0.7637 - val_loss: 0.9616 - val_acc: 0.7650
    Epoch 304/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9591 - acc: 0.7645 - val_loss: 0.9591 - val_acc: 0.7630
    Epoch 305/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9572 - acc: 0.7644 - val_loss: 0.9633 - val_acc: 0.7670
    Epoch 306/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9575 - acc: 0.7647 - val_loss: 0.9641 - val_acc: 0.7630
    Epoch 307/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9565 - acc: 0.7636 - val_loss: 0.9604 - val_acc: 0.7640
    Epoch 308/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9565 - acc: 0.7641 - val_loss: 0.9600 - val_acc: 0.7600
    Epoch 309/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9554 - acc: 0.7640 - val_loss: 0.9574 - val_acc: 0.7670
    Epoch 310/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9545 - acc: 0.7636 - val_loss: 0.9611 - val_acc: 0.7670
    Epoch 311/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9545 - acc: 0.7653 - val_loss: 0.9637 - val_acc: 0.7640
    Epoch 312/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9536 - acc: 0.7661 - val_loss: 0.9570 - val_acc: 0.7570
    Epoch 313/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9528 - acc: 0.7641 - val_loss: 0.9556 - val_acc: 0.7610
    Epoch 314/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9524 - acc: 0.7637 - val_loss: 0.9629 - val_acc: 0.7670
    Epoch 315/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9520 - acc: 0.7648 - val_loss: 0.9568 - val_acc: 0.7620
    Epoch 316/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9514 - acc: 0.7669 - val_loss: 0.9593 - val_acc: 0.7670
    Epoch 317/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9498 - acc: 0.7648 - val_loss: 0.9557 - val_acc: 0.7680
    Epoch 318/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9497 - acc: 0.7639 - val_loss: 0.9549 - val_acc: 0.7680
    Epoch 319/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9487 - acc: 0.7655 - val_loss: 0.9528 - val_acc: 0.7680
    Epoch 320/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9480 - acc: 0.7663 - val_loss: 0.9532 - val_acc: 0.7670
    Epoch 321/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9478 - acc: 0.7661 - val_loss: 0.9537 - val_acc: 0.7620
    Epoch 322/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9463 - acc: 0.7681 - val_loss: 0.9512 - val_acc: 0.7700
    Epoch 323/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9462 - acc: 0.7669 - val_loss: 0.9529 - val_acc: 0.7640
    Epoch 324/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9459 - acc: 0.7652 - val_loss: 0.9516 - val_acc: 0.7590
    Epoch 325/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9449 - acc: 0.7677 - val_loss: 0.9569 - val_acc: 0.7670
    Epoch 326/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9444 - acc: 0.7668 - val_loss: 0.9519 - val_acc: 0.7650
    Epoch 327/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9443 - acc: 0.7652 - val_loss: 0.9484 - val_acc: 0.7690
    Epoch 328/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9426 - acc: 0.7661 - val_loss: 0.9483 - val_acc: 0.7620
    Epoch 329/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9420 - acc: 0.7669 - val_loss: 0.9483 - val_acc: 0.7680
    Epoch 330/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9417 - acc: 0.7680 - val_loss: 0.9491 - val_acc: 0.7590
    Epoch 331/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9410 - acc: 0.7651 - val_loss: 0.9468 - val_acc: 0.7670
    Epoch 332/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9406 - acc: 0.7676 - val_loss: 0.9497 - val_acc: 0.7610
    Epoch 333/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9412 - acc: 0.7663 - val_loss: 0.9470 - val_acc: 0.7630
    Epoch 334/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9398 - acc: 0.7673 - val_loss: 0.9461 - val_acc: 0.7710
    Epoch 335/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9387 - acc: 0.7683 - val_loss: 0.9489 - val_acc: 0.7670
    Epoch 336/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9384 - acc: 0.7669 - val_loss: 0.9506 - val_acc: 0.7610
    Epoch 337/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9385 - acc: 0.7673 - val_loss: 0.9528 - val_acc: 0.7600
    Epoch 338/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9368 - acc: 0.7673 - val_loss: 0.9437 - val_acc: 0.7660
    Epoch 339/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9372 - acc: 0.7683 - val_loss: 0.9499 - val_acc: 0.7720
    Epoch 340/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9365 - acc: 0.7679 - val_loss: 0.9446 - val_acc: 0.7710
    Epoch 341/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9360 - acc: 0.7688 - val_loss: 0.9415 - val_acc: 0.7660
    Epoch 342/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9360 - acc: 0.7675 - val_loss: 0.9418 - val_acc: 0.7730
    Epoch 343/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9344 - acc: 0.7685 - val_loss: 0.9419 - val_acc: 0.7650
    Epoch 344/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9342 - acc: 0.7667 - val_loss: 0.9490 - val_acc: 0.7620
    Epoch 345/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9341 - acc: 0.7659 - val_loss: 0.9419 - val_acc: 0.7680
    Epoch 346/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9334 - acc: 0.7691 - val_loss: 0.9400 - val_acc: 0.7730
    Epoch 347/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9323 - acc: 0.7679 - val_loss: 0.9443 - val_acc: 0.7690
    Epoch 348/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9323 - acc: 0.7700 - val_loss: 0.9398 - val_acc: 0.7700
    Epoch 349/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9324 - acc: 0.7677 - val_loss: 0.9451 - val_acc: 0.7700
    Epoch 350/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9311 - acc: 0.7688 - val_loss: 0.9441 - val_acc: 0.7680
    Epoch 351/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9303 - acc: 0.7688 - val_loss: 0.9435 - val_acc: 0.7660
    Epoch 352/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9306 - acc: 0.7684 - val_loss: 0.9405 - val_acc: 0.7730
    Epoch 353/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9296 - acc: 0.7691 - val_loss: 0.9402 - val_acc: 0.7720
    Epoch 354/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9284 - acc: 0.7705 - val_loss: 0.9494 - val_acc: 0.7640
    Epoch 355/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9292 - acc: 0.7707 - val_loss: 0.9412 - val_acc: 0.7720
    Epoch 356/1000
    7500/7500 [==============================] - 0s 22us/step - loss: 0.9289 - acc: 0.7715 - val_loss: 0.9393 - val_acc: 0.7720
    Epoch 357/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9278 - acc: 0.7692 - val_loss: 0.9440 - val_acc: 0.7690
    Epoch 358/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9273 - acc: 0.7701 - val_loss: 0.9396 - val_acc: 0.7650
    Epoch 359/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9270 - acc: 0.7685 - val_loss: 0.9472 - val_acc: 0.7700
    Epoch 360/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9270 - acc: 0.7697 - val_loss: 0.9343 - val_acc: 0.7740
    Epoch 361/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9257 - acc: 0.7715 - val_loss: 0.9328 - val_acc: 0.7630
    Epoch 362/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9259 - acc: 0.7701 - val_loss: 0.9351 - val_acc: 0.7740
    Epoch 363/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9246 - acc: 0.7696 - val_loss: 0.9376 - val_acc: 0.7710
    Epoch 364/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9239 - acc: 0.7707 - val_loss: 0.9339 - val_acc: 0.7720
    Epoch 365/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9240 - acc: 0.7699 - val_loss: 0.9311 - val_acc: 0.7700
    Epoch 366/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9239 - acc: 0.7697 - val_loss: 0.9398 - val_acc: 0.7720
    Epoch 367/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9234 - acc: 0.7704 - val_loss: 0.9327 - val_acc: 0.7640
    Epoch 368/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9226 - acc: 0.7721 - val_loss: 0.9372 - val_acc: 0.7720
    Epoch 369/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9223 - acc: 0.7731 - val_loss: 0.9307 - val_acc: 0.7640
    Epoch 370/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9218 - acc: 0.7727 - val_loss: 0.9401 - val_acc: 0.7660
    Epoch 371/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9219 - acc: 0.7715 - val_loss: 0.9396 - val_acc: 0.7620
    Epoch 372/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9216 - acc: 0.7688 - val_loss: 0.9374 - val_acc: 0.7700
    Epoch 373/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9213 - acc: 0.7711 - val_loss: 0.9298 - val_acc: 0.7630
    Epoch 374/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9198 - acc: 0.7715 - val_loss: 0.9295 - val_acc: 0.7670
    Epoch 375/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9198 - acc: 0.7717 - val_loss: 0.9290 - val_acc: 0.7650
    Epoch 376/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9194 - acc: 0.7716 - val_loss: 0.9298 - val_acc: 0.7680
    Epoch 377/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9185 - acc: 0.7711 - val_loss: 0.9410 - val_acc: 0.7670
    Epoch 378/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9179 - acc: 0.7707 - val_loss: 0.9273 - val_acc: 0.7650
    Epoch 379/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9176 - acc: 0.7717 - val_loss: 0.9297 - val_acc: 0.7630
    Epoch 380/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9179 - acc: 0.7712 - val_loss: 0.9287 - val_acc: 0.7680
    Epoch 381/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9187 - acc: 0.7695 - val_loss: 0.9354 - val_acc: 0.7740
    Epoch 382/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9173 - acc: 0.7728 - val_loss: 0.9331 - val_acc: 0.7730
    Epoch 383/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9172 - acc: 0.7707 - val_loss: 0.9320 - val_acc: 0.7710
    Epoch 384/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9167 - acc: 0.7724 - val_loss: 0.9331 - val_acc: 0.7650
    Epoch 385/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9166 - acc: 0.7716 - val_loss: 0.9301 - val_acc: 0.7710
    Epoch 386/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9158 - acc: 0.7721 - val_loss: 0.9262 - val_acc: 0.7710
    Epoch 387/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9150 - acc: 0.7704 - val_loss: 0.9355 - val_acc: 0.7700
    Epoch 388/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9148 - acc: 0.7724 - val_loss: 0.9270 - val_acc: 0.7660
    Epoch 389/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9142 - acc: 0.7729 - val_loss: 0.9323 - val_acc: 0.7700
    Epoch 390/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9143 - acc: 0.7733 - val_loss: 0.9344 - val_acc: 0.7660
    Epoch 391/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9143 - acc: 0.7724 - val_loss: 0.9292 - val_acc: 0.7720
    Epoch 392/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9130 - acc: 0.7727 - val_loss: 0.9272 - val_acc: 0.7690
    Epoch 393/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9131 - acc: 0.7733 - val_loss: 0.9253 - val_acc: 0.7740
    Epoch 394/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9120 - acc: 0.7748 - val_loss: 0.9266 - val_acc: 0.7710
    Epoch 395/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9126 - acc: 0.7735 - val_loss: 0.9259 - val_acc: 0.7650
    Epoch 396/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9119 - acc: 0.7735 - val_loss: 0.9309 - val_acc: 0.7660
    Epoch 397/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9113 - acc: 0.7732 - val_loss: 0.9246 - val_acc: 0.7740
    Epoch 398/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9115 - acc: 0.7729 - val_loss: 0.9276 - val_acc: 0.7710
    Epoch 399/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9112 - acc: 0.7736 - val_loss: 0.9248 - val_acc: 0.7710
    Epoch 400/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9101 - acc: 0.7729 - val_loss: 0.9356 - val_acc: 0.7670
    Epoch 401/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9103 - acc: 0.7721 - val_loss: 0.9216 - val_acc: 0.7660
    Epoch 402/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9089 - acc: 0.7767 - val_loss: 0.9225 - val_acc: 0.7720
    Epoch 403/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9092 - acc: 0.7727 - val_loss: 0.9223 - val_acc: 0.7680
    Epoch 404/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9102 - acc: 0.7737 - val_loss: 0.9224 - val_acc: 0.7710
    Epoch 405/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9094 - acc: 0.7741 - val_loss: 0.9239 - val_acc: 0.7750
    Epoch 406/1000
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9088 - acc: 0.7724 - val_loss: 0.9224 - val_acc: 0.7750
    Epoch 407/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9085 - acc: 0.7749 - val_loss: 0.9225 - val_acc: 0.7710
    Epoch 408/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9076 - acc: 0.7743 - val_loss: 0.9298 - val_acc: 0.7680
    Epoch 409/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9072 - acc: 0.7748 - val_loss: 0.9198 - val_acc: 0.7710
    Epoch 410/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9072 - acc: 0.7744 - val_loss: 0.9191 - val_acc: 0.7670
    Epoch 411/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9070 - acc: 0.7752 - val_loss: 0.9188 - val_acc: 0.7720
    Epoch 412/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9065 - acc: 0.7747 - val_loss: 0.9205 - val_acc: 0.7730
    Epoch 413/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9058 - acc: 0.7773 - val_loss: 0.9232 - val_acc: 0.7740
    Epoch 414/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9056 - acc: 0.7747 - val_loss: 0.9287 - val_acc: 0.7720
    Epoch 415/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9065 - acc: 0.7763 - val_loss: 0.9195 - val_acc: 0.7710
    Epoch 416/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9052 - acc: 0.7752 - val_loss: 0.9202 - val_acc: 0.7700
    Epoch 417/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9049 - acc: 0.7751 - val_loss: 0.9269 - val_acc: 0.7700
    Epoch 418/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9047 - acc: 0.7736 - val_loss: 0.9200 - val_acc: 0.7760
    Epoch 419/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9041 - acc: 0.7752 - val_loss: 0.9197 - val_acc: 0.7700
    Epoch 420/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9041 - acc: 0.7757 - val_loss: 0.9174 - val_acc: 0.7690
    Epoch 421/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9032 - acc: 0.7751 - val_loss: 0.9225 - val_acc: 0.7720
    Epoch 422/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9038 - acc: 0.7765 - val_loss: 0.9167 - val_acc: 0.7710
    Epoch 423/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9029 - acc: 0.7755 - val_loss: 0.9268 - val_acc: 0.7710
    Epoch 424/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9038 - acc: 0.7745 - val_loss: 0.9204 - val_acc: 0.7720
    Epoch 425/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9022 - acc: 0.7761 - val_loss: 0.9179 - val_acc: 0.7800
    Epoch 426/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.9025 - acc: 0.7755 - val_loss: 0.9194 - val_acc: 0.7710
    Epoch 427/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9011 - acc: 0.7768 - val_loss: 0.9236 - val_acc: 0.7650
    Epoch 428/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9013 - acc: 0.7759 - val_loss: 0.9173 - val_acc: 0.7720
    Epoch 429/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.9015 - acc: 0.7751 - val_loss: 0.9210 - val_acc: 0.7750
    Epoch 430/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9005 - acc: 0.7776 - val_loss: 0.9269 - val_acc: 0.7630
    Epoch 431/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9010 - acc: 0.7761 - val_loss: 0.9371 - val_acc: 0.7670
    Epoch 432/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.9009 - acc: 0.7763 - val_loss: 0.9224 - val_acc: 0.7720
    Epoch 433/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8995 - acc: 0.7771 - val_loss: 0.9196 - val_acc: 0.7720
    Epoch 434/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.9000 - acc: 0.7763 - val_loss: 0.9126 - val_acc: 0.7790
    Epoch 435/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8989 - acc: 0.7765 - val_loss: 0.9145 - val_acc: 0.7690
    Epoch 436/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8985 - acc: 0.7776 - val_loss: 0.9140 - val_acc: 0.7790
    Epoch 437/1000
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8986 - acc: 0.7772 - val_loss: 0.9158 - val_acc: 0.7740
    Epoch 438/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8995 - acc: 0.7764 - val_loss: 0.9126 - val_acc: 0.7730
    Epoch 439/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8985 - acc: 0.7771 - val_loss: 0.9127 - val_acc: 0.7720
    Epoch 440/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8977 - acc: 0.7784 - val_loss: 0.9107 - val_acc: 0.7730
    Epoch 441/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8977 - acc: 0.7760 - val_loss: 0.9141 - val_acc: 0.7750
    Epoch 442/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8973 - acc: 0.7748 - val_loss: 0.9217 - val_acc: 0.7700
    Epoch 443/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8973 - acc: 0.7771 - val_loss: 0.9186 - val_acc: 0.7730
    Epoch 444/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8969 - acc: 0.7773 - val_loss: 0.9225 - val_acc: 0.7680
    Epoch 445/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8967 - acc: 0.7768 - val_loss: 0.9177 - val_acc: 0.7760
    Epoch 446/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8965 - acc: 0.7772 - val_loss: 0.9176 - val_acc: 0.7790
    Epoch 447/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8953 - acc: 0.7772 - val_loss: 0.9173 - val_acc: 0.7760
    Epoch 448/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8953 - acc: 0.7767 - val_loss: 0.9168 - val_acc: 0.7730
    Epoch 449/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8957 - acc: 0.7781 - val_loss: 0.9148 - val_acc: 0.7750
    Epoch 450/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8952 - acc: 0.7765 - val_loss: 0.9094 - val_acc: 0.7750
    Epoch 451/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8950 - acc: 0.7771 - val_loss: 0.9173 - val_acc: 0.7740
    Epoch 452/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8943 - acc: 0.7764 - val_loss: 0.9281 - val_acc: 0.7660
    Epoch 453/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8944 - acc: 0.7772 - val_loss: 0.9163 - val_acc: 0.7730
    Epoch 454/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8947 - acc: 0.7775 - val_loss: 0.9112 - val_acc: 0.7780
    Epoch 455/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8942 - acc: 0.7773 - val_loss: 0.9125 - val_acc: 0.7760
    Epoch 456/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8934 - acc: 0.7777 - val_loss: 0.9241 - val_acc: 0.7640
    Epoch 457/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8936 - acc: 0.7764 - val_loss: 0.9086 - val_acc: 0.7750
    Epoch 458/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8928 - acc: 0.7783 - val_loss: 0.9235 - val_acc: 0.7700
    Epoch 459/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8935 - acc: 0.7776 - val_loss: 0.9136 - val_acc: 0.7700
    Epoch 460/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8922 - acc: 0.7777 - val_loss: 0.9134 - val_acc: 0.7770
    Epoch 461/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8926 - acc: 0.7772 - val_loss: 0.9134 - val_acc: 0.7750
    Epoch 462/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8919 - acc: 0.7781 - val_loss: 0.9088 - val_acc: 0.7810
    Epoch 463/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8915 - acc: 0.7772 - val_loss: 0.9135 - val_acc: 0.7730
    Epoch 464/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8925 - acc: 0.7787 - val_loss: 0.9088 - val_acc: 0.7710
    Epoch 465/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8914 - acc: 0.7791 - val_loss: 0.9067 - val_acc: 0.7780
    Epoch 466/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8906 - acc: 0.7803 - val_loss: 0.9100 - val_acc: 0.7760
    Epoch 467/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8902 - acc: 0.7791 - val_loss: 0.9154 - val_acc: 0.7740
    Epoch 468/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8902 - acc: 0.7793 - val_loss: 0.9134 - val_acc: 0.7770
    Epoch 469/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8902 - acc: 0.7784 - val_loss: 0.9084 - val_acc: 0.7770
    Epoch 470/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8897 - acc: 0.7785 - val_loss: 0.9109 - val_acc: 0.7780
    Epoch 471/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8898 - acc: 0.7768 - val_loss: 0.9062 - val_acc: 0.7780
    Epoch 472/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8892 - acc: 0.7776 - val_loss: 0.9105 - val_acc: 0.7760
    Epoch 473/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8892 - acc: 0.7789 - val_loss: 0.9085 - val_acc: 0.7750
    Epoch 474/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8891 - acc: 0.7773 - val_loss: 0.9071 - val_acc: 0.7810
    Epoch 475/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8890 - acc: 0.7784 - val_loss: 0.9104 - val_acc: 0.7740
    Epoch 476/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8885 - acc: 0.7791 - val_loss: 0.9128 - val_acc: 0.7740
    Epoch 477/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8873 - acc: 0.7775 - val_loss: 0.9111 - val_acc: 0.7770
    Epoch 478/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8883 - acc: 0.7800 - val_loss: 0.9066 - val_acc: 0.7770
    Epoch 479/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8868 - acc: 0.7787 - val_loss: 0.9092 - val_acc: 0.7790
    Epoch 480/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8877 - acc: 0.7801 - val_loss: 0.9089 - val_acc: 0.7780
    Epoch 481/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8874 - acc: 0.7781 - val_loss: 0.9098 - val_acc: 0.7700
    Epoch 482/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8869 - acc: 0.7783 - val_loss: 0.9088 - val_acc: 0.7760
    Epoch 483/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8866 - acc: 0.7789 - val_loss: 0.9089 - val_acc: 0.7730
    Epoch 484/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8872 - acc: 0.7815 - val_loss: 0.9074 - val_acc: 0.7770
    Epoch 485/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8862 - acc: 0.7787 - val_loss: 0.9110 - val_acc: 0.7740
    Epoch 486/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8854 - acc: 0.7809 - val_loss: 0.9125 - val_acc: 0.7770
    Epoch 487/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8856 - acc: 0.7804 - val_loss: 0.9160 - val_acc: 0.7720
    Epoch 488/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8878 - acc: 0.7777 - val_loss: 0.9062 - val_acc: 0.7740
    Epoch 489/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8854 - acc: 0.7791 - val_loss: 0.9137 - val_acc: 0.7790
    Epoch 490/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8855 - acc: 0.7803 - val_loss: 0.9116 - val_acc: 0.7730
    Epoch 491/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8851 - acc: 0.7791 - val_loss: 0.9106 - val_acc: 0.7780
    Epoch 492/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8847 - acc: 0.7811 - val_loss: 0.9119 - val_acc: 0.7720
    Epoch 493/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8840 - acc: 0.7800 - val_loss: 0.9028 - val_acc: 0.7770
    Epoch 494/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8834 - acc: 0.7821 - val_loss: 0.9034 - val_acc: 0.7800
    Epoch 495/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8848 - acc: 0.7791 - val_loss: 0.9047 - val_acc: 0.7810
    Epoch 496/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8841 - acc: 0.7809 - val_loss: 0.9021 - val_acc: 0.7790
    Epoch 497/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8826 - acc: 0.7799 - val_loss: 0.9032 - val_acc: 0.7780
    Epoch 498/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8838 - acc: 0.7813 - val_loss: 0.9172 - val_acc: 0.7660
    Epoch 499/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8843 - acc: 0.7796 - val_loss: 0.9256 - val_acc: 0.7660
    Epoch 500/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8834 - acc: 0.7784 - val_loss: 0.9039 - val_acc: 0.7820
    Epoch 501/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8830 - acc: 0.7812 - val_loss: 0.9023 - val_acc: 0.7770
    Epoch 502/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8822 - acc: 0.7800 - val_loss: 0.9023 - val_acc: 0.7750
    Epoch 503/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8824 - acc: 0.7803 - val_loss: 0.9060 - val_acc: 0.7850
    Epoch 504/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8822 - acc: 0.7803 - val_loss: 0.9045 - val_acc: 0.7780
    Epoch 505/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8824 - acc: 0.7809 - val_loss: 0.9070 - val_acc: 0.7770
    Epoch 506/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8818 - acc: 0.7816 - val_loss: 0.9034 - val_acc: 0.7800
    Epoch 507/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8812 - acc: 0.7805 - val_loss: 0.9056 - val_acc: 0.7750
    Epoch 508/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8828 - acc: 0.7795 - val_loss: 0.9015 - val_acc: 0.7820
    Epoch 509/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8801 - acc: 0.7805 - val_loss: 0.9014 - val_acc: 0.7820
    Epoch 510/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8809 - acc: 0.7813 - val_loss: 0.9034 - val_acc: 0.7810
    Epoch 511/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8803 - acc: 0.7807 - val_loss: 0.9022 - val_acc: 0.7800
    Epoch 512/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8787 - acc: 0.7821 - val_loss: 0.9050 - val_acc: 0.7770
    Epoch 513/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8797 - acc: 0.7801 - val_loss: 0.8998 - val_acc: 0.7770
    Epoch 514/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8797 - acc: 0.7821 - val_loss: 0.9039 - val_acc: 0.7770
    Epoch 515/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8797 - acc: 0.7807 - val_loss: 0.9065 - val_acc: 0.7710
    Epoch 516/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8787 - acc: 0.7817 - val_loss: 0.9016 - val_acc: 0.7800
    Epoch 517/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8788 - acc: 0.7803 - val_loss: 0.9023 - val_acc: 0.7820
    Epoch 518/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8783 - acc: 0.7831 - val_loss: 0.9020 - val_acc: 0.7760
    Epoch 519/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8788 - acc: 0.7813 - val_loss: 0.8999 - val_acc: 0.7850
    Epoch 520/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8791 - acc: 0.7812 - val_loss: 0.8994 - val_acc: 0.7760
    Epoch 521/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8780 - acc: 0.7796 - val_loss: 0.9093 - val_acc: 0.7760
    Epoch 522/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8783 - acc: 0.7804 - val_loss: 0.8990 - val_acc: 0.7760
    Epoch 523/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8773 - acc: 0.7839 - val_loss: 0.9015 - val_acc: 0.7790
    Epoch 524/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8780 - acc: 0.7809 - val_loss: 0.8988 - val_acc: 0.7810
    Epoch 525/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8777 - acc: 0.7815 - val_loss: 0.9027 - val_acc: 0.7810
    Epoch 526/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8777 - acc: 0.7792 - val_loss: 0.9067 - val_acc: 0.7800
    Epoch 527/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8768 - acc: 0.7827 - val_loss: 0.8967 - val_acc: 0.7800
    Epoch 528/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8766 - acc: 0.7819 - val_loss: 0.8988 - val_acc: 0.7830
    Epoch 529/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8764 - acc: 0.7820 - val_loss: 0.9031 - val_acc: 0.7780
    Epoch 530/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8765 - acc: 0.7829 - val_loss: 0.9000 - val_acc: 0.7830
    Epoch 531/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8765 - acc: 0.7816 - val_loss: 0.8981 - val_acc: 0.7790
    Epoch 532/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8759 - acc: 0.7812 - val_loss: 0.9062 - val_acc: 0.7750
    Epoch 533/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8764 - acc: 0.7808 - val_loss: 0.8978 - val_acc: 0.7790
    Epoch 534/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8762 - acc: 0.7807 - val_loss: 0.9009 - val_acc: 0.7800
    Epoch 535/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8757 - acc: 0.7803 - val_loss: 0.9080 - val_acc: 0.7720
    Epoch 536/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8760 - acc: 0.7807 - val_loss: 0.8994 - val_acc: 0.7810
    Epoch 537/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8759 - acc: 0.7817 - val_loss: 0.8997 - val_acc: 0.7800
    Epoch 538/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8753 - acc: 0.7811 - val_loss: 0.9058 - val_acc: 0.7770
    Epoch 539/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8749 - acc: 0.7809 - val_loss: 0.8981 - val_acc: 0.7750
    Epoch 540/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8743 - acc: 0.7817 - val_loss: 0.9028 - val_acc: 0.7870
    Epoch 541/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8754 - acc: 0.7823 - val_loss: 0.9013 - val_acc: 0.7730
    Epoch 542/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8753 - acc: 0.7811 - val_loss: 0.9057 - val_acc: 0.7710
    Epoch 543/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8745 - acc: 0.7823 - val_loss: 0.9219 - val_acc: 0.7660
    Epoch 544/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8751 - acc: 0.7827 - val_loss: 0.9069 - val_acc: 0.7790
    Epoch 545/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8742 - acc: 0.7819 - val_loss: 0.9011 - val_acc: 0.7770
    Epoch 546/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8731 - acc: 0.7837 - val_loss: 0.8947 - val_acc: 0.7810
    Epoch 547/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8728 - acc: 0.7819 - val_loss: 0.9043 - val_acc: 0.7790
    Epoch 548/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8733 - acc: 0.7823 - val_loss: 0.8976 - val_acc: 0.7810
    Epoch 549/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8729 - acc: 0.7828 - val_loss: 0.9009 - val_acc: 0.7780
    Epoch 550/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8736 - acc: 0.7824 - val_loss: 0.9133 - val_acc: 0.7700
    Epoch 551/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8727 - acc: 0.7820 - val_loss: 0.8998 - val_acc: 0.7800
    Epoch 552/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8730 - acc: 0.7833 - val_loss: 0.9000 - val_acc: 0.7740
    Epoch 553/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8722 - acc: 0.7833 - val_loss: 0.8965 - val_acc: 0.7770
    Epoch 554/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8721 - acc: 0.7836 - val_loss: 0.8960 - val_acc: 0.7820
    Epoch 555/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8716 - acc: 0.7832 - val_loss: 0.8994 - val_acc: 0.7800
    Epoch 556/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8721 - acc: 0.7832 - val_loss: 0.8957 - val_acc: 0.7760
    Epoch 557/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8717 - acc: 0.7828 - val_loss: 0.8962 - val_acc: 0.7820
    Epoch 558/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8715 - acc: 0.7819 - val_loss: 0.8948 - val_acc: 0.7820
    Epoch 559/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8709 - acc: 0.7844 - val_loss: 0.8958 - val_acc: 0.7810
    Epoch 560/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8711 - acc: 0.7841 - val_loss: 0.9049 - val_acc: 0.7780
    Epoch 561/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8715 - acc: 0.7839 - val_loss: 0.8975 - val_acc: 0.7790
    Epoch 562/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8709 - acc: 0.7852 - val_loss: 0.8965 - val_acc: 0.7770
    Epoch 563/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8716 - acc: 0.7819 - val_loss: 0.8926 - val_acc: 0.7780
    Epoch 564/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8695 - acc: 0.7829 - val_loss: 0.9166 - val_acc: 0.7640
    Epoch 565/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8704 - acc: 0.7825 - val_loss: 0.9051 - val_acc: 0.7790
    Epoch 566/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8702 - acc: 0.7837 - val_loss: 0.9099 - val_acc: 0.7740
    Epoch 567/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8707 - acc: 0.7857 - val_loss: 0.9061 - val_acc: 0.7720
    Epoch 568/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8696 - acc: 0.7841 - val_loss: 0.9031 - val_acc: 0.7700
    Epoch 569/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8696 - acc: 0.7840 - val_loss: 0.9015 - val_acc: 0.7800
    Epoch 570/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8695 - acc: 0.7847 - val_loss: 0.8986 - val_acc: 0.7800
    Epoch 571/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8685 - acc: 0.7845 - val_loss: 0.8974 - val_acc: 0.7760
    Epoch 572/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8687 - acc: 0.7859 - val_loss: 0.8997 - val_acc: 0.7780
    Epoch 573/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8688 - acc: 0.7832 - val_loss: 0.8989 - val_acc: 0.7760
    Epoch 574/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8679 - acc: 0.7840 - val_loss: 0.8931 - val_acc: 0.7830
    Epoch 575/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8674 - acc: 0.7857 - val_loss: 0.9107 - val_acc: 0.7790
    Epoch 576/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8688 - acc: 0.7841 - val_loss: 0.8937 - val_acc: 0.7810
    Epoch 577/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8673 - acc: 0.7860 - val_loss: 0.8980 - val_acc: 0.7800
    Epoch 578/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8674 - acc: 0.7809 - val_loss: 0.9000 - val_acc: 0.7740
    Epoch 579/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8672 - acc: 0.7832 - val_loss: 0.9121 - val_acc: 0.7680
    Epoch 580/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8676 - acc: 0.7843 - val_loss: 0.8957 - val_acc: 0.7720
    Epoch 581/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8681 - acc: 0.7828 - val_loss: 0.9033 - val_acc: 0.7750
    Epoch 582/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8674 - acc: 0.7833 - val_loss: 0.8942 - val_acc: 0.7850
    Epoch 583/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8671 - acc: 0.7819 - val_loss: 0.8909 - val_acc: 0.7760
    Epoch 584/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8665 - acc: 0.7836 - val_loss: 0.8946 - val_acc: 0.7820
    Epoch 585/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8670 - acc: 0.7831 - val_loss: 0.8958 - val_acc: 0.7770
    Epoch 586/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8651 - acc: 0.7851 - val_loss: 0.8940 - val_acc: 0.7790
    Epoch 587/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8669 - acc: 0.7853 - val_loss: 0.8978 - val_acc: 0.7790
    Epoch 588/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8659 - acc: 0.7847 - val_loss: 0.8908 - val_acc: 0.7870
    Epoch 589/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8658 - acc: 0.7855 - val_loss: 0.8894 - val_acc: 0.7840
    Epoch 590/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8659 - acc: 0.7877 - val_loss: 0.8921 - val_acc: 0.7810
    Epoch 591/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8653 - acc: 0.7856 - val_loss: 0.8913 - val_acc: 0.7820
    Epoch 592/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8645 - acc: 0.7856 - val_loss: 0.8931 - val_acc: 0.7840
    Epoch 593/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8657 - acc: 0.7829 - val_loss: 0.8915 - val_acc: 0.7800
    Epoch 594/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8645 - acc: 0.7836 - val_loss: 0.9091 - val_acc: 0.7750
    Epoch 595/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8644 - acc: 0.7851 - val_loss: 0.8968 - val_acc: 0.7780
    Epoch 596/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8641 - acc: 0.7864 - val_loss: 0.9016 - val_acc: 0.7730
    Epoch 597/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8653 - acc: 0.7840 - val_loss: 0.8950 - val_acc: 0.7780
    Epoch 598/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8640 - acc: 0.7859 - val_loss: 0.9015 - val_acc: 0.7740
    Epoch 599/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8643 - acc: 0.7841 - val_loss: 0.9000 - val_acc: 0.7780
    Epoch 600/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8634 - acc: 0.7863 - val_loss: 0.8929 - val_acc: 0.7800
    Epoch 601/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8632 - acc: 0.7849 - val_loss: 0.8987 - val_acc: 0.7790
    Epoch 602/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8631 - acc: 0.7860 - val_loss: 0.9312 - val_acc: 0.7720
    Epoch 603/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8657 - acc: 0.7833 - val_loss: 0.8910 - val_acc: 0.7830
    Epoch 604/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8633 - acc: 0.7857 - val_loss: 0.8905 - val_acc: 0.7800
    Epoch 605/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8637 - acc: 0.7836 - val_loss: 0.8922 - val_acc: 0.7800
    Epoch 606/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8623 - acc: 0.7851 - val_loss: 0.8882 - val_acc: 0.7850
    Epoch 607/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8614 - acc: 0.7863 - val_loss: 0.9030 - val_acc: 0.7690
    Epoch 608/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8629 - acc: 0.7871 - val_loss: 0.8911 - val_acc: 0.7790
    Epoch 609/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8617 - acc: 0.7845 - val_loss: 0.8902 - val_acc: 0.7800
    Epoch 610/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8628 - acc: 0.7871 - val_loss: 0.9009 - val_acc: 0.7730
    Epoch 611/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8629 - acc: 0.7857 - val_loss: 0.8952 - val_acc: 0.7730
    Epoch 612/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8606 - acc: 0.7889 - val_loss: 0.8938 - val_acc: 0.7740
    Epoch 613/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8619 - acc: 0.7845 - val_loss: 0.8887 - val_acc: 0.7780
    Epoch 614/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8616 - acc: 0.7863 - val_loss: 0.8933 - val_acc: 0.7830
    Epoch 615/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8609 - acc: 0.7871 - val_loss: 0.8915 - val_acc: 0.7850
    Epoch 616/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8620 - acc: 0.7859 - val_loss: 0.8892 - val_acc: 0.7820
    Epoch 617/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8615 - acc: 0.7852 - val_loss: 0.8955 - val_acc: 0.7830
    Epoch 618/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8623 - acc: 0.7855 - val_loss: 0.8973 - val_acc: 0.7770
    Epoch 619/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8614 - acc: 0.7856 - val_loss: 0.8917 - val_acc: 0.7840
    Epoch 620/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8603 - acc: 0.7837 - val_loss: 0.8890 - val_acc: 0.7810
    Epoch 621/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8601 - acc: 0.7873 - val_loss: 0.8904 - val_acc: 0.7840
    Epoch 622/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8607 - acc: 0.7876 - val_loss: 0.8904 - val_acc: 0.7840
    Epoch 623/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8588 - acc: 0.7872 - val_loss: 0.8983 - val_acc: 0.7820
    Epoch 624/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8608 - acc: 0.7869 - val_loss: 0.9171 - val_acc: 0.7730
    Epoch 625/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8610 - acc: 0.7864 - val_loss: 0.8994 - val_acc: 0.7790
    Epoch 626/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8605 - acc: 0.7871 - val_loss: 0.8894 - val_acc: 0.7750
    Epoch 627/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8601 - acc: 0.7868 - val_loss: 0.8948 - val_acc: 0.7750
    Epoch 628/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8593 - acc: 0.7891 - val_loss: 0.8867 - val_acc: 0.7850
    Epoch 629/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8596 - acc: 0.7864 - val_loss: 0.8927 - val_acc: 0.7800
    Epoch 630/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8605 - acc: 0.7852 - val_loss: 0.9066 - val_acc: 0.7750
    Epoch 631/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8609 - acc: 0.7868 - val_loss: 0.8944 - val_acc: 0.7700
    Epoch 632/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8582 - acc: 0.7873 - val_loss: 0.8936 - val_acc: 0.7780
    Epoch 633/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8579 - acc: 0.7881 - val_loss: 0.8971 - val_acc: 0.7800
    Epoch 634/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8594 - acc: 0.7859 - val_loss: 0.8901 - val_acc: 0.7800
    Epoch 635/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8585 - acc: 0.7884 - val_loss: 0.8936 - val_acc: 0.7800
    Epoch 636/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8578 - acc: 0.7859 - val_loss: 0.8887 - val_acc: 0.7810
    Epoch 637/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8576 - acc: 0.7860 - val_loss: 0.8881 - val_acc: 0.7790
    Epoch 638/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8575 - acc: 0.7887 - val_loss: 0.9072 - val_acc: 0.7710
    Epoch 639/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8570 - acc: 0.7867 - val_loss: 0.8917 - val_acc: 0.7780
    Epoch 640/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8567 - acc: 0.7881 - val_loss: 0.8924 - val_acc: 0.7780
    Epoch 641/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8570 - acc: 0.7879 - val_loss: 0.8887 - val_acc: 0.7810
    Epoch 642/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8572 - acc: 0.7871 - val_loss: 0.8880 - val_acc: 0.7830
    Epoch 643/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8567 - acc: 0.7896 - val_loss: 0.8987 - val_acc: 0.7750
    Epoch 644/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8567 - acc: 0.7872 - val_loss: 0.8876 - val_acc: 0.7850
    Epoch 645/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8554 - acc: 0.7884 - val_loss: 0.8857 - val_acc: 0.7830
    Epoch 646/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8559 - acc: 0.7863 - val_loss: 0.8885 - val_acc: 0.7840
    Epoch 647/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8561 - acc: 0.7865 - val_loss: 0.8869 - val_acc: 0.7800
    Epoch 648/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8557 - acc: 0.7869 - val_loss: 0.9200 - val_acc: 0.7700
    Epoch 649/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8566 - acc: 0.7884 - val_loss: 0.8865 - val_acc: 0.7820
    Epoch 650/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8545 - acc: 0.7891 - val_loss: 0.8906 - val_acc: 0.7780
    Epoch 651/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8557 - acc: 0.7880 - val_loss: 0.8919 - val_acc: 0.7760
    Epoch 652/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8549 - acc: 0.7873 - val_loss: 0.8920 - val_acc: 0.7740
    Epoch 653/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8556 - acc: 0.7892 - val_loss: 0.8908 - val_acc: 0.7760
    Epoch 654/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8550 - acc: 0.7877 - val_loss: 0.8853 - val_acc: 0.7820
    Epoch 655/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8544 - acc: 0.7884 - val_loss: 0.8944 - val_acc: 0.7690
    Epoch 656/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8543 - acc: 0.7885 - val_loss: 0.8871 - val_acc: 0.7830
    Epoch 657/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8539 - acc: 0.7872 - val_loss: 0.8897 - val_acc: 0.7800
    Epoch 658/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8540 - acc: 0.7900 - val_loss: 0.8988 - val_acc: 0.7670
    Epoch 659/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8555 - acc: 0.7887 - val_loss: 0.9005 - val_acc: 0.7770
    Epoch 660/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8540 - acc: 0.7883 - val_loss: 0.8928 - val_acc: 0.7810
    Epoch 661/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8544 - acc: 0.7877 - val_loss: 0.8958 - val_acc: 0.7790
    Epoch 662/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8541 - acc: 0.7884 - val_loss: 0.8884 - val_acc: 0.7830
    Epoch 663/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8518 - acc: 0.7875 - val_loss: 0.8926 - val_acc: 0.7810
    Epoch 664/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8531 - acc: 0.7901 - val_loss: 0.8930 - val_acc: 0.7750
    Epoch 665/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8522 - acc: 0.7919 - val_loss: 0.8877 - val_acc: 0.7810
    Epoch 666/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8537 - acc: 0.7887 - val_loss: 0.8865 - val_acc: 0.7830
    Epoch 667/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8529 - acc: 0.7885 - val_loss: 0.8990 - val_acc: 0.7760
    Epoch 668/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8541 - acc: 0.7883 - val_loss: 0.8852 - val_acc: 0.7810
    Epoch 669/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8532 - acc: 0.7900 - val_loss: 0.8875 - val_acc: 0.7800
    Epoch 670/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8538 - acc: 0.7887 - val_loss: 0.8908 - val_acc: 0.7890
    Epoch 671/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8525 - acc: 0.7915 - val_loss: 0.8951 - val_acc: 0.7810
    Epoch 672/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8522 - acc: 0.7897 - val_loss: 0.8871 - val_acc: 0.7730
    Epoch 673/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8519 - acc: 0.7887 - val_loss: 0.8877 - val_acc: 0.7850
    Epoch 674/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8526 - acc: 0.7895 - val_loss: 0.8916 - val_acc: 0.7800
    Epoch 675/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8539 - acc: 0.7875 - val_loss: 0.8922 - val_acc: 0.7770
    Epoch 676/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8516 - acc: 0.7908 - val_loss: 0.8895 - val_acc: 0.7780
    Epoch 677/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8522 - acc: 0.7919 - val_loss: 0.8866 - val_acc: 0.7850
    Epoch 678/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8513 - acc: 0.7889 - val_loss: 0.8893 - val_acc: 0.7790
    Epoch 679/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8512 - acc: 0.7905 - val_loss: 0.8874 - val_acc: 0.7780
    Epoch 680/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8513 - acc: 0.7895 - val_loss: 0.8935 - val_acc: 0.7750
    Epoch 681/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8511 - acc: 0.7888 - val_loss: 0.8870 - val_acc: 0.7810
    Epoch 682/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8499 - acc: 0.7900 - val_loss: 0.8960 - val_acc: 0.7720
    Epoch 683/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8512 - acc: 0.7891 - val_loss: 0.8877 - val_acc: 0.7790
    Epoch 684/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8506 - acc: 0.7905 - val_loss: 0.8858 - val_acc: 0.7810
    Epoch 685/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8498 - acc: 0.7911 - val_loss: 0.8863 - val_acc: 0.7800
    Epoch 686/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8500 - acc: 0.7911 - val_loss: 0.8860 - val_acc: 0.7820
    Epoch 687/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8509 - acc: 0.7916 - val_loss: 0.8931 - val_acc: 0.7810
    Epoch 688/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8504 - acc: 0.7905 - val_loss: 0.8849 - val_acc: 0.7830
    Epoch 689/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8496 - acc: 0.7899 - val_loss: 0.8827 - val_acc: 0.7830
    Epoch 690/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8483 - acc: 0.7908 - val_loss: 0.8867 - val_acc: 0.7690
    Epoch 691/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8492 - acc: 0.7889 - val_loss: 0.8939 - val_acc: 0.7730
    Epoch 692/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8497 - acc: 0.7913 - val_loss: 0.8981 - val_acc: 0.7790
    Epoch 693/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8504 - acc: 0.7892 - val_loss: 0.8831 - val_acc: 0.7840
    Epoch 694/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8483 - acc: 0.7905 - val_loss: 0.8849 - val_acc: 0.7840
    Epoch 695/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8484 - acc: 0.7908 - val_loss: 0.9029 - val_acc: 0.7790
    Epoch 696/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8501 - acc: 0.7900 - val_loss: 0.8844 - val_acc: 0.7810
    Epoch 697/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8495 - acc: 0.7901 - val_loss: 0.8915 - val_acc: 0.7800
    Epoch 698/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8481 - acc: 0.7905 - val_loss: 0.8845 - val_acc: 0.7750
    Epoch 699/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8478 - acc: 0.7909 - val_loss: 0.8907 - val_acc: 0.7800
    Epoch 700/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8471 - acc: 0.7901 - val_loss: 0.8913 - val_acc: 0.7720
    Epoch 701/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8490 - acc: 0.7896 - val_loss: 0.8884 - val_acc: 0.7820
    Epoch 702/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8485 - acc: 0.7912 - val_loss: 0.9003 - val_acc: 0.7720
    Epoch 703/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8498 - acc: 0.7925 - val_loss: 0.8983 - val_acc: 0.7740
    Epoch 704/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8490 - acc: 0.7904 - val_loss: 0.8937 - val_acc: 0.7770
    Epoch 705/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8477 - acc: 0.7931 - val_loss: 0.8830 - val_acc: 0.7830
    Epoch 706/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8463 - acc: 0.7915 - val_loss: 0.9006 - val_acc: 0.7740
    Epoch 707/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8479 - acc: 0.7901 - val_loss: 0.8928 - val_acc: 0.7740
    Epoch 708/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8473 - acc: 0.7919 - val_loss: 0.9070 - val_acc: 0.7720
    Epoch 709/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8472 - acc: 0.7923 - val_loss: 0.8840 - val_acc: 0.7820
    Epoch 710/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8462 - acc: 0.7931 - val_loss: 0.8816 - val_acc: 0.7820
    Epoch 711/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8466 - acc: 0.7916 - val_loss: 0.8936 - val_acc: 0.7770
    Epoch 712/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8466 - acc: 0.7932 - val_loss: 0.8888 - val_acc: 0.7820
    Epoch 713/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8460 - acc: 0.7907 - val_loss: 0.8844 - val_acc: 0.7860
    Epoch 714/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8462 - acc: 0.7925 - val_loss: 0.8843 - val_acc: 0.7820
    Epoch 715/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8469 - acc: 0.7907 - val_loss: 0.8826 - val_acc: 0.7820
    Epoch 716/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8440 - acc: 0.7921 - val_loss: 0.9187 - val_acc: 0.7500
    Epoch 717/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8468 - acc: 0.7901 - val_loss: 0.8854 - val_acc: 0.7810
    Epoch 718/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8462 - acc: 0.7939 - val_loss: 0.8925 - val_acc: 0.7750
    Epoch 719/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8450 - acc: 0.7909 - val_loss: 0.9043 - val_acc: 0.7780
    Epoch 720/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8468 - acc: 0.7936 - val_loss: 0.8837 - val_acc: 0.7840
    Epoch 721/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8453 - acc: 0.7931 - val_loss: 0.8846 - val_acc: 0.7840
    Epoch 722/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8448 - acc: 0.7913 - val_loss: 0.8825 - val_acc: 0.7820
    Epoch 723/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8457 - acc: 0.7904 - val_loss: 0.8873 - val_acc: 0.7860
    Epoch 724/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8462 - acc: 0.7919 - val_loss: 0.9013 - val_acc: 0.7790
    Epoch 725/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8446 - acc: 0.7892 - val_loss: 0.9126 - val_acc: 0.7640
    Epoch 726/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8454 - acc: 0.7937 - val_loss: 0.8897 - val_acc: 0.7850
    Epoch 727/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8444 - acc: 0.7928 - val_loss: 0.8855 - val_acc: 0.7840
    Epoch 728/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8440 - acc: 0.7911 - val_loss: 0.8895 - val_acc: 0.7770
    Epoch 729/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8438 - acc: 0.7924 - val_loss: 0.8973 - val_acc: 0.7810
    Epoch 730/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8455 - acc: 0.7903 - val_loss: 0.8833 - val_acc: 0.7870
    Epoch 731/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8434 - acc: 0.7923 - val_loss: 0.8846 - val_acc: 0.7780
    Epoch 732/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8431 - acc: 0.7921 - val_loss: 0.8837 - val_acc: 0.7870
    Epoch 733/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8423 - acc: 0.7963 - val_loss: 0.8869 - val_acc: 0.7830
    Epoch 734/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8432 - acc: 0.7928 - val_loss: 0.8832 - val_acc: 0.7810
    Epoch 735/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8428 - acc: 0.7904 - val_loss: 0.8947 - val_acc: 0.7830
    Epoch 736/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8440 - acc: 0.7947 - val_loss: 0.8818 - val_acc: 0.7790
    Epoch 737/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8420 - acc: 0.7917 - val_loss: 0.8940 - val_acc: 0.7700
    Epoch 738/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8430 - acc: 0.7908 - val_loss: 0.8847 - val_acc: 0.7800
    Epoch 739/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8435 - acc: 0.7927 - val_loss: 0.8865 - val_acc: 0.7790
    Epoch 740/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8426 - acc: 0.7943 - val_loss: 0.8859 - val_acc: 0.7760
    Epoch 741/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8425 - acc: 0.7939 - val_loss: 0.9050 - val_acc: 0.7740
    Epoch 742/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8425 - acc: 0.7928 - val_loss: 0.8997 - val_acc: 0.7780
    Epoch 743/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8421 - acc: 0.7923 - val_loss: 0.8893 - val_acc: 0.7820
    Epoch 744/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8438 - acc: 0.7913 - val_loss: 0.8822 - val_acc: 0.7750
    Epoch 745/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8423 - acc: 0.7928 - val_loss: 0.9366 - val_acc: 0.7670
    Epoch 746/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8441 - acc: 0.7939 - val_loss: 0.9119 - val_acc: 0.7780
    Epoch 747/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8423 - acc: 0.7931 - val_loss: 0.8834 - val_acc: 0.7760
    Epoch 748/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8413 - acc: 0.7931 - val_loss: 0.9075 - val_acc: 0.7730
    Epoch 749/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8414 - acc: 0.7947 - val_loss: 0.8813 - val_acc: 0.7880
    Epoch 750/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8412 - acc: 0.7937 - val_loss: 0.8936 - val_acc: 0.7750
    Epoch 751/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8421 - acc: 0.7945 - val_loss: 0.9002 - val_acc: 0.7770
    Epoch 752/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8423 - acc: 0.7956 - val_loss: 0.8854 - val_acc: 0.7820
    Epoch 753/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8402 - acc: 0.7943 - val_loss: 0.8942 - val_acc: 0.7740
    Epoch 754/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8407 - acc: 0.7924 - val_loss: 0.8893 - val_acc: 0.7760
    Epoch 755/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8400 - acc: 0.7920 - val_loss: 0.8843 - val_acc: 0.7790
    Epoch 756/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8406 - acc: 0.7933 - val_loss: 0.8810 - val_acc: 0.7850
    Epoch 757/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8395 - acc: 0.7919 - val_loss: 0.8845 - val_acc: 0.7820
    Epoch 758/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8421 - acc: 0.7964 - val_loss: 0.8804 - val_acc: 0.7780
    Epoch 759/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8397 - acc: 0.7951 - val_loss: 0.8851 - val_acc: 0.7800
    Epoch 760/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8389 - acc: 0.7944 - val_loss: 0.8923 - val_acc: 0.7720
    Epoch 761/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8410 - acc: 0.7916 - val_loss: 0.8793 - val_acc: 0.7850
    Epoch 762/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8389 - acc: 0.7963 - val_loss: 0.8828 - val_acc: 0.7770
    Epoch 763/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8384 - acc: 0.7959 - val_loss: 0.8856 - val_acc: 0.7740
    Epoch 764/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8400 - acc: 0.7955 - val_loss: 0.8850 - val_acc: 0.7830
    Epoch 765/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8390 - acc: 0.7948 - val_loss: 0.8872 - val_acc: 0.7780
    Epoch 766/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8390 - acc: 0.7921 - val_loss: 0.8819 - val_acc: 0.7850
    Epoch 767/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8388 - acc: 0.7952 - val_loss: 0.8823 - val_acc: 0.7810
    Epoch 768/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8390 - acc: 0.7949 - val_loss: 0.8953 - val_acc: 0.7810
    Epoch 769/1000
    7500/7500 [==============================] - 0s 17us/step - loss: 0.8413 - acc: 0.7935 - val_loss: 0.8839 - val_acc: 0.7810
    Epoch 770/1000
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8383 - acc: 0.7947 - val_loss: 0.8879 - val_acc: 0.7760
    Epoch 771/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8401 - acc: 0.7968 - val_loss: 0.8827 - val_acc: 0.7860
    Epoch 772/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8377 - acc: 0.7955 - val_loss: 0.9031 - val_acc: 0.7790
    Epoch 773/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8385 - acc: 0.7961 - val_loss: 0.8869 - val_acc: 0.7760
    Epoch 774/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8389 - acc: 0.7945 - val_loss: 0.8810 - val_acc: 0.7860
    Epoch 775/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8374 - acc: 0.7943 - val_loss: 0.8809 - val_acc: 0.7860
    Epoch 776/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8391 - acc: 0.7928 - val_loss: 0.8962 - val_acc: 0.7690
    Epoch 777/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8400 - acc: 0.7924 - val_loss: 0.8813 - val_acc: 0.7780
    Epoch 778/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8376 - acc: 0.7963 - val_loss: 0.8876 - val_acc: 0.7710
    Epoch 779/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8381 - acc: 0.7943 - val_loss: 0.8806 - val_acc: 0.7850
    Epoch 780/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8374 - acc: 0.7933 - val_loss: 0.8887 - val_acc: 0.7730
    Epoch 781/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8361 - acc: 0.7945 - val_loss: 0.8795 - val_acc: 0.7790
    Epoch 782/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8378 - acc: 0.7963 - val_loss: 0.8880 - val_acc: 0.7850
    Epoch 783/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8382 - acc: 0.7928 - val_loss: 0.8798 - val_acc: 0.7820
    Epoch 784/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8359 - acc: 0.7971 - val_loss: 0.8801 - val_acc: 0.7870
    Epoch 785/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8384 - acc: 0.7949 - val_loss: 0.8794 - val_acc: 0.7860
    Epoch 786/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8377 - acc: 0.7963 - val_loss: 0.8877 - val_acc: 0.7810
    Epoch 787/1000
    7500/7500 [==============================] - 0s 16us/step - loss: 0.8371 - acc: 0.7961 - val_loss: 0.9023 - val_acc: 0.7740
    Epoch 788/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8373 - acc: 0.7989 - val_loss: 0.9211 - val_acc: 0.7680
    Epoch 789/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8367 - acc: 0.7963 - val_loss: 0.8993 - val_acc: 0.7760
    Epoch 790/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8365 - acc: 0.7960 - val_loss: 0.8887 - val_acc: 0.7780
    Epoch 791/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8355 - acc: 0.7969 - val_loss: 0.9002 - val_acc: 0.7700
    Epoch 792/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8358 - acc: 0.7968 - val_loss: 0.8993 - val_acc: 0.7830
    Epoch 793/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8390 - acc: 0.7968 - val_loss: 0.8859 - val_acc: 0.7800
    Epoch 794/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8361 - acc: 0.7968 - val_loss: 0.8826 - val_acc: 0.7830
    Epoch 795/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8360 - acc: 0.7947 - val_loss: 0.8808 - val_acc: 0.7820
    Epoch 796/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8354 - acc: 0.7973 - val_loss: 0.8828 - val_acc: 0.7790
    Epoch 797/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8338 - acc: 0.7955 - val_loss: 0.8913 - val_acc: 0.7730
    Epoch 798/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8367 - acc: 0.7957 - val_loss: 0.8972 - val_acc: 0.7840
    Epoch 799/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8348 - acc: 0.7965 - val_loss: 0.8832 - val_acc: 0.7790
    Epoch 800/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8356 - acc: 0.7940 - val_loss: 0.8844 - val_acc: 0.7820
    Epoch 801/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8358 - acc: 0.7975 - val_loss: 0.8779 - val_acc: 0.7870
    Epoch 802/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8354 - acc: 0.7961 - val_loss: 0.8796 - val_acc: 0.7840
    Epoch 803/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8340 - acc: 0.7955 - val_loss: 0.8781 - val_acc: 0.7870
    Epoch 804/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8332 - acc: 0.7965 - val_loss: 0.8893 - val_acc: 0.7790
    Epoch 805/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8341 - acc: 0.7987 - val_loss: 0.8833 - val_acc: 0.7860
    Epoch 806/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8342 - acc: 0.7969 - val_loss: 0.9129 - val_acc: 0.7770
    Epoch 807/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8377 - acc: 0.7925 - val_loss: 0.8776 - val_acc: 0.7890
    Epoch 808/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8330 - acc: 0.7973 - val_loss: 0.8793 - val_acc: 0.7800
    Epoch 809/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8327 - acc: 0.7969 - val_loss: 0.9022 - val_acc: 0.7710
    Epoch 810/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8336 - acc: 0.7952 - val_loss: 0.9033 - val_acc: 0.7700
    Epoch 811/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8370 - acc: 0.7933 - val_loss: 0.8888 - val_acc: 0.7740
    Epoch 812/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8329 - acc: 0.7985 - val_loss: 0.9064 - val_acc: 0.7740
    Epoch 813/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8370 - acc: 0.7957 - val_loss: 0.8871 - val_acc: 0.7720
    Epoch 814/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8330 - acc: 0.7972 - val_loss: 0.8872 - val_acc: 0.7750
    Epoch 815/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8338 - acc: 0.7961 - val_loss: 0.8819 - val_acc: 0.7810
    Epoch 816/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8328 - acc: 0.7971 - val_loss: 0.8826 - val_acc: 0.7790
    Epoch 817/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8327 - acc: 0.7955 - val_loss: 0.8945 - val_acc: 0.7820
    Epoch 818/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8318 - acc: 0.7985 - val_loss: 0.8809 - val_acc: 0.7790
    Epoch 819/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8322 - acc: 0.7980 - val_loss: 0.8957 - val_acc: 0.7800
    Epoch 820/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8322 - acc: 0.7973 - val_loss: 0.8853 - val_acc: 0.7840
    Epoch 821/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8330 - acc: 0.7975 - val_loss: 0.8849 - val_acc: 0.7800
    Epoch 822/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8321 - acc: 0.7979 - val_loss: 0.8959 - val_acc: 0.7820
    Epoch 823/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8328 - acc: 0.7973 - val_loss: 0.8815 - val_acc: 0.7850
    Epoch 824/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8325 - acc: 0.7976 - val_loss: 0.8794 - val_acc: 0.7820
    Epoch 825/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8325 - acc: 0.7960 - val_loss: 0.8800 - val_acc: 0.7840
    Epoch 826/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8309 - acc: 0.7965 - val_loss: 0.8869 - val_acc: 0.7770
    Epoch 827/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8311 - acc: 0.7975 - val_loss: 0.8780 - val_acc: 0.7900
    Epoch 828/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8312 - acc: 0.7968 - val_loss: 0.8917 - val_acc: 0.7790
    Epoch 829/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8336 - acc: 0.7971 - val_loss: 0.8899 - val_acc: 0.7750
    Epoch 830/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8315 - acc: 0.7984 - val_loss: 0.8856 - val_acc: 0.7810
    Epoch 831/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8316 - acc: 0.7988 - val_loss: 0.8832 - val_acc: 0.7870
    Epoch 832/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8332 - acc: 0.7971 - val_loss: 0.8888 - val_acc: 0.7800
    Epoch 833/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8319 - acc: 0.7995 - val_loss: 0.8774 - val_acc: 0.7830
    Epoch 834/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8314 - acc: 0.7972 - val_loss: 0.8870 - val_acc: 0.7850
    Epoch 835/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8301 - acc: 0.7972 - val_loss: 0.8839 - val_acc: 0.7860
    Epoch 836/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8315 - acc: 0.7985 - val_loss: 0.8801 - val_acc: 0.7880
    Epoch 837/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8331 - acc: 0.7947 - val_loss: 0.8952 - val_acc: 0.7830
    Epoch 838/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8300 - acc: 0.7993 - val_loss: 0.8967 - val_acc: 0.7780
    Epoch 839/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8316 - acc: 0.7965 - val_loss: 0.8776 - val_acc: 0.7850
    Epoch 840/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8301 - acc: 0.7981 - val_loss: 0.8813 - val_acc: 0.7850
    Epoch 841/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8294 - acc: 0.8005 - val_loss: 0.8992 - val_acc: 0.7820
    Epoch 842/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8305 - acc: 0.7973 - val_loss: 0.8770 - val_acc: 0.7910
    Epoch 843/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8291 - acc: 0.7995 - val_loss: 0.8826 - val_acc: 0.7820
    Epoch 844/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8294 - acc: 0.7984 - val_loss: 0.9032 - val_acc: 0.7800
    Epoch 845/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8296 - acc: 0.7995 - val_loss: 0.8820 - val_acc: 0.7840
    Epoch 846/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8318 - acc: 0.7979 - val_loss: 0.9109 - val_acc: 0.7820
    Epoch 847/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8314 - acc: 0.7977 - val_loss: 0.8815 - val_acc: 0.7840
    Epoch 848/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8281 - acc: 0.7999 - val_loss: 0.8924 - val_acc: 0.7820
    Epoch 849/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8301 - acc: 0.7989 - val_loss: 0.8900 - val_acc: 0.7780
    Epoch 850/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8301 - acc: 0.7983 - val_loss: 0.8802 - val_acc: 0.7830
    Epoch 851/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8285 - acc: 0.8011 - val_loss: 0.8919 - val_acc: 0.7730
    Epoch 852/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8287 - acc: 0.8005 - val_loss: 0.8824 - val_acc: 0.7840
    Epoch 853/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8266 - acc: 0.8008 - val_loss: 0.8833 - val_acc: 0.7830
    Epoch 854/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8285 - acc: 0.8000 - val_loss: 0.8771 - val_acc: 0.7860
    Epoch 855/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8287 - acc: 0.7988 - val_loss: 0.8921 - val_acc: 0.7860
    Epoch 856/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8283 - acc: 0.7971 - val_loss: 0.8828 - val_acc: 0.7810
    Epoch 857/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8270 - acc: 0.8001 - val_loss: 0.8812 - val_acc: 0.7820
    Epoch 858/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8290 - acc: 0.7981 - val_loss: 0.8802 - val_acc: 0.7840
    Epoch 859/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8282 - acc: 0.8021 - val_loss: 0.8976 - val_acc: 0.7680
    Epoch 860/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8279 - acc: 0.7984 - val_loss: 0.8945 - val_acc: 0.7700
    Epoch 861/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8277 - acc: 0.8024 - val_loss: 0.8868 - val_acc: 0.7860
    Epoch 862/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8270 - acc: 0.7979 - val_loss: 0.8819 - val_acc: 0.7790
    Epoch 863/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8291 - acc: 0.7988 - val_loss: 0.8818 - val_acc: 0.7840
    Epoch 864/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8265 - acc: 0.7987 - val_loss: 0.8880 - val_acc: 0.7800
    Epoch 865/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8272 - acc: 0.7993 - val_loss: 0.8792 - val_acc: 0.7870
    Epoch 866/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8273 - acc: 0.8013 - val_loss: 0.8761 - val_acc: 0.7860
    Epoch 867/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8270 - acc: 0.7983 - val_loss: 0.8841 - val_acc: 0.7820
    Epoch 868/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8270 - acc: 0.7987 - val_loss: 0.8770 - val_acc: 0.7780
    Epoch 869/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8271 - acc: 0.8015 - val_loss: 0.8813 - val_acc: 0.7810
    Epoch 870/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8273 - acc: 0.7976 - val_loss: 0.8883 - val_acc: 0.7780
    Epoch 871/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8247 - acc: 0.7993 - val_loss: 0.9146 - val_acc: 0.7780
    Epoch 872/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8283 - acc: 0.7989 - val_loss: 0.8813 - val_acc: 0.7830
    Epoch 873/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8260 - acc: 0.8023 - val_loss: 0.8891 - val_acc: 0.7900
    Epoch 874/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8279 - acc: 0.8012 - val_loss: 0.8811 - val_acc: 0.7830
    Epoch 875/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8266 - acc: 0.7995 - val_loss: 0.8790 - val_acc: 0.7850
    Epoch 876/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8266 - acc: 0.8027 - val_loss: 0.8876 - val_acc: 0.7820
    Epoch 877/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8281 - acc: 0.8024 - val_loss: 0.8894 - val_acc: 0.7750
    Epoch 878/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8250 - acc: 0.8007 - val_loss: 0.8775 - val_acc: 0.7800
    Epoch 879/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8258 - acc: 0.8031 - val_loss: 0.8886 - val_acc: 0.7810
    Epoch 880/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8270 - acc: 0.7996 - val_loss: 0.8804 - val_acc: 0.7800
    Epoch 881/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8263 - acc: 0.8007 - val_loss: 0.8799 - val_acc: 0.7890
    Epoch 882/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8256 - acc: 0.8004 - val_loss: 0.8852 - val_acc: 0.7940
    Epoch 883/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8267 - acc: 0.7971 - val_loss: 0.8929 - val_acc: 0.7840
    Epoch 884/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8250 - acc: 0.8017 - val_loss: 0.8922 - val_acc: 0.7810
    Epoch 885/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8249 - acc: 0.8016 - val_loss: 0.8954 - val_acc: 0.7770
    Epoch 886/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8270 - acc: 0.8032 - val_loss: 0.8948 - val_acc: 0.7800
    Epoch 887/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8241 - acc: 0.8003 - val_loss: 0.8827 - val_acc: 0.7850
    Epoch 888/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8252 - acc: 0.8027 - val_loss: 0.8885 - val_acc: 0.7860
    Epoch 889/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8256 - acc: 0.8033 - val_loss: 0.8966 - val_acc: 0.7700
    Epoch 890/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8255 - acc: 0.8015 - val_loss: 0.8834 - val_acc: 0.7830
    Epoch 891/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8241 - acc: 0.8009 - val_loss: 0.9061 - val_acc: 0.7680
    Epoch 892/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8245 - acc: 0.8019 - val_loss: 0.8854 - val_acc: 0.7840
    Epoch 893/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8223 - acc: 0.8021 - val_loss: 0.8868 - val_acc: 0.7810
    Epoch 894/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8246 - acc: 0.8011 - val_loss: 0.8796 - val_acc: 0.7860
    Epoch 895/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8242 - acc: 0.8025 - val_loss: 0.8756 - val_acc: 0.7910
    Epoch 896/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8235 - acc: 0.8025 - val_loss: 0.8860 - val_acc: 0.7790
    Epoch 897/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8241 - acc: 0.8015 - val_loss: 0.8919 - val_acc: 0.7830
    Epoch 898/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8257 - acc: 0.8011 - val_loss: 0.8818 - val_acc: 0.7900
    Epoch 899/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8263 - acc: 0.8023 - val_loss: 0.8757 - val_acc: 0.7920
    Epoch 900/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8228 - acc: 0.7989 - val_loss: 0.9042 - val_acc: 0.7810
    Epoch 901/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8250 - acc: 0.8020 - val_loss: 0.8756 - val_acc: 0.7950
    Epoch 902/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8237 - acc: 0.8035 - val_loss: 0.8882 - val_acc: 0.7840
    Epoch 903/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8218 - acc: 0.8016 - val_loss: 0.8838 - val_acc: 0.7850
    Epoch 904/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8235 - acc: 0.8001 - val_loss: 0.8829 - val_acc: 0.7780
    Epoch 905/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8216 - acc: 0.8024 - val_loss: 0.8868 - val_acc: 0.7780
    Epoch 906/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8198 - acc: 0.8039 - val_loss: 0.8845 - val_acc: 0.7800
    Epoch 907/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8214 - acc: 0.8048 - val_loss: 0.8952 - val_acc: 0.7750
    Epoch 908/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8220 - acc: 0.8029 - val_loss: 0.8940 - val_acc: 0.7850
    Epoch 909/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8220 - acc: 0.8035 - val_loss: 0.9055 - val_acc: 0.7640
    Epoch 910/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8240 - acc: 0.7999 - val_loss: 0.8798 - val_acc: 0.7850
    Epoch 911/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8234 - acc: 0.8007 - val_loss: 0.8980 - val_acc: 0.7760
    Epoch 912/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8256 - acc: 0.7993 - val_loss: 0.8883 - val_acc: 0.7760
    Epoch 913/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8212 - acc: 0.8025 - val_loss: 0.8823 - val_acc: 0.7860
    Epoch 914/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8220 - acc: 0.8007 - val_loss: 0.8785 - val_acc: 0.7850
    Epoch 915/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8216 - acc: 0.8029 - val_loss: 0.8775 - val_acc: 0.7880
    Epoch 916/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8213 - acc: 0.8036 - val_loss: 0.8810 - val_acc: 0.7870
    Epoch 917/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8225 - acc: 0.8039 - val_loss: 0.9121 - val_acc: 0.7710
    Epoch 918/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8208 - acc: 0.8021 - val_loss: 0.9108 - val_acc: 0.7790
    Epoch 919/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8215 - acc: 0.8047 - val_loss: 0.9092 - val_acc: 0.7790
    Epoch 920/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8213 - acc: 0.8040 - val_loss: 0.8793 - val_acc: 0.7940
    Epoch 921/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8224 - acc: 0.8025 - val_loss: 0.8840 - val_acc: 0.7900
    Epoch 922/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8225 - acc: 0.8033 - val_loss: 0.8801 - val_acc: 0.7770
    Epoch 923/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8221 - acc: 0.8024 - val_loss: 0.8920 - val_acc: 0.7840
    Epoch 924/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8221 - acc: 0.8004 - val_loss: 0.9032 - val_acc: 0.7760
    Epoch 925/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8215 - acc: 0.8063 - val_loss: 0.8782 - val_acc: 0.7880
    Epoch 926/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8189 - acc: 0.8052 - val_loss: 0.8863 - val_acc: 0.7830
    Epoch 927/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8199 - acc: 0.8031 - val_loss: 0.8846 - val_acc: 0.7790
    Epoch 928/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8210 - acc: 0.8027 - val_loss: 0.8843 - val_acc: 0.7820
    Epoch 929/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8204 - acc: 0.8021 - val_loss: 0.8820 - val_acc: 0.7720
    Epoch 930/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8198 - acc: 0.8023 - val_loss: 0.8822 - val_acc: 0.7860
    Epoch 931/1000
    7500/7500 [==============================] - 0s 13us/step - loss: 0.8212 - acc: 0.8045 - val_loss: 0.8767 - val_acc: 0.7910
    Epoch 932/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8211 - acc: 0.8029 - val_loss: 0.8864 - val_acc: 0.7850
    Epoch 933/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8187 - acc: 0.8028 - val_loss: 0.8877 - val_acc: 0.7790
    Epoch 934/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8205 - acc: 0.8035 - val_loss: 0.8872 - val_acc: 0.7830
    Epoch 935/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8208 - acc: 0.8023 - val_loss: 0.8827 - val_acc: 0.7870
    Epoch 936/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8204 - acc: 0.8023 - val_loss: 0.8780 - val_acc: 0.7870
    Epoch 937/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8180 - acc: 0.8036 - val_loss: 0.8819 - val_acc: 0.7860
    Epoch 938/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8194 - acc: 0.8029 - val_loss: 0.8828 - val_acc: 0.7930
    Epoch 939/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8196 - acc: 0.8051 - val_loss: 0.8804 - val_acc: 0.7910
    Epoch 940/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8191 - acc: 0.8027 - val_loss: 0.8966 - val_acc: 0.7820
    Epoch 941/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8181 - acc: 0.8028 - val_loss: 0.8790 - val_acc: 0.7890
    Epoch 942/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8181 - acc: 0.8044 - val_loss: 0.8995 - val_acc: 0.7780
    Epoch 943/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8198 - acc: 0.8053 - val_loss: 0.8916 - val_acc: 0.7750
    Epoch 944/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8216 - acc: 0.8029 - val_loss: 0.9019 - val_acc: 0.7750
    Epoch 945/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8176 - acc: 0.8063 - val_loss: 0.8900 - val_acc: 0.7780
    Epoch 946/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8205 - acc: 0.8029 - val_loss: 0.8798 - val_acc: 0.7850
    Epoch 947/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8194 - acc: 0.8033 - val_loss: 0.8959 - val_acc: 0.7820
    Epoch 948/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8210 - acc: 0.8028 - val_loss: 0.8803 - val_acc: 0.7910
    Epoch 949/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8179 - acc: 0.8073 - val_loss: 0.8862 - val_acc: 0.7790
    Epoch 950/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8165 - acc: 0.8040 - val_loss: 0.9317 - val_acc: 0.7620
    Epoch 951/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8179 - acc: 0.8071 - val_loss: 0.8750 - val_acc: 0.7900
    Epoch 952/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8178 - acc: 0.8041 - val_loss: 0.8797 - val_acc: 0.7790
    Epoch 953/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8175 - acc: 0.8039 - val_loss: 0.8930 - val_acc: 0.7880
    Epoch 954/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8179 - acc: 0.8044 - val_loss: 0.8786 - val_acc: 0.7880
    Epoch 955/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8186 - acc: 0.8033 - val_loss: 0.8794 - val_acc: 0.7860
    Epoch 956/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8162 - acc: 0.8041 - val_loss: 0.8800 - val_acc: 0.7900
    Epoch 957/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8177 - acc: 0.8059 - val_loss: 0.8820 - val_acc: 0.7810
    Epoch 958/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8174 - acc: 0.8057 - val_loss: 0.8742 - val_acc: 0.7820
    Epoch 959/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8148 - acc: 0.8081 - val_loss: 0.8805 - val_acc: 0.7840
    Epoch 960/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8157 - acc: 0.8045 - val_loss: 0.8899 - val_acc: 0.7840
    Epoch 961/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8162 - acc: 0.8041 - val_loss: 0.9292 - val_acc: 0.7670
    Epoch 962/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8172 - acc: 0.8088 - val_loss: 0.8766 - val_acc: 0.7910
    Epoch 963/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8185 - acc: 0.8044 - val_loss: 0.8776 - val_acc: 0.7810
    Epoch 964/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8173 - acc: 0.8036 - val_loss: 0.8911 - val_acc: 0.7790
    Epoch 965/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8162 - acc: 0.8063 - val_loss: 0.8854 - val_acc: 0.7770
    Epoch 966/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8160 - acc: 0.8079 - val_loss: 0.8840 - val_acc: 0.7840
    Epoch 967/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8165 - acc: 0.8068 - val_loss: 0.8796 - val_acc: 0.7780
    Epoch 968/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8162 - acc: 0.8056 - val_loss: 0.8888 - val_acc: 0.7810
    Epoch 969/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8162 - acc: 0.8045 - val_loss: 0.8868 - val_acc: 0.7770
    Epoch 970/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8145 - acc: 0.8065 - val_loss: 0.9019 - val_acc: 0.7830
    Epoch 971/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8165 - acc: 0.8068 - val_loss: 0.8838 - val_acc: 0.7810
    Epoch 972/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8151 - acc: 0.8056 - val_loss: 0.8833 - val_acc: 0.7750
    Epoch 973/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8163 - acc: 0.8060 - val_loss: 0.9443 - val_acc: 0.7800
    Epoch 974/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8187 - acc: 0.8027 - val_loss: 0.8825 - val_acc: 0.7830
    Epoch 975/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8158 - acc: 0.8065 - val_loss: 0.8768 - val_acc: 0.7850
    Epoch 976/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8150 - acc: 0.8071 - val_loss: 0.8841 - val_acc: 0.7800
    Epoch 977/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8141 - acc: 0.8073 - val_loss: 0.8789 - val_acc: 0.7860
    Epoch 978/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8163 - acc: 0.8053 - val_loss: 0.8818 - val_acc: 0.7770
    Epoch 979/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8145 - acc: 0.8072 - val_loss: 0.8860 - val_acc: 0.7800
    Epoch 980/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8142 - acc: 0.8072 - val_loss: 0.8945 - val_acc: 0.7750
    Epoch 981/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8154 - acc: 0.8060 - val_loss: 0.9381 - val_acc: 0.7660
    Epoch 982/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8157 - acc: 0.8048 - val_loss: 0.8944 - val_acc: 0.7860
    Epoch 983/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8148 - acc: 0.8075 - val_loss: 0.8859 - val_acc: 0.7880
    Epoch 984/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8132 - acc: 0.8079 - val_loss: 0.8953 - val_acc: 0.7700
    Epoch 985/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8169 - acc: 0.8063 - val_loss: 0.9216 - val_acc: 0.7710
    Epoch 986/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8152 - acc: 0.8048 - val_loss: 0.8769 - val_acc: 0.7830
    Epoch 987/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8145 - acc: 0.8077 - val_loss: 0.9095 - val_acc: 0.7800
    Epoch 988/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8174 - acc: 0.8057 - val_loss: 0.8812 - val_acc: 0.7780
    Epoch 989/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8155 - acc: 0.8028 - val_loss: 0.8764 - val_acc: 0.7950
    Epoch 990/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8147 - acc: 0.8079 - val_loss: 0.8881 - val_acc: 0.7860
    Epoch 991/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8122 - acc: 0.8085 - val_loss: 0.9002 - val_acc: 0.7820
    Epoch 992/1000
    7500/7500 [==============================] - 0s 14us/step - loss: 0.8130 - acc: 0.8093 - val_loss: 0.8827 - val_acc: 0.7750
    Epoch 993/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8133 - acc: 0.8056 - val_loss: 0.8779 - val_acc: 0.7820
    Epoch 994/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8132 - acc: 0.8088 - val_loss: 0.8861 - val_acc: 0.7960
    Epoch 995/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8128 - acc: 0.8065 - val_loss: 0.8856 - val_acc: 0.7740
    Epoch 996/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8132 - acc: 0.8076 - val_loss: 0.8790 - val_acc: 0.7950
    Epoch 997/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8145 - acc: 0.8069 - val_loss: 0.8991 - val_acc: 0.7670
    Epoch 998/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8133 - acc: 0.8064 - val_loss: 0.8902 - val_acc: 0.7770
    Epoch 999/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8137 - acc: 0.8075 - val_loss: 0.8871 - val_acc: 0.7720
    Epoch 1000/1000
    7500/7500 [==============================] - 0s 15us/step - loss: 0.8137 - acc: 0.8087 - val_loss: 0.8996 - val_acc: 0.7790



```python
# __SOLUTION__ 
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


![png](index_files/index_95_0.png)



```python
# __SOLUTION__ 
results_train = model.evaluate(train_final, label_train_final)

results_test = model.evaluate(X_test, y_test)
```

    7500/7500 [==============================] - 0s 22us/step
    1500/1500 [==============================] - 0s 22us/step



```python
# __SOLUTION__ 
results_train
```




    [0.8224783395131429, 0.7962666666348776]




```python
# __SOLUTION__ 
results_test
```




    [0.9036528784434, 0.7673333330154419]



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


```python
results_train = model.evaluate(train_final, label_train_final)
results_test = model.evaluate(X_test, y_test)
```


```python
results_train
```


```python
results_test
```


```python
# __SOLUTION__ 
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
    7500/7500 [==============================] - 0s 46us/step - loss: 1.9610 - acc: 0.1509 - val_loss: 1.9359 - val_acc: 0.1570
    Epoch 2/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.9396 - acc: 0.1693 - val_loss: 1.9257 - val_acc: 0.1680
    Epoch 3/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9336 - acc: 0.1725 - val_loss: 1.9176 - val_acc: 0.1810
    Epoch 4/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9287 - acc: 0.1849 - val_loss: 1.9109 - val_acc: 0.1910
    Epoch 5/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.9187 - acc: 0.1817 - val_loss: 1.9039 - val_acc: 0.1920
    Epoch 6/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.9145 - acc: 0.1867 - val_loss: 1.8971 - val_acc: 0.1900
    Epoch 7/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.9088 - acc: 0.1941 - val_loss: 1.8896 - val_acc: 0.1860
    Epoch 8/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.9021 - acc: 0.2044 - val_loss: 1.8808 - val_acc: 0.1930
    Epoch 9/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.8865 - acc: 0.2076 - val_loss: 1.8709 - val_acc: 0.1920
    Epoch 10/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.8854 - acc: 0.2111 - val_loss: 1.8612 - val_acc: 0.1950
    Epoch 11/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8810 - acc: 0.2143 - val_loss: 1.8508 - val_acc: 0.1970
    Epoch 12/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8695 - acc: 0.2223 - val_loss: 1.8392 - val_acc: 0.2040
    Epoch 13/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.8588 - acc: 0.2307 - val_loss: 1.8262 - val_acc: 0.2150
    Epoch 14/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.8447 - acc: 0.2389 - val_loss: 1.8120 - val_acc: 0.2190
    Epoch 15/200
    7500/7500 [==============================] - 0s 21us/step - loss: 1.8339 - acc: 0.2493 - val_loss: 1.7977 - val_acc: 0.2390
    Epoch 16/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8225 - acc: 0.2527 - val_loss: 1.7822 - val_acc: 0.2620
    Epoch 17/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8193 - acc: 0.2547 - val_loss: 1.7671 - val_acc: 0.2770
    Epoch 18/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.8031 - acc: 0.2656 - val_loss: 1.7503 - val_acc: 0.2940
    Epoch 19/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7927 - acc: 0.2744 - val_loss: 1.7320 - val_acc: 0.3120
    Epoch 20/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7814 - acc: 0.2839 - val_loss: 1.7130 - val_acc: 0.3390
    Epoch 21/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7593 - acc: 0.2959 - val_loss: 1.6926 - val_acc: 0.3600
    Epoch 22/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.7517 - acc: 0.3031 - val_loss: 1.6722 - val_acc: 0.3720
    Epoch 23/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.7395 - acc: 0.3063 - val_loss: 1.6523 - val_acc: 0.3770
    Epoch 24/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.7142 - acc: 0.3168 - val_loss: 1.6308 - val_acc: 0.3990
    Epoch 25/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6996 - acc: 0.3361 - val_loss: 1.6081 - val_acc: 0.4120
    Epoch 26/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6929 - acc: 0.3347 - val_loss: 1.5861 - val_acc: 0.4350
    Epoch 27/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6649 - acc: 0.3432 - val_loss: 1.5629 - val_acc: 0.4570
    Epoch 28/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.6606 - acc: 0.3471 - val_loss: 1.5405 - val_acc: 0.4670
    Epoch 29/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.6454 - acc: 0.3529 - val_loss: 1.5193 - val_acc: 0.4830
    Epoch 30/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.6219 - acc: 0.3712 - val_loss: 1.4961 - val_acc: 0.4810
    Epoch 31/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5969 - acc: 0.3796 - val_loss: 1.4729 - val_acc: 0.4970
    Epoch 32/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.5867 - acc: 0.3851 - val_loss: 1.4520 - val_acc: 0.5090
    Epoch 33/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5802 - acc: 0.3915 - val_loss: 1.4315 - val_acc: 0.5220
    Epoch 34/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.5605 - acc: 0.4008 - val_loss: 1.4094 - val_acc: 0.5230
    Epoch 35/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.5511 - acc: 0.3927 - val_loss: 1.3909 - val_acc: 0.5370
    Epoch 36/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5306 - acc: 0.4127 - val_loss: 1.3703 - val_acc: 0.5470
    Epoch 37/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.5265 - acc: 0.4139 - val_loss: 1.3522 - val_acc: 0.5550
    Epoch 38/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.5032 - acc: 0.4197 - val_loss: 1.3319 - val_acc: 0.5630
    Epoch 39/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.4957 - acc: 0.4205 - val_loss: 1.3130 - val_acc: 0.5650
    Epoch 40/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.4774 - acc: 0.4296 - val_loss: 1.2927 - val_acc: 0.5800
    Epoch 41/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.4600 - acc: 0.4392 - val_loss: 1.2755 - val_acc: 0.5800
    Epoch 42/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.4551 - acc: 0.4367 - val_loss: 1.2587 - val_acc: 0.5910
    Epoch 43/200
    7500/7500 [==============================] - 0s 21us/step - loss: 1.4378 - acc: 0.4467 - val_loss: 1.2407 - val_acc: 0.6040
    Epoch 44/200
    7500/7500 [==============================] - 0s 21us/step - loss: 1.4254 - acc: 0.4465 - val_loss: 1.2243 - val_acc: 0.6050
    Epoch 45/200
    7500/7500 [==============================] - 0s 21us/step - loss: 1.4109 - acc: 0.4600 - val_loss: 1.2067 - val_acc: 0.6060
    Epoch 46/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3954 - acc: 0.4604 - val_loss: 1.1919 - val_acc: 0.6110
    Epoch 47/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3970 - acc: 0.4664 - val_loss: 1.1781 - val_acc: 0.6230
    Epoch 48/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3806 - acc: 0.4771 - val_loss: 1.1628 - val_acc: 0.6260
    Epoch 49/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.3716 - acc: 0.4717 - val_loss: 1.1477 - val_acc: 0.6260
    Epoch 50/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.3580 - acc: 0.4832 - val_loss: 1.1337 - val_acc: 0.6350
    Epoch 51/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3441 - acc: 0.4920 - val_loss: 1.1199 - val_acc: 0.6390
    Epoch 52/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3335 - acc: 0.4957 - val_loss: 1.1045 - val_acc: 0.6490
    Epoch 53/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3123 - acc: 0.5005 - val_loss: 1.0902 - val_acc: 0.6500
    Epoch 54/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.3165 - acc: 0.4967 - val_loss: 1.0796 - val_acc: 0.6530
    Epoch 55/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.3035 - acc: 0.5013 - val_loss: 1.0662 - val_acc: 0.6650
    Epoch 56/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2895 - acc: 0.4993 - val_loss: 1.0527 - val_acc: 0.6690
    Epoch 57/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2844 - acc: 0.5068 - val_loss: 1.0410 - val_acc: 0.6730
    Epoch 58/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2781 - acc: 0.5105 - val_loss: 1.0323 - val_acc: 0.6760
    Epoch 59/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2558 - acc: 0.5188 - val_loss: 1.0170 - val_acc: 0.6800
    Epoch 60/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2356 - acc: 0.5264 - val_loss: 1.0025 - val_acc: 0.6830
    Epoch 61/200
    7500/7500 [==============================] - 0s 17us/step - loss: 1.2457 - acc: 0.5271 - val_loss: 0.9947 - val_acc: 0.6820
    Epoch 62/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2294 - acc: 0.5359 - val_loss: 0.9866 - val_acc: 0.6870
    Epoch 63/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2345 - acc: 0.5317 - val_loss: 0.9769 - val_acc: 0.6940
    Epoch 64/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2212 - acc: 0.5404 - val_loss: 0.9678 - val_acc: 0.6930
    Epoch 65/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2089 - acc: 0.5463 - val_loss: 0.9577 - val_acc: 0.6990
    Epoch 66/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.2080 - acc: 0.5413 - val_loss: 0.9482 - val_acc: 0.7010
    Epoch 67/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1889 - acc: 0.5528 - val_loss: 0.9365 - val_acc: 0.7010
    Epoch 68/200
    7500/7500 [==============================] - 0s 17us/step - loss: 1.1824 - acc: 0.5563 - val_loss: 0.9301 - val_acc: 0.7060
    Epoch 69/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1884 - acc: 0.5532 - val_loss: 0.9211 - val_acc: 0.7160
    Epoch 70/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1560 - acc: 0.5644 - val_loss: 0.9130 - val_acc: 0.7100
    Epoch 71/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1620 - acc: 0.5592 - val_loss: 0.9055 - val_acc: 0.7090
    Epoch 72/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1670 - acc: 0.5577 - val_loss: 0.8989 - val_acc: 0.7170
    Epoch 73/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1403 - acc: 0.5587 - val_loss: 0.8899 - val_acc: 0.7150
    Epoch 74/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1426 - acc: 0.5699 - val_loss: 0.8827 - val_acc: 0.7160
    Epoch 75/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.1317 - acc: 0.5656 - val_loss: 0.8754 - val_acc: 0.7200
    Epoch 76/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1224 - acc: 0.5832 - val_loss: 0.8677 - val_acc: 0.7230
    Epoch 77/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1198 - acc: 0.5780 - val_loss: 0.8620 - val_acc: 0.7230
    Epoch 78/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1135 - acc: 0.5764 - val_loss: 0.8546 - val_acc: 0.7260
    Epoch 79/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1095 - acc: 0.5825 - val_loss: 0.8499 - val_acc: 0.7270
    Epoch 80/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.1155 - acc: 0.5756 - val_loss: 0.8452 - val_acc: 0.7200
    Epoch 81/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0909 - acc: 0.5839 - val_loss: 0.8383 - val_acc: 0.7290
    Epoch 82/200
    7500/7500 [==============================] - 0s 22us/step - loss: 1.0916 - acc: 0.5887 - val_loss: 0.8317 - val_acc: 0.7300
    Epoch 83/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.0772 - acc: 0.5943 - val_loss: 0.8243 - val_acc: 0.7280
    Epoch 84/200
    7500/7500 [==============================] - 0s 22us/step - loss: 1.0717 - acc: 0.5925 - val_loss: 0.8199 - val_acc: 0.7350
    Epoch 85/200
    7500/7500 [==============================] - 0s 23us/step - loss: 1.0726 - acc: 0.5956 - val_loss: 0.8128 - val_acc: 0.7370
    Epoch 86/200
    7500/7500 [==============================] - 0s 22us/step - loss: 1.0700 - acc: 0.6005 - val_loss: 0.8095 - val_acc: 0.7390
    Epoch 87/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0636 - acc: 0.5981 - val_loss: 0.8034 - val_acc: 0.7370
    Epoch 88/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0530 - acc: 0.6021 - val_loss: 0.7973 - val_acc: 0.7350
    Epoch 89/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0480 - acc: 0.6063 - val_loss: 0.7919 - val_acc: 0.7380
    Epoch 90/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0559 - acc: 0.6041 - val_loss: 0.7897 - val_acc: 0.7400
    Epoch 91/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.0451 - acc: 0.6068 - val_loss: 0.7854 - val_acc: 0.7440
    Epoch 92/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0395 - acc: 0.6059 - val_loss: 0.7819 - val_acc: 0.7440
    Epoch 93/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0356 - acc: 0.6121 - val_loss: 0.7768 - val_acc: 0.7380
    Epoch 94/200
    7500/7500 [==============================] - 0s 18us/step - loss: 1.0296 - acc: 0.6225 - val_loss: 0.7719 - val_acc: 0.7400
    Epoch 95/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0206 - acc: 0.6171 - val_loss: 0.7664 - val_acc: 0.7410
    Epoch 96/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0225 - acc: 0.6204 - val_loss: 0.7641 - val_acc: 0.7450
    Epoch 97/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0141 - acc: 0.6251 - val_loss: 0.7618 - val_acc: 0.7420
    Epoch 98/200
    7500/7500 [==============================] - 0s 19us/step - loss: 1.0079 - acc: 0.6232 - val_loss: 0.7583 - val_acc: 0.7430
    Epoch 99/200
    7500/7500 [==============================] - 0s 20us/step - loss: 1.0132 - acc: 0.6229 - val_loss: 0.7548 - val_acc: 0.7430
    Epoch 100/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9964 - acc: 0.6163 - val_loss: 0.7488 - val_acc: 0.7450
    Epoch 101/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9813 - acc: 0.6339 - val_loss: 0.7433 - val_acc: 0.7420
    Epoch 102/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9743 - acc: 0.6317 - val_loss: 0.7390 - val_acc: 0.7460
    Epoch 103/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9870 - acc: 0.6327 - val_loss: 0.7366 - val_acc: 0.7450
    Epoch 104/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9703 - acc: 0.6345 - val_loss: 0.7324 - val_acc: 0.7480
    Epoch 105/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9700 - acc: 0.6388 - val_loss: 0.7320 - val_acc: 0.7470
    Epoch 106/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9727 - acc: 0.6329 - val_loss: 0.7256 - val_acc: 0.7540
    Epoch 107/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9710 - acc: 0.6343 - val_loss: 0.7243 - val_acc: 0.7510
    Epoch 108/200
    7500/7500 [==============================] - 0s 21us/step - loss: 0.9628 - acc: 0.6392 - val_loss: 0.7204 - val_acc: 0.7520
    Epoch 109/200
    7500/7500 [==============================] - 0s 21us/step - loss: 0.9580 - acc: 0.6451 - val_loss: 0.7170 - val_acc: 0.7530
    Epoch 110/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9702 - acc: 0.6317 - val_loss: 0.7160 - val_acc: 0.7540
    Epoch 111/200
    7500/7500 [==============================] - 0s 21us/step - loss: 0.9601 - acc: 0.6429 - val_loss: 0.7129 - val_acc: 0.7500
    Epoch 112/200
    7500/7500 [==============================] - 0s 21us/step - loss: 0.9536 - acc: 0.6401 - val_loss: 0.7083 - val_acc: 0.7570
    Epoch 113/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9368 - acc: 0.6439 - val_loss: 0.7040 - val_acc: 0.7540
    Epoch 114/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9465 - acc: 0.6480 - val_loss: 0.7026 - val_acc: 0.7570
    Epoch 115/200
    7500/7500 [==============================] - 0s 22us/step - loss: 0.9605 - acc: 0.6433 - val_loss: 0.7028 - val_acc: 0.7560
    Epoch 116/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9476 - acc: 0.6379 - val_loss: 0.6993 - val_acc: 0.7600
    Epoch 117/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9416 - acc: 0.6448 - val_loss: 0.6982 - val_acc: 0.7560
    Epoch 118/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9288 - acc: 0.6488 - val_loss: 0.6943 - val_acc: 0.7570
    Epoch 119/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9342 - acc: 0.6436 - val_loss: 0.6922 - val_acc: 0.7620
    Epoch 120/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9258 - acc: 0.6537 - val_loss: 0.6903 - val_acc: 0.7540
    Epoch 121/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.9308 - acc: 0.6511 - val_loss: 0.6881 - val_acc: 0.7570
    Epoch 122/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9253 - acc: 0.6573 - val_loss: 0.6865 - val_acc: 0.7570
    Epoch 123/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9111 - acc: 0.6545 - val_loss: 0.6842 - val_acc: 0.7580
    Epoch 124/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9187 - acc: 0.6604 - val_loss: 0.6829 - val_acc: 0.7570
    Epoch 125/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9143 - acc: 0.6593 - val_loss: 0.6781 - val_acc: 0.7610
    Epoch 126/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.9084 - acc: 0.6652 - val_loss: 0.6756 - val_acc: 0.7620
    Epoch 127/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8999 - acc: 0.6572 - val_loss: 0.6748 - val_acc: 0.7610
    Epoch 128/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.9022 - acc: 0.6643 - val_loss: 0.6726 - val_acc: 0.7620
    Epoch 129/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8974 - acc: 0.6639 - val_loss: 0.6725 - val_acc: 0.7610
    Epoch 130/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8816 - acc: 0.6672 - val_loss: 0.6687 - val_acc: 0.7620
    Epoch 131/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8949 - acc: 0.6667 - val_loss: 0.6665 - val_acc: 0.7650
    Epoch 132/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8908 - acc: 0.6628 - val_loss: 0.6647 - val_acc: 0.7640
    Epoch 133/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8916 - acc: 0.6684 - val_loss: 0.6637 - val_acc: 0.7620
    Epoch 134/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8732 - acc: 0.6749 - val_loss: 0.6589 - val_acc: 0.7630
    Epoch 135/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8817 - acc: 0.6731 - val_loss: 0.6584 - val_acc: 0.7670
    Epoch 136/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8798 - acc: 0.6665 - val_loss: 0.6576 - val_acc: 0.7670
    Epoch 137/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8723 - acc: 0.6759 - val_loss: 0.6553 - val_acc: 0.7670
    Epoch 138/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8853 - acc: 0.6675 - val_loss: 0.6565 - val_acc: 0.7610
    Epoch 139/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8740 - acc: 0.6724 - val_loss: 0.6517 - val_acc: 0.7680
    Epoch 140/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8615 - acc: 0.6800 - val_loss: 0.6491 - val_acc: 0.7710
    Epoch 141/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8684 - acc: 0.6759 - val_loss: 0.6487 - val_acc: 0.7700
    Epoch 142/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8462 - acc: 0.6769 - val_loss: 0.6448 - val_acc: 0.7710
    Epoch 143/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8493 - acc: 0.6764 - val_loss: 0.6441 - val_acc: 0.7740
    Epoch 144/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8806 - acc: 0.6729 - val_loss: 0.6448 - val_acc: 0.7700
    Epoch 145/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8598 - acc: 0.6787 - val_loss: 0.6446 - val_acc: 0.7710
    Epoch 146/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8612 - acc: 0.6787 - val_loss: 0.6427 - val_acc: 0.7710
    Epoch 147/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8461 - acc: 0.6792 - val_loss: 0.6427 - val_acc: 0.7720
    Epoch 148/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8538 - acc: 0.6809 - val_loss: 0.6400 - val_acc: 0.7700
    Epoch 149/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8544 - acc: 0.6812 - val_loss: 0.6387 - val_acc: 0.7720
    Epoch 150/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8582 - acc: 0.6829 - val_loss: 0.6382 - val_acc: 0.7690
    Epoch 151/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8322 - acc: 0.6880 - val_loss: 0.6361 - val_acc: 0.7740
    Epoch 152/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8284 - acc: 0.6901 - val_loss: 0.6325 - val_acc: 0.7730
    Epoch 153/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8297 - acc: 0.6951 - val_loss: 0.6303 - val_acc: 0.7710
    Epoch 154/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8298 - acc: 0.6935 - val_loss: 0.6284 - val_acc: 0.7790
    Epoch 155/200
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8395 - acc: 0.6875 - val_loss: 0.6274 - val_acc: 0.7710
    Epoch 156/200
    7500/7500 [==============================] - 0s 22us/step - loss: 0.8333 - acc: 0.6923 - val_loss: 0.6264 - val_acc: 0.7730
    Epoch 157/200
    7500/7500 [==============================] - 0s 21us/step - loss: 0.8220 - acc: 0.6969 - val_loss: 0.6248 - val_acc: 0.7740
    Epoch 158/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8409 - acc: 0.6872 - val_loss: 0.6247 - val_acc: 0.7720
    Epoch 159/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8299 - acc: 0.6949 - val_loss: 0.6255 - val_acc: 0.7740
    Epoch 160/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8299 - acc: 0.6928 - val_loss: 0.6235 - val_acc: 0.7750
    Epoch 161/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8301 - acc: 0.6943 - val_loss: 0.6234 - val_acc: 0.7790
    Epoch 162/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.8075 - acc: 0.7037 - val_loss: 0.6197 - val_acc: 0.7760
    Epoch 163/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8180 - acc: 0.6931 - val_loss: 0.6192 - val_acc: 0.7800
    Epoch 164/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8035 - acc: 0.6971 - val_loss: 0.6182 - val_acc: 0.7750
    Epoch 165/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8055 - acc: 0.6956 - val_loss: 0.6164 - val_acc: 0.7740
    Epoch 166/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8248 - acc: 0.6911 - val_loss: 0.6177 - val_acc: 0.7750
    Epoch 167/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8188 - acc: 0.6957 - val_loss: 0.6168 - val_acc: 0.7750
    Epoch 168/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8145 - acc: 0.6939 - val_loss: 0.6164 - val_acc: 0.7720
    Epoch 169/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8132 - acc: 0.6977 - val_loss: 0.6143 - val_acc: 0.7760
    Epoch 170/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.8000 - acc: 0.7032 - val_loss: 0.6128 - val_acc: 0.7740
    Epoch 171/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7947 - acc: 0.6999 - val_loss: 0.6128 - val_acc: 0.7760
    Epoch 172/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7933 - acc: 0.6993 - val_loss: 0.6106 - val_acc: 0.7750
    Epoch 173/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7968 - acc: 0.7000 - val_loss: 0.6102 - val_acc: 0.7790
    Epoch 174/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.8083 - acc: 0.6997 - val_loss: 0.6082 - val_acc: 0.7780
    Epoch 175/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7887 - acc: 0.7020 - val_loss: 0.6092 - val_acc: 0.7810
    Epoch 176/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7854 - acc: 0.7083 - val_loss: 0.6082 - val_acc: 0.7810
    Epoch 177/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7763 - acc: 0.7116 - val_loss: 0.6063 - val_acc: 0.7760
    Epoch 178/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7865 - acc: 0.7017 - val_loss: 0.6063 - val_acc: 0.7770
    Epoch 179/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7852 - acc: 0.7145 - val_loss: 0.6058 - val_acc: 0.7760
    Epoch 180/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7945 - acc: 0.6997 - val_loss: 0.6056 - val_acc: 0.7790
    Epoch 181/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7897 - acc: 0.7055 - val_loss: 0.6041 - val_acc: 0.7790
    Epoch 182/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7765 - acc: 0.7017 - val_loss: 0.6018 - val_acc: 0.7780
    Epoch 183/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7888 - acc: 0.7088 - val_loss: 0.6019 - val_acc: 0.7800
    Epoch 184/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7684 - acc: 0.7144 - val_loss: 0.6000 - val_acc: 0.7810
    Epoch 185/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7768 - acc: 0.7099 - val_loss: 0.5987 - val_acc: 0.7790
    Epoch 186/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7712 - acc: 0.7109 - val_loss: 0.5984 - val_acc: 0.7800
    Epoch 187/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7652 - acc: 0.7101 - val_loss: 0.5984 - val_acc: 0.7810
    Epoch 188/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7751 - acc: 0.7125 - val_loss: 0.5988 - val_acc: 0.7780
    Epoch 189/200
    7500/7500 [==============================] - 0s 20us/step - loss: 0.7703 - acc: 0.7132 - val_loss: 0.5969 - val_acc: 0.7770
    Epoch 190/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7669 - acc: 0.7085 - val_loss: 0.5957 - val_acc: 0.7760
    Epoch 191/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7584 - acc: 0.7080 - val_loss: 0.5958 - val_acc: 0.7820
    Epoch 192/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7460 - acc: 0.7177 - val_loss: 0.5928 - val_acc: 0.7820
    Epoch 193/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7613 - acc: 0.7211 - val_loss: 0.5910 - val_acc: 0.7820
    Epoch 194/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7572 - acc: 0.7164 - val_loss: 0.5921 - val_acc: 0.7840
    Epoch 195/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7562 - acc: 0.7120 - val_loss: 0.5907 - val_acc: 0.7820
    Epoch 196/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7629 - acc: 0.7187 - val_loss: 0.5910 - val_acc: 0.7800
    Epoch 197/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7484 - acc: 0.7188 - val_loss: 0.5890 - val_acc: 0.7820
    Epoch 198/200
    7500/7500 [==============================] - 0s 19us/step - loss: 0.7539 - acc: 0.7197 - val_loss: 0.5899 - val_acc: 0.7830
    Epoch 199/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7448 - acc: 0.7137 - val_loss: 0.5892 - val_acc: 0.7850
    Epoch 200/200
    7500/7500 [==============================] - 0s 18us/step - loss: 0.7518 - acc: 0.7175 - val_loss: 0.5901 - val_acc: 0.7850



```python
# __SOLUTION__ 
results_train = model.evaluate(train_final, label_train_final)
results_test = model.evaluate(X_test, y_test)
```

    7500/7500 [==============================] - 0s 22us/step
    1500/1500 [==============================] - 0s 22us/step



```python
# __SOLUTION__ 
results_train
```




    [0.4718774256070455, 0.8294666666348776]




```python
# __SOLUTION__ 
results_test
```




    [0.6141259970664978, 0.7619999996821085]



You can see here that the validation performance has improved again! The variance did become higher again compared to L1-regularization.

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


```python
results_train = model.evaluate(train_final, label_train_final)
results_test = model.evaluate(test, label_test)
```


```python
results_train
```


```python
results_test
```


```python
# __SOLUTION__ 
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
# __SOLUTION__ 
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
    33000/33000 [==============================] - 1s 18us/step - loss: 1.9124 - acc: 0.1945 - val_loss: 1.8530 - val_acc: 0.2520
    Epoch 2/120
    33000/33000 [==============================] - 0s 12us/step - loss: 1.7752 - acc: 0.3128 - val_loss: 1.6854 - val_acc: 0.3737
    Epoch 3/120
    33000/33000 [==============================] - 0s 12us/step - loss: 1.5886 - acc: 0.4248 - val_loss: 1.4948 - val_acc: 0.4693
    Epoch 4/120
    33000/33000 [==============================] - 0s 12us/step - loss: 1.3949 - acc: 0.5276 - val_loss: 1.3089 - val_acc: 0.5733
    Epoch 5/120
    33000/33000 [==============================] - 0s 13us/step - loss: 1.2185 - acc: 0.6142 - val_loss: 1.1536 - val_acc: 0.6440
    Epoch 6/120
    33000/33000 [==============================] - 0s 15us/step - loss: 1.0738 - acc: 0.6624 - val_loss: 1.0292 - val_acc: 0.6740
    Epoch 7/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.9595 - acc: 0.6946 - val_loss: 0.9336 - val_acc: 0.6857
    Epoch 8/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.8732 - acc: 0.7142 - val_loss: 0.8624 - val_acc: 0.7050
    Epoch 9/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.8087 - acc: 0.7277 - val_loss: 0.8095 - val_acc: 0.7193
    Epoch 10/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.7605 - acc: 0.7379 - val_loss: 0.7718 - val_acc: 0.7300
    Epoch 11/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.7237 - acc: 0.7471 - val_loss: 0.7406 - val_acc: 0.7353
    Epoch 12/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.6951 - acc: 0.7532 - val_loss: 0.7173 - val_acc: 0.7390
    Epoch 13/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.6715 - acc: 0.7608 - val_loss: 0.7006 - val_acc: 0.7413
    Epoch 14/120
    33000/33000 [==============================] - 1s 15us/step - loss: 0.6525 - acc: 0.7653 - val_loss: 0.6858 - val_acc: 0.7510
    Epoch 15/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.6360 - acc: 0.7699 - val_loss: 0.6728 - val_acc: 0.7567
    Epoch 16/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.6219 - acc: 0.7745 - val_loss: 0.6616 - val_acc: 0.7543
    Epoch 17/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.6093 - acc: 0.7801 - val_loss: 0.6544 - val_acc: 0.7567
    Epoch 18/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.5978 - acc: 0.7833 - val_loss: 0.6441 - val_acc: 0.7593
    Epoch 19/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.5876 - acc: 0.7858 - val_loss: 0.6390 - val_acc: 0.7633
    Epoch 20/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.5783 - acc: 0.7900 - val_loss: 0.6321 - val_acc: 0.7663
    Epoch 21/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.5696 - acc: 0.7926 - val_loss: 0.6254 - val_acc: 0.7650
    Epoch 22/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.5614 - acc: 0.7961 - val_loss: 0.6197 - val_acc: 0.7667
    Epoch 23/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.5537 - acc: 0.7987 - val_loss: 0.6182 - val_acc: 0.7733
    Epoch 24/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.5468 - acc: 0.8009 - val_loss: 0.6116 - val_acc: 0.7750
    Epoch 25/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.5398 - acc: 0.8041 - val_loss: 0.6085 - val_acc: 0.7753
    Epoch 26/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.5333 - acc: 0.8068 - val_loss: 0.6040 - val_acc: 0.7773
    Epoch 27/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.5272 - acc: 0.8090 - val_loss: 0.6011 - val_acc: 0.7770
    Epoch 28/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.5212 - acc: 0.8095 - val_loss: 0.5966 - val_acc: 0.7813
    Epoch 29/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.5158 - acc: 0.8141 - val_loss: 0.5945 - val_acc: 0.7827
    Epoch 30/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.5100 - acc: 0.8160 - val_loss: 0.5934 - val_acc: 0.7837
    Epoch 31/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.5052 - acc: 0.8176 - val_loss: 0.5910 - val_acc: 0.7843
    Epoch 32/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.5000 - acc: 0.8202 - val_loss: 0.5868 - val_acc: 0.7890
    Epoch 33/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4952 - acc: 0.8208 - val_loss: 0.5849 - val_acc: 0.7893
    Epoch 34/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4906 - acc: 0.8222 - val_loss: 0.5831 - val_acc: 0.7890
    Epoch 35/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4862 - acc: 0.8248 - val_loss: 0.5819 - val_acc: 0.7893
    Epoch 36/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4821 - acc: 0.8269 - val_loss: 0.5785 - val_acc: 0.7907
    Epoch 37/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4773 - acc: 0.8290 - val_loss: 0.5779 - val_acc: 0.7930
    Epoch 38/120
    33000/33000 [==============================] - 0s 11us/step - loss: 0.4735 - acc: 0.8299 - val_loss: 0.5753 - val_acc: 0.7957
    Epoch 39/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4693 - acc: 0.8311 - val_loss: 0.5742 - val_acc: 0.7990
    Epoch 40/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4657 - acc: 0.8339 - val_loss: 0.5732 - val_acc: 0.7963
    Epoch 41/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4618 - acc: 0.8348 - val_loss: 0.5725 - val_acc: 0.7947
    Epoch 42/120
    33000/33000 [==============================] - 0s 11us/step - loss: 0.4586 - acc: 0.8352 - val_loss: 0.5702 - val_acc: 0.8000
    Epoch 43/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4548 - acc: 0.8374 - val_loss: 0.5691 - val_acc: 0.7970
    Epoch 44/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4513 - acc: 0.8385 - val_loss: 0.5707 - val_acc: 0.8013
    Epoch 45/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4482 - acc: 0.8395 - val_loss: 0.5692 - val_acc: 0.7950
    Epoch 46/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4447 - acc: 0.8410 - val_loss: 0.5659 - val_acc: 0.8003
    Epoch 47/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4416 - acc: 0.8423 - val_loss: 0.5657 - val_acc: 0.7983
    Epoch 48/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.4389 - acc: 0.8426 - val_loss: 0.5654 - val_acc: 0.8013
    Epoch 49/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.4357 - acc: 0.8440 - val_loss: 0.5634 - val_acc: 0.8003
    Epoch 50/120
    33000/33000 [==============================] - 0s 15us/step - loss: 0.4328 - acc: 0.8458 - val_loss: 0.5629 - val_acc: 0.8020
    Epoch 51/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4299 - acc: 0.8477 - val_loss: 0.5643 - val_acc: 0.8020
    Epoch 52/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4272 - acc: 0.8483 - val_loss: 0.5629 - val_acc: 0.8030
    Epoch 53/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4246 - acc: 0.8490 - val_loss: 0.5614 - val_acc: 0.8047
    Epoch 54/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4218 - acc: 0.8500 - val_loss: 0.5611 - val_acc: 0.7993
    Epoch 55/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4194 - acc: 0.8502 - val_loss: 0.5607 - val_acc: 0.8037
    Epoch 56/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4169 - acc: 0.8510 - val_loss: 0.5616 - val_acc: 0.8037
    Epoch 57/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4144 - acc: 0.8525 - val_loss: 0.5635 - val_acc: 0.8027
    Epoch 58/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4118 - acc: 0.8529 - val_loss: 0.5621 - val_acc: 0.8003
    Epoch 59/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4093 - acc: 0.8552 - val_loss: 0.5637 - val_acc: 0.8040
    Epoch 60/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.4070 - acc: 0.8550 - val_loss: 0.5586 - val_acc: 0.8050
    Epoch 61/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4048 - acc: 0.8568 - val_loss: 0.5604 - val_acc: 0.8033
    Epoch 62/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4025 - acc: 0.8572 - val_loss: 0.5601 - val_acc: 0.8007
    Epoch 63/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.4001 - acc: 0.8585 - val_loss: 0.5605 - val_acc: 0.8013
    Epoch 64/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3985 - acc: 0.8589 - val_loss: 0.5612 - val_acc: 0.8030
    Epoch 65/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3961 - acc: 0.8594 - val_loss: 0.5606 - val_acc: 0.8040
    Epoch 66/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3941 - acc: 0.8602 - val_loss: 0.5612 - val_acc: 0.8017
    Epoch 67/120
    33000/33000 [==============================] - 0s 11us/step - loss: 0.3922 - acc: 0.8618 - val_loss: 0.5628 - val_acc: 0.8030
    Epoch 68/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3900 - acc: 0.8621 - val_loss: 0.5608 - val_acc: 0.8037
    Epoch 69/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3877 - acc: 0.8625 - val_loss: 0.5615 - val_acc: 0.8013
    Epoch 70/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3861 - acc: 0.8635 - val_loss: 0.5600 - val_acc: 0.8040
    Epoch 71/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3840 - acc: 0.8638 - val_loss: 0.5607 - val_acc: 0.8020
    Epoch 72/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3824 - acc: 0.8654 - val_loss: 0.5610 - val_acc: 0.8037
    Epoch 73/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3803 - acc: 0.8666 - val_loss: 0.5620 - val_acc: 0.8027
    Epoch 74/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3786 - acc: 0.8664 - val_loss: 0.5627 - val_acc: 0.8040
    Epoch 75/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3768 - acc: 0.8671 - val_loss: 0.5617 - val_acc: 0.8027
    Epoch 76/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3749 - acc: 0.8672 - val_loss: 0.5609 - val_acc: 0.8027
    Epoch 77/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3734 - acc: 0.8685 - val_loss: 0.5616 - val_acc: 0.8030
    Epoch 78/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3714 - acc: 0.8692 - val_loss: 0.5626 - val_acc: 0.8020
    Epoch 79/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3698 - acc: 0.8699 - val_loss: 0.5629 - val_acc: 0.8017
    Epoch 80/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3682 - acc: 0.8705 - val_loss: 0.5638 - val_acc: 0.8033
    Epoch 81/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3665 - acc: 0.8699 - val_loss: 0.5640 - val_acc: 0.8020
    Epoch 82/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3648 - acc: 0.8712 - val_loss: 0.5654 - val_acc: 0.8027
    Epoch 83/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3632 - acc: 0.8718 - val_loss: 0.5637 - val_acc: 0.8037
    Epoch 84/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3617 - acc: 0.8726 - val_loss: 0.5648 - val_acc: 0.8017
    Epoch 85/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3602 - acc: 0.8729 - val_loss: 0.5649 - val_acc: 0.8017
    Epoch 86/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3584 - acc: 0.8733 - val_loss: 0.5662 - val_acc: 0.8020
    Epoch 87/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3570 - acc: 0.8744 - val_loss: 0.5649 - val_acc: 0.8007
    Epoch 88/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.3559 - acc: 0.8744 - val_loss: 0.5655 - val_acc: 0.8020
    Epoch 89/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3540 - acc: 0.8755 - val_loss: 0.5676 - val_acc: 0.8017
    Epoch 90/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3525 - acc: 0.8758 - val_loss: 0.5717 - val_acc: 0.7987
    Epoch 91/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3512 - acc: 0.8755 - val_loss: 0.5686 - val_acc: 0.8020
    Epoch 92/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3496 - acc: 0.8771 - val_loss: 0.5693 - val_acc: 0.8000
    Epoch 93/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3483 - acc: 0.8779 - val_loss: 0.5720 - val_acc: 0.8027
    Epoch 94/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3467 - acc: 0.8791 - val_loss: 0.5730 - val_acc: 0.8013
    Epoch 95/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3455 - acc: 0.8780 - val_loss: 0.5718 - val_acc: 0.7983
    Epoch 96/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3440 - acc: 0.8797 - val_loss: 0.5699 - val_acc: 0.7990
    Epoch 97/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3430 - acc: 0.8788 - val_loss: 0.5717 - val_acc: 0.8013
    Epoch 98/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3416 - acc: 0.8792 - val_loss: 0.5729 - val_acc: 0.8023
    Epoch 99/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3402 - acc: 0.8803 - val_loss: 0.5725 - val_acc: 0.8020
    Epoch 100/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.3387 - acc: 0.8811 - val_loss: 0.5755 - val_acc: 0.7980
    Epoch 101/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3376 - acc: 0.8815 - val_loss: 0.5743 - val_acc: 0.8013
    Epoch 102/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.3362 - acc: 0.8818 - val_loss: 0.5754 - val_acc: 0.7997
    Epoch 103/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3351 - acc: 0.8823 - val_loss: 0.5757 - val_acc: 0.8030
    Epoch 104/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.3336 - acc: 0.8834 - val_loss: 0.5770 - val_acc: 0.8013
    Epoch 105/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.3324 - acc: 0.8843 - val_loss: 0.5790 - val_acc: 0.8017
    Epoch 106/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.3309 - acc: 0.8838 - val_loss: 0.5791 - val_acc: 0.7990
    Epoch 107/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.3299 - acc: 0.8845 - val_loss: 0.5789 - val_acc: 0.8017
    Epoch 108/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3286 - acc: 0.8842 - val_loss: 0.5801 - val_acc: 0.8013
    Epoch 109/120
    33000/33000 [==============================] - 0s 14us/step - loss: 0.3272 - acc: 0.8852 - val_loss: 0.5824 - val_acc: 0.8010
    Epoch 110/120
    33000/33000 [==============================] - 0s 13us/step - loss: 0.3260 - acc: 0.8856 - val_loss: 0.5830 - val_acc: 0.7993
    Epoch 111/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3251 - acc: 0.8861 - val_loss: 0.5832 - val_acc: 0.7993
    Epoch 112/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3241 - acc: 0.8866 - val_loss: 0.5842 - val_acc: 0.7987
    Epoch 113/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3226 - acc: 0.8877 - val_loss: 0.5821 - val_acc: 0.8000
    Epoch 114/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3212 - acc: 0.8879 - val_loss: 0.5864 - val_acc: 0.7953
    Epoch 115/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3203 - acc: 0.8877 - val_loss: 0.5879 - val_acc: 0.7990
    Epoch 116/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3194 - acc: 0.8880 - val_loss: 0.5855 - val_acc: 0.8023
    Epoch 117/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3180 - acc: 0.8889 - val_loss: 0.5900 - val_acc: 0.7987
    Epoch 118/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3165 - acc: 0.8904 - val_loss: 0.5908 - val_acc: 0.7973
    Epoch 119/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3155 - acc: 0.8896 - val_loss: 0.5912 - val_acc: 0.7990
    Epoch 120/120
    33000/33000 [==============================] - 0s 12us/step - loss: 0.3143 - acc: 0.8914 - val_loss: 0.5912 - val_acc: 0.7970



```python
# __SOLUTION__ 
results_train = model.evaluate(train_final, label_train_final)
results_test = model.evaluate(test, label_test)
```

    33000/33000 [==============================] - 1s 20us/step
    4000/4000 [==============================] - 0s 20us/step



```python
# __SOLUTION__ 
results_train
```




    [0.31041816379807213, 0.8919393939393939]




```python
# __SOLUTION__ 
results_test
```




    [0.5526078445911408, 0.812]



With the same amount of epochs, you were able to get a fairly similar validation accuracy of 89.67 (compared to 88.45 in obtained in the first model in this lab). Your test set accuracy went up from 75.8 to a staggering 80.225% though, without any other regularization technique. You can still consider early stopping, L1, L2 and dropout here. It's clear that having more data has a strong impact on model performance!

## Additional Resources

* https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb
* https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
* https://catalog.data.gov/dataset/consumer-complaint-database

## Summary  

In this lesson, you not only built an initial deep-learning model, you then used a validation set to tune your model using various types of regularization.
