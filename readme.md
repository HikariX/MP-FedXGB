# MP-FedXGB

This is a federated XGBoost project. Our project is designed using secret sharing (SS) encryption scheme. It's tightly linked to *An Efficient Learning Framework For Federated XGBoostUsing Secret Sharing And Distributed Optimization*.

This project is based on mpi4py, you should install this package first.

By executing "mpiexec -n 5 python VerticalXGBoost.py" in the directory can you start training the model. BTW, we provide the main function with properly split dataset, defined in main1, main2, and main3. Feel free to modify them in VerticalXGBoost.py.

## Trees.py

We define the base tree and FL-XGB tree classes in this file.

### setMapping

This function is designed for feature index mask. That is, other participants refer to the feature from one participant except themselves by random feature index.

### AggBucket

This function is used for bucket statistics' aggregation.

### buildTree

During the development, we implemented 4 versions of this function. The original version nearly implemented the FL-XGB proposed in *A Hybrid-Domain Framework for Secure Gradient Tree Boosting*. And **ver2** rips out division in argmax operation. In **ver3** we even implemented the model with the completely secure trick mentioned in *SecureBoost- A Lossless Federated Learning Framework*. Finally we implemented gradient descent for leaf weights computations in **ver4**.

### getInfo

This function is designed to retrieve prediction information including local indicator vector and weights.

### fit

The entrance for choosing different tree-building functions.

### classify

The function to predict one data instance, using informations obtained from **getInfo**.

### predict

The function to predict the whole set of data, calling **classify** many times.

## SSCalculation.py

We define calculations about SS in this file.

### SSSplit

**SHR** operation, splits data into secret shares.

### SMUL

**MUL** operation, does multiplications in all participants.

### SDIV

**DIV** operation, does divisions in all participants.

### SARGMAX

This function is one of the core of our design. It has 4 versions. The original one is designed with **DIV** operations. In **ver2** we remov divisions and reshape the calculation. We implement complete secure FL-XGB in **ver3** with the idea originated from *SecureBoost- A Lossless Federated Learning Framework*. Finally we proposed our First-Layer-Mask trick and implemented it in **ver4**.

### SSIGN

This function is set for judging whether the loss reduction is positive. 

This function has 2 versions. The original one collect data from different participants. These nominators and denominators are both masked with same random values. And the division result is the same as random factors are cancelled by  reduction of a fraction. For example, we want to compute (a+b+c)/(d+e+f), it's the same as (ma+mb+mc)/(md+me+mf). We mask each shared value but get the final division result.  

The **ver2** view the original problem the same as that in **SARGMAX**. That is, reduction of fractions is conducted again, and division is removed totally.

### S_GD

This is another innovation point. We take advantage of distributed optimization and gradient descent, utilizing this  problem structure into our SS calculation.

## VerticalXGBoost.py

We define the main model in this file.

### getQuantile

This function is used for getting quantile sketch for one feature in each participant. It can be finished offline (without interatcion with other participants).

### getAllQuantile

The function to get all quantile sketches in each participant.

### fit

The function to start training and update predictions after each tree.

### predict

The function to generate predictions.