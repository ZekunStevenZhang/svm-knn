NCSU ECE759 course final project: 

Classfication by two algorithms svm and knn.

The datasets used in this project are MNIST and Caltech 10.

# svm-knn

1.	All code in this project is implemented by Python3.5.2

2.	There should be a data folder in the same path of these code files. In data folder, there shall be two folders--”mnist” and “Catech 10”. In mnist, there shall be four files downloaded directed from the official website of mnist data--”train_images”, ”train_labels”, “test_images”, “test_labels”. In catech 10 folder, there are 10 category folder. Inside these 10 folders are the images belong to these categories.

3.	Except some src files, like dataloader, svmmodel and knnmodel, the other files are written in jupyter notebook which stores our running results

4.	knn_mnist_cross_validation_k1-8_p4().ipynb uses 8 threads running simultaneously. Due to the different computer configuration, you may change the code to use less threads. Btw, for this file, we import _thread and threading which are not included in python2.

5.	knn_mnist_cross_validation_k31323_p123().ipynb was first used to search for the optimal parameters among all combination of k= [3, 13, 23] and p = [1, 2, 3]. But after we did cross validation on knn for caltech dataset, we find that higher order distance function may obtain better performance. So we redo cross validation on mnist knn model by knn_mnist_cross_validation_k1-8_p4().ipynb.

6.	KNN_MNIST model requires huge computation. For knn_mnist_cross_validation_k31323_p123().ipynb file, we ran almost 15 hours on it. As for knn_mnist_cross_validation_k1-8_p4().ipynb file, we still spent almost 6 hours on it, though the parallel threads saved us 87.5% time. 

7.	The variable “result” in each model means the time cost of the function, i forgot to change the name.

8.	model_retrain().ipynb is used to test the performance of the optimal hyperparameters selected by cross validation.

9.	For caltech_svm model, i made a mistake on the function name, i still used Runsvmmnist for that model. But the code is correct, it’s just a name mistake.
