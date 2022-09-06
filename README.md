# Debiasing-Facial-Recognition-Systems-MIT-DL-Lab2.2

![ezgif-2-253dfd3f9097](https://user-images.githubusercontent.com/68354896/188734878-05cbdcd7-6c63-4306-9245-31fe9847f292.gif)

This project is a part of MIT 6.S191 Introduction to Deep Learning course. In this lab, I explored two prominent aspects of applied deep learning: facial detection and algorithmic bias.

Deploying fair, unbiased AI systems is critical to their long-term acceptance. Consider the task of facial detection: given an image, is it an image of a face? This seemingly simple, but extremely important, task is subject to significant amounts of algorithmic bias among select demographics.

In this lab, I build a facial detection model that learns the latent variables underlying face image datasets and uses this to adaptively re-sample the training data, thus mitigating any biases that may be present in order to train a debiased model.

Variational autoencoder (VAE) for learning latent structure:

The accuracy of the CNN varies across the four demographics. To think about why this may be, consider the dataset the model was trained on, CelebA. If certain features, such as dark skin or hats, are rare in CelebA, the model may end up biased against these as a result of training with a biased dataset. That is to say, its classification accuracy will be worse on faces that have under-represented features, such as dark-skinned faces or faces with hats, relevative to faces with features well-represented in the training data! This is a problem.

My goal is to train a debiased version of this classifier -- one that accounts for potential disparities in feature representation within the training data. Specifically, to build a debiased facial classifier, I trained a model that learns a representation of the underlying latent space to the face training data. The model then uses this information to mitigate unwanted biases by sampling faces with rare features, like dark skin or hats, more frequently during training. The key design requirement for my model is that it can learn an encoding of the latent features in the face data in an entirely unsupervised way. To achieve this, I used variational autoencoders (VAEs).

![VAE structure](https://user-images.githubusercontent.com/68354896/188732996-7b066cb4-d21d-4981-8429-8603370450c7.jpg)

Latent loss ( L_KL ): measures how closely the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence. The equation for the latent loss is provided by:

![latent-loss](https://user-images.githubusercontent.com/68354896/188732728-dedfe29c-331c-414b-b369-8563475e80e7.png)

Reconstruction loss ( L_x(x,x^) ): measures how accurately the reconstructed outputs match the input and is given by the  L1  norm of the input image and its reconstructed output. The equation for the reconstruction loss is provided by:

![reconstruction-loss](https://user-images.githubusercontent.com/68354896/188732743-84b77442-3204-49cd-9923-a29bd2032178.png)

The equation for the VAE loss:

![VAE-loss](https://user-images.githubusercontent.com/68354896/188732807-c520eab4-71b2-46a2-b804-2b23dab97a9e.png)

where  c  is a weighting coefficient used for regularization.

Debiasing variational autoencoder (DB-VAE):

Now, I used the general idea behind the VAE architecture to build a model, termed a debiasing variational autoencoder or DB-VAE, to mitigate (potentially) unknown biases present within the training idea. I trained my DB-VAE model on the facial detection task, run the debiasing operation during training, evaluate on the PPB dataset, and compare its accuracy to our original, biased CNN model.

The key idea behind this debiasing approach is to use the latent variables learned via a VAE to adaptively re-sample the CelebA data during training. Specifically, I will alter the probability that a given image is used during training based on how often its latent features appear in the dataset. So, faces with rarer features (like dark skin, sunglasses, or hats) should become more likely to be sampled during training, while the sampling probability for faces with features that are over-represented in the training dataset should decrease (relative to uniform random sampling across the training data).

A general schematic of the DB-VAE approach is shown here:

<img width="1109" alt="DB-VAE-Schematic" src="https://user-images.githubusercontent.com/68354896/188734149-abca6347-6ead-4307-8cc0-8b6f8124abb1.png">

VAE loss ( L_VAE ): consists of the latent loss and the reconstruction loss.

Classification loss ( L_y(y,y^) ): standard cross-entropy loss for a binary classification problem.

![total-loss](https://user-images.githubusercontent.com/68354896/188734325-bb646182-11cd-463b-b897-60d39bd98d88.png)

Evaluation of DB-VAE on test dataset:

Test the DB-VAE model on the test dataset, looking specifically at its accuracy on each the "Dark Male", "Dark Female", "Light Male", and "Light Female" demographics and compare the performance of this debiased model against the (potentially biased) standard CNN.

![predic](https://user-images.githubusercontent.com/68354896/188735024-e9409141-2ec6-4a58-9582-369e27ceae68.png)


