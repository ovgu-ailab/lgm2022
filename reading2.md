---
layout: default
title: Reading Assignment 2
id: reading2
---


# Reading Assignment 2: Partition Function & RBMs

From the [Deep Learning Book - Chapter 18: Confronting the Partition Function](http://www.deeplearningbook.org/contents/partition.html), please read these sections:

* Sections 18, 18.1, 18.2 & 18.3 with focus on:
  * What are positive and negative phase of learning?
  * Which role does the partition function play in learning generative models?
  * How do we avoid to compute the partition function directly and why is it possible to do this?
  * How does Contrastive Divergence (CD) work and what (dis)advantages does this method have?
  * How does Persistent Contrastive Divergence (PCD) work and what (dis)advantages does this method have?
  * What is the relation between PCD and Stochastic Maximum Likelihood?
  * What is the idea behind Pseudolikelihood?

* Sections 18.7 & 18.7.1 (no details necessary):
  * What are possible ways of estimating the partition function?
  * What is the idea of Annealed Importance Sampling (AIS)?

For the actual application of these techniques, please read these sections from the [Deep Learning Book - Chapter 20: Deep Generative Models](http://www.deeplearningbook.org/contents/generative_models.html):

* Sections 20 - 20.2 with focus on:
  * What are the Energy functions of a Boltzmann Machine (BM) and a Restricted Boltzmann Machine (RBM)?
  * How would the partition function of an RBM be computed?
  * What makes sampling in an RBM efficient?
  * What is the sampling procedure and how does it relate to MCMC?
  * How can BMs and RBMs be trained?
  * Where are the connections between training an RBM and the concepts of chapter 18 (partition function, CD/PCD, AIS)

## More optional material about (R)BM training:

* [Salakutinov & Hinton (2009) Deep Boltzmann Machines](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf)  
Sections 1 and 2: general BM Learning Procedure with a simple update rule (i.e. without computing gradients of weights with tf.gradient() )

* [A theano implementation with condensed explanation using gradient on free energy](http://deeplearning.net/tutorial/rbm.html)