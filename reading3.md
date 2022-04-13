---
layout: default
title: Reading 3
id: reading3
---


## Reading Assignment 3: Approximate Inference, DBNs & DBMs

[Chapter 19](http://www.deeplearningbook.org/contents/inference.html)  
(complete) with focus on:
- Why is inference hard/intractable?
- ELBO, pros/cons, conditions under which the bound is tight
- (optional) Expectation-Maximization algorithm (approach & pros/cons)
- (19.3 is optional) Maximum a Posteriori inference (approach & pros/cons)
- What is variational inference/variational learning?
- (19.4.1-19.4.3 can be skipped)
- How does using approximate inference influence the learning?
- Wake-Sleep algorithm
- What is an Inference Network and where is it needed?
- Inference network vs mean field fixed point equations

[Chapter 20.3 on Deep Belief Networks](http://www.deeplearningbook.org/contents/generative_models.html)  
with focus on:

- Structure
- Layer-wise pre-training
- Generative (wake-sleep) and discriminative (DBN to MLP) fine-tuning
- Use cases, pros & cons


_Bonus:_   
[Chapter 20.4-20.4.4 on Deep Boltzmann Machines](http://www.deeplearningbook.org/contents/generative_models.html)  
with focus on:
- Variational Learning with Mean Field Approach (especially for DBM training)
- What is the difference between mean-field fixed-point updates and gibbs sampling?

**Here are some additional/alternative resources, if you want some more visual explanations than the book.**

_DBN_

- Salakutinov & Murray (2008) :
  - Section 4 on Deep Belief Network learning in  
    Paper: [On the Quantitative Analysis of Deep Belief Networks](https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf)    

- Hinton:  
  - Video: [Wake-Sleep algorithm](https://youtu.be/VKpc_z7b9I0)
  - Video: [Stacking RBMs (to a DBN)](https://youtu.be/cYAlxZI-VM8)
  - Video: [DBN discriminative fine-tuning](https://youtu.be/0gX4BtLSqIY)    
  - Paper: [Up-Down Algorithm as contrastive wake-sleep](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)  

- Hugo Larochelle:
  - Video: [Deep Belief Network](https://youtu.be/vkb6AWYXZ5I)
  - Video: [Variational Bound](https://youtu.be/pStDscJh2Wo)
  - Video: [DBN pre-training](https://youtu.be/35MUlYCColk)

_Bonus: DBM training_
- Salakhutdinov & Hinton (2009) [Deep Boltzmann Machines](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf)
- Salakhutdinov & Larochelle (2010) [Efficient Learning of Deep Boltzmann Machines](http://proceedings.mlr.press/v9/salakhutdinov10a/salakhutdinov10a.pdf)
