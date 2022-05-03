---
layout: default
title: Reading 5
id: reading5
---


## Reading Assignment 5: Advanced VAE Techniques

In this reading assignment, you will develop a better understanding of the typical problems in VAE training and learn about techniques that aim to remedy them.

### (V)AEs
- Start with the nice overview blog post ["From Autoencoder to Beta-VAE"](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html) by Lilian Weng. (You can skim over the already familiar 1st half.)
- Continue with these blog posts on VAEs, their problems and variants: [Part 1](https://towardsdatascience.com/what-a-disentangled-net-we-weave-representation-learning-in-vaes-pt-1-9e5dbc205bd1)
 [Part 2](https://towardsdatascience.com/with-great-power-comes-poor-latent-codes-representation-learning-in-vaes-pt-2-57403690e92b)
- Dive deeper with three recent papers on 
  - [Preventing posterior collapse with delta-VAEs](https://arxiv.org/abs/1901.03416) - focus on the general idea
  - [Don’t Blame the ELBO! A Linear VAE Perspective on Posterior Collapse](http://papers.nips.cc/paper/9138-dont-blame-the-elbo-a-linear-vae-perspective-on-posterior-collapse) - focus on the definition and the method to measure posterior collapse
  - NVAE: A Deep Hierarchical Variational Autoencoder [paper](https://arxiv.org/abs/2007.03898) [code](https://github.com/NVlabs/NVAE) - try to gather as many good practices as possible
- Optionally, read more about [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://deepmind.com/research/publications/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2) - [Hi-Res Version](https://drive.google.com/file/d/1H2nr_Cu7OK18tRemsWn_6o5DGMNYentM/view) (already covered in the blog above)

### Further optional readings on VAEs
Read as many papers/posts as you can/want. Many are concerned with general architectural
considerations and/or "tricks", but some are about specific applications (bias
towards music). Pick those that interest you. Also, please share any interesting sources you find on the topic!

- Learn about VAEs with spherical latent spaces in [Hyperspherical Variational Auto-Encoders](https://arxiv.org/abs/1804.00891)
- If you want to work with a VQ-VAE, this paper on [Theory and Experiments on Vector Quantized Autoencoders](https://arxiv.org/abs/1805.11063) will be helpful.
- Check out the [hierarchical VQ-VAE "Jukebox" by OpenAI](https://openai.com/blog/jukebox/)!
- [Taming VAEs](https://arxiv.org/pdf/1810.00597.pdf) or [a similar shorter paper](http://bayesiandeeplearning.org/2018/papers/34.pdf)
- [InfoVAE](https://arxiv.org/abs/1706.02262)
- [Wasserstein AEs](https://arxiv.org/abs/1711.01558), along with [a follow-up](https://arxiv.org/abs/1802.03761).
Bonus question: How are InfoVAE and WAEs related?
- [VAEs for learning timbre spaces](http://ismir2018.ircam.fr/doc/pdfs/219_Paper.pdf)



### Further optional readings on VAE Hybrids (for later, when we have covered the other techniques)
- AE + GAN:
  - [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)
  - [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)
  - [Autoencoding Beyond Pixels](https://arxiv.org/abs/1512.09300)
  - [Variational Approaches for AE GANs](https://arxiv.org/pdf/1706.04987.pdf)
- AE + Autoregressive
  - [MusicVAE](https://arxiv.org/abs/1803.05428)
  - [PixelVAE](https://arxiv.org/pdf/1611.05013.pdf)
  - [Autoregressive Discrete Autoencoders for music](http://arxiv.org/abs/1806.10474)
  - [NSynth with Wavenet autoencoders](http://arxiv.org/abs/1704.01279)
- GAN + Autoregressive
  - ??? Find some :)


Feel free to share additional papers or other resources you come across!

