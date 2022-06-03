---
layout: default
title: Assignment 08
id: ass08
---


# Assignment 8: Autoregressive Models
**Discussion: June 10th**
**Deadline: June 9th, 23:59**

In this assignment, we will try to implement simple versions of
autoregressive models like PixelCNN.


## Autoregressive Modeling

It is possible to view an image as a sequence of pixel values, and set up a
generative process accordingly. The inductive bias of this model is questionable,
but the practical results are rather strong. In the course repository, you can
find a notebook that applies this idea in a very naive fashion to the MNIST dataset.
This model ignores the fact that images have rows and columns and thus tends to
generate outputs in the wrong place. This also means that, for example, the
pixel directly above another one is treated as farther away than the one directly
to the left. Overall, it's just not a good model! Instead, let's use ideas from
the [PixelRNN paper](https://arxiv.org/pdf/1601.06759.pdf), in particular 
PixelCNN.

PixelCNN is relatively simple to implement by using a _masked_ convolutional
layer, zeroing out certain components. We need to do 
this because the network cannot look into the future, i.e. it cannot make use
of pixels that have not been generated yet (according to whatever generation
order is being used). See figure 4 in the paper, or figure 1 in 
[the follow-up](https://arxiv.org/pdf/1606.05328.pdf).
 You can find sample code in the course repository. Use
this (or your own implementation) to build a stack of convolutional layers,
being mindful of the differences between _type A_ and _type B_ masks. The
final layer should have 256 output units and acts as the softmax predictor just
like in the RNN code. Training is straightforward -- the target is equal to the
input (but note that you may want to scale inputs to [0, 1] whereas targets should
be category indices from 0 to 255)! Because the layers are masked such that the
central pixel is not looked at, this works out just right -- we are predicting
the central pixel from those above and to the left of it.

With the model trained (which can take a while), you can use it for generation!
This proceeds sequentially, one pixel after the other, and is very slow. There
is sample code in the repo, but this is not optimized very well, so you may be
able to improve on it. Hopefully, your results are better than with the RNN!

If you get bad results:
- Use large models and train for a long time; it can take thousands of steps
until you get proper results, and a better likelihood might not mean better
samples.
- Consider using residual connections (see figure 5
in the paper), which should improve performance considerably.
- You can also try
other changes from the follow-up (linked above) or... 
[the follow-follow-up](https://arxiv.org/pdf/1701.05517.pdf).  
- A note of caution on 1x1 convolutions and residual connections: It's easy to
accidentally propagate information from the center pixel all the way to the top
of the network (1x1 conv obviously shouldn't use masks, and the residual
connections go around the masked convolutions). To avoid "leakage", make sure
that the first layer of your network is a pure masked convolution with no
residual connections!
- In a similar vein, normalization methods such as batchnorm are _technically_
off-limits since the cause _data leakage_: When predicting a pixel, no information
from the bottom or right should be used, but most data-based normalization methods
(batchnorm, layernorm, instancenorm etc.) normalize using the _whole_ image.
This can also lead to problems in generation, as here you might start with a
full-size, but empty, image, which has completely different statistics than
those seen in training! This is not a problem with batchnorm though, since it uses
the statistics accumulated during training.
- Despite the above point, feel free to use batchnorm if it helps with the model, but
keep this issue in mind. From experience, simply removing it seems to actually
lead to better generation results, and normalization is not as necessary with
residual connections.

Also: Transferring autoregressive models to color datasets is a bit 
more complicated because of
how the masks interact with the color channels. In the PixelRNN paper, every
color value is generated one-by-one, which complicates the masks. However, it
should also be possible to sample all color values of one pixel independently in
parallel, although technically this weakens the model. The easiest thing is to
just use grayscale datasets -- you can also convert color datasets such as Flickr Faces
to grayscale using `tf.image.rgb_to_grayscale`.
