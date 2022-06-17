---
layout: default
title: Assignment 10
id: ass10
---


# Assignment 10: Autoregressive Models & VQ-VAE
**Discussion: June 24th**  
**Deadline: June 23rd, 23:59**

In this multi-part assignment, we will try to implement VQ-VAEs and combine them
with autoregressive models.

## VQ-VAE

Note: Some "starter code" for this is available in the course repository! Also,
feel free to refer back to [the paper](https://arxiv.org/pdf/1711.00937.pdf) at
any time.

Let's start with a relatively simple VQ-VAE. Build an autoencoder on a dataset
of your choice; this should be easy for you at this point (just take old code if
you have it!). Now, turn it into a VQ-VAE as follows:
- Add a _codebook_. This should be a `K x d` matrix, where `d` is the latent
dimension of the autoencoder, and `K` is the number of discrete codes you want
-- another hyperparameter to tune!
- Change your model slightly as follows. After the encoder produces a code, do
_not_ put this into the decoder. Instead:
  - Compute the squared difference between the code batch and all entries in the
  codebook. The result should be a `b x K` matrix (`b` is batch size). This is
  efficiently computable using broadcasting.
  - For each row in this distance matrix, find the lowest-distance entry via
  `tf.argmin`. This gives you, for each batch element, the codebook entry
  that is closest to the encoder result.
  - Take those entries from the codebook via `tf.gather`.
  - Put those codebook entries into the decoder.
  
Obviously this limits the model somewhat, since it can only produce `K` distinct
outputs. You can view this as a kind of clustering of the dataset into `K`
classes.

### Gradients
Does the above work? No? Something about no gradients for the encoder?  
...The problem is of course that the discrete `argmin` operation does not allow
for gradient flow. We can fix this via the "straight-through estimator". 
This can be implemented as `x + stop_gradient(y - x)`.
- Here, `y` is the result of the codebook lookup and `x` are the encoder outputs.
- The forward result of the above is just `y`.
- The gradients, however, are taken from `x` -- the encoder output.
- Taken together, this means that the output is that of the codebook, but the
gradients flow into the encoder.

### Losses
Finally, we need to train the VQ-VAE. The loss has several components:
- The reconstruction loss on the data.
- The codebook loss: `square(stop_gradient(encodings) - codes)` where `encodings`
are the encoder outputs and `codes` the corresponding codebook entries. This 
draws the codebook entries closer to the encoder outputs without modifying the
encoder.
- The commitment loss `square(encodings - stop_gradient(codes))`, which is
supposed to encourage the encoder to "stick with" a codebook entry and not jump
around too much.
- The KL-divergence? The way the model is interpreted (the encoder+codebook 
return one-hot distributions and the prior is assumed to be uniform), this term
is actually constant and so doesn't contribute to the gradient -- we can ignore
it for the loss.

As usual, you will need to balance these losses reasonably well. The paper
proposes scaling the commitment loss lower than the codebook loss. If your
net doesn't train properly, you might need to scale up the reconstruction loss.

### Generation
If the model trains reasonably well, generation is simple: Draw a random vector
from the codebook and put it through the decoder. This approach has issues, 
though. Encoding a large amount of images into a single set of quantized codes
doesn't work well -- you can only generate as many different outputs as you
have codebook entries! With a reasonable number (say, 256), this severely limits
model capacity and thus output quality. If you tried it like this, you likely
got _very_ blurry results.

Instead, VQ-VAEs are rather used to encode data into a _grid_ (or sequence) of
codes -- e.g. use a CNN encoder, but do _not_ flatten the data at any point.
That way, you get a _grid_ of codes. Say, 32x32 inputs, with 8x8 overall stride
and the final convolutional layer has 64 channels -- this would mean you end up
with a 4x4 "image" where each pixel is a 64-d code. Now, the idea is to quantize
each code separately -- allowing for far more variety in the outputs as now
we have 16 pixels with 256 possible values each (instead of just 256 possible
values overall -- still sticking with the example of having 256 entries in the
codebook). You should be able to reach similar reconstruction quality to a
normal autoencoder!

...how do we generate new images now? In principle, you would need to sample
one code for each of the 4x4 pixels and send this "image" through the decoder.
The problem here is that, by sampling each pixel independently, the resulting
images will lack global coherence. We need to somehow sample codes in a
dependent manner. Which brings us to...


## Putting it all together

Recall that we trained autoregressive models directly on (image) data. Turns out,
since the VQ-VAE output above is essentially an image with discrete elements 
(codebook indices), we can also train autoregressive models on those compressed
images! This means that the autoregressive model learns to generate codes that
"fit together" well, and then the VQ-VAE decompresses them back to image space.

- Train a VQ-VAE on a dataset of your choice.
- When it's done training, process the entire training dataset via the encoder.
Compute the encodings and map them to the codebook entries -- but keep
the _indices_ as well as the actual codes!
- Now, for each image in the dataset (of course you can also work on non-image
data...), you should have a smaller image (remember the encoder being a
convolutional net) of code indices, and one image of codes. Put those in a new dataset.
- Train an autoregressive model on this dataset. It should work exactly as
before -- instead of pixel values in [0, 255] we have indices in [0, K] where K
is the size of the codebook (number of entries).
  - You probably want the _inputs_ of the autoregressive model to be the
  actual codebook vectors, and only the _targets_ to be the indices. Similar to
  how the sample code for images uses integers as targets, but floats between
  [0, 1] as inputs.
  - Still, you could also experiment with using the code indices as inputs.
  Likely, you would then want the first layer of the model to be an `Embedding`.
    This would allow the autoregressive model to choose its own "interpretaition"
    of what the code indices stand for, instead of being forced to work on the
    original codes.
- Now you can sample code sequences from the autoregressive model, and put them
through the decoder of the VQ-VAE to receive images! The fact that we are now
working on encoded, downscaled images means the autoregressive model should be
easier to train, and we can probably get away with smaller/less complex models.
