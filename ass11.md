---
layout: default
title: Assignment 11
id: ass11
---


# Assignment 11: Denoising Diffusion Models
**Discussion: July 1st**  
**Deadline: June 30th, 23:59**

In this assignment, we will implement the final "class" of generative models that
have been treated in this course. Diffusion models have seen a stark rise in popularity
recently. They are stable to train and can produce very high-quality outputs.

See the reference paper [here](https://arxiv.org/pdf/2006.11239.pdf). In what follows,
we will list the main steps required for a successful implementation.


## Forward Process
First, make sure that you have understood the forward process and verified your
implementation of it.
- For reasons of statistical consistency, it is recommended to scale your data
 to the interval [-1, 1]. It could perhaps also work (or be even better?) to scale
  your data such that it has mean 0 and standard deviation of 1, just like the
  noise it will be diffused to. However, computing these statistics on the training
  set can be very costly for large datasets, whereas scaling to [-1, 1] is trivial.
- You need to decide on how many steps the diffusion process should run, as well
as a "noise schedule" `beta_t`. The paper recommends `T = 1000` steps and `beta_t`
  increasing linearly from 1e-4 to 0.02 (see top of section 4 or appendix B). 
  You can use a `linspace` to create this directly.
- You will need to run the forward process for each training step. This would be very
slow. Luckily, any time step in the process can be sampled directly; see equation 4
  in the paper. For this purpose, compute the `alpha_t` and `alpha_bar_t` as noted
  in the paper (note `np.cumprod`).
- Test the diffusion process by plotting some data points at intermediate steps 
  `t` (say, every 50 or 100 steps or so). You should be able to see how the data is slowly
  overpowered by noise.
  
## Model
To train the backward process, we need a parametric model that basically separates
the noise from the data. You can use any kind of model that takes an input with
the same shape as the data, and returns the same shape as well. Papers tend to use
U-net-like models, which are basically encoder-decoder networks with skip-connections
connecting encoder layers to decoder layers operating at the same resolution.
Building such a model should be simple in principle. There is just one complication...

### Conditioning on t
In the mathematical framework of diffusion models, our network has to be able to
return different outputs depending on where we are in the process (step `t`).
Thus, `t` needs to be given as input to the model in some shape or form, or the
model has to be otherwise conditioned on it. 

Some possible choices were discussed in the
exercise, but the large `t` we tend to deal with in diffusion models makes many
of these infeasible. The paper uses _positional encodings_ similar
to what was proposed in the original paper on Transformers. You can find sample
code in the course repository on gitlab. This codes takes a "batch" of `t` values
and returns the encoding for each, with a desired number of frequencies for the
sinusoids. You should then broadcast this over the image dimensions, i.e. create
"feature maps" that are constant over width and height, and where each dimension
of the positional encoding is one channel. Afterwards, concatenate this "constant image"
with the channels of the actual input; this is now the input to the model (you can
also set up a model with two inputs and do the concatenation etc as part of the model).

For a working model, it should suffice to add `t` only to the input, but note that
the paper proposes adding it to _each_ residual block in the model!


## Training
Training is surprisingly simple; just follow algorithm 1 on page 4 of the paper.
For each training step
- Sample a batch of data points
- Generate a batch of targets (random noise!)
- Sample a batch of `t` values; it is recommended to sample a `t` for each batch 
  element instead of using one `t` for the whole batch
- Diffuse each data point by the corresponding `t`
- Put the diffused data and `t` into the model
- Compute the squared error between the noise targets and model outputs

You may have trouble doing all these steps in tensorflow. But note that everything
before running the actual model does not need to be backpropagated through. That is,
you can do all these steps "outside" the training step, using general Python code,
numpy etc.


## Sampling
For sampling, we have to fully implement the "backward process". This is given in
algorithm 2 of the paper.
- Start with samples drawn from a standard normal distribution
- Go through the `t` steps _backwards_ (important to apply the correct conditioning
  in the model etc) and apply the formulas. Be mindful of choosing the correct
  `alpha_t`, `alpha_bar_`, correctly applying square roots (or not), etc.
- The formula mentions a `sigma_t`; possible choices for this are discussed in
the beginning of section 3.2 of the paper, e.g. `sigma_t = sqrt(beta_t)`
- Remember that when plotting samples, you should reverse the scaling from before
  (e.g. scale images back to [0, 1]).
  

That's it! Have fun with your model. Besides straight generation, you can also
do a few more things with the diffusion process:
- Given an image, only apply the diffusion up to some step (say, 250 or 500 or...)
and then revert it. That is, run the reverse process starting from that `t` instead
  of the maximum, and use the diffused image instead of a random sample as the
  initialization. You should get images that look similar to the original, but
  somewhat different (the more diffusion steps you run, the more different it will be).
- Similarly, you could partially diffuse two images and then interpolate between
these "latent variables" and reverse the diffusion again. This should produce
  images that slowly morph between the two sources, but again with some variation.
- You should be able to use diffusion models for denoising. However, you would have
to somehow estimate how much noise is on the image so you can start the reverse
  process at the correct time step. It may also work only for Gaussian noise.
