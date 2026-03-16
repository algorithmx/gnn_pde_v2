### Introducing a learnable temperature value into the softmax self-attention scores

SUMMARY
=======

Adding a per-head parameterized scaling factor to the query-key attention scores (analogous to adding a learnable temperature to the softmax) slightly improves performance transformer performance.

\[Update 11/2024: If this interests you some recent work re-examines the role of softmax in a similar vein. DeepMind’s [softmax is not enough](https://arxiv.org/pdf/2410.01104) proposes adapting softmax temperature based on the entropy of the input, applied to the language modeling head only. I had no luck with entropy-based adaptation, see “what didn’t work” section at bottom.\]

INTRODUCTION
============

About a year ago I wondered if the softmax used in attention would benefit from a learnable temperature term. A single parameterized scalar value could help the model essentially flatten or sharpen the distribution of attention scores. Since different heads take on different roles, the model might benefit from the option of using temperature per-head terms to further differentiate the different heads.

![\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{temp * QK^T}{\sqrt{d_k}}\right)V ](https://s0.wp.com/latex.php?latex=%5Ctext%7BAttention%7D%28Q%2C+K%2C+V%29+%3D+%5Ctext%7Bsoftmax%7D%5Cleft%28%5Cfrac%7Btemp+%2A+QK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%5Cright%29V+&bg=ffffff&fg=000&s=0&c=20201002)

where ![temp](https://s0.wp.com/latex.php?latex=temp&bg=ffffff&fg=000&s=0&c=20201002) is a single parameterized scalar value.

Each attention head at each layer has its own temperature term, so in total the model adds (heads \* layers) new parameters. In practice the temperature term can equivalently be factored out and used to multiply just the queries or just the keys prior to self-attention computation. This allows one to still use hard-coded library implementations like flash attention.

Note that the normal sqrt(head\_dimension) scaling factor remains to control for variance.

**NB**: Also note that temperature here multiplies the scores pre-softmax rather than dividing them as is customarily done. If it helps you can regard the term here as inverse temperature. In this work, low temperature results in flat, high entropy distributions and high temperature results in spiky, low entropy distributions.

HYPOTHESIS AND RESULT
=====================

If the model’s parameterized temperature terms a) change significantly from their initial value of 1.0 and b) show high variance in values across heads at a given layer, then we can take this as loose evidence these temperature terms are being utilized by the model effectively.

I played around with this idea and found that these two conditions a) and b) hold. Furthermore, a model with these temperature terms provides a slight decrease in loss against baseline for pretty much every language modeling architecture and setup I’ve used. (NB: all such architectures and setups are small: <1B parameters, <.25B tokens.) This is at the minimal cost of (heads \* layers) additional parameters.

![](https://nickcdryan.com/wp-content/uploads/2024/08/image.png?w=1024)

Adding per-head temperature term (yellow line) results in a slightly improved language modeling loss over baseline (black line) [wandb](https://api.wandb.ai/links/nickcdryan/c3f7f9fv)

![](https://nickcdryan.com/wp-content/uploads/2024/08/unnamed.png?w=668)

Each line shows the mean value of the temperature term per layer +/- variance across temperature terms per layer.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfcfwYgRSIkfUANRa-KZ5rNSwwSkNwM-Yt_in5UccwxIcxDMY3Nf58HgLj_g4xMGT6XEk2thNJbnFOSt5KMHDRnNQsvfSGpGoUQV-brPAgJiYygIpjs2u8e_mYTI7rHmVeSD2Nh5WfVpprYOQA7nnCAZR4?key=iAICGCRMRs6HrpkfpnONxA)

Another view, another model post-training

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcXU2_oZChgr1BUICsV-OdchjfpBFFz_oEE33A2niz4y7Z0Ljlr996FA0rhcajpPa5APD1lt0tUX0DIjF14qRXroq9loFlSiN7WIZopITnkSbgs9DVNWf3d4qZN84HwQs72tHs81d6y7WAgxBLhFvR_UPT2?key=Nnnnkv0Jz2ArLj_0xQzt9A)

Examining the query values across a single layer, you can see distinct splits where each head starts and ends. This effect is exaggerated with parameterized temperature terms.

Additionally, inspecting the value of the temperature terms provides another perspective on model behavior. For example, the temperature values of a model that turns out to be too deep given its width / insufficiently wide given its depth looks like this:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe7-hJRRa0vkPOSex5KC8mXpI68l9ecDCusqUiKGs12l7bCtn7UHWk6gV9bJEXoBli_USOoIyVClid_G1LKNzamO702kfj-8z1hjZUKaUOmjK0SF1sJbPkEahutRrpmj8N45eGuIo9-xPDYZCkXC8LQAGU?key=Nnnnkv0Jz2ArLj_0xQzt9A)

A .5B param model with 32 layers trained for 4000 iterations. Temperature terms initialized at .5

If you retrain this model configuration but with 18 layers instead of 32, the model now has the same or better loss and the temperature values look like this:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeiXI7SKEJj_qbgF8CUrxYBKID1FT8TNiU-8bsVdphUsBWut7YR65LNo9OAJztpIf0uYy6jPblRMrPXa2osB2vctNcLhSTAVHEXy1wXLQcJ7yKruCU9iS-U956tvbKPqi9HVAH8DwXHl7Hwi7GNEQboaXG9?key=Nnnnkv0Jz2ArLj_0xQzt9A)

Suggesting that the later layers were redundant. This diagnosis could clearly be derived other ways, for example by examining the gradient or skip connections. Nevertheless it serves as a very simple diagnostic tool helping me to design and examine the behavior of models.

FUTURE WORK
===========

In future work, one could pair the temperature values per head against an analysis of the head function. This could help shed light on why different layers learn very different temperature values for different heads.

For example, different assumed head roles (copy suppression, induction, negation, name mover) may benefit from sharper or smoother distributions depending on how specific or “all or nothing” their function is. A head focusing on induction might benefit from very sharp scores that can categorically identify whether or not a sequence has occurred in the past context, whereas a head with a more nuanced task may benefit from much smoother attention scores.

All this to say, the per-head temperature values may be beneficial because they provide an easy mechanism for the model to dial up or down the entropy of attention scores in a given head.

RELATED WORK
============

I’ve neglected to write up these results for a while. This is partly because the idea is very simple, and partly because I’ve since seen [very](https://ar5iv.labs.arxiv.org/html/2302.06130) [close](https://arxiv.org/pdf/2110.04403.pdf) [variations](https://arxiv.org/pdf/2103.00020.pdf) of [this](https://www.bmvc2021-virtualconference.com/assets/papers/1275.pdf) [idea](https://aclanthology.org/D18-1331.pdf) in a few places.

YaRN, for example, includes this trick, though the value of ![t](https://s0.wp.com/latex.php?latex=t&bg=ffffff&fg=000&s=0&c=20201002) is fixed. I asked one of the authors where this came from, but he didn’t have a source and said they included it because it worked.

![](https://nickcdryan.com/wp-content/uploads/2024/08/image-1.png?w=1024)

The self-attention mechanism for [YaRN](https://arxiv.org/pdf/2309.00071) includes a (non-parameterized) temperature scaling value for the pre-softmax query-key scores.

This led me to believe that this idea of temperature scaling the attention scores was diffuse enough that it was in the category of “trick” that many already knew, so writing it up wouldn’t add anything.

By pure coincidence I was recently led to the closest source of this idea I can find thanks to [BirchLabs](https://x.com/Birchlabs), who led me to an [existing implementation](https://github.com/crowsonkb/k-diffusion/blob/21d12c91ad4550e8fcf3308ff9fe7116b3f19a08/k_diffusion/models/image_transformer_v2.py#L378) in the [k-diffusion library](https://github.com/crowsonkb/k-diffusion) and the probable source, “[Query-Key Normalization for Transformers](https://arxiv.org/pdf/2010.04245)“

![](https://nickcdryan.com/wp-content/uploads/2024/08/image-2.png?w=934)

Query-key normalization does two things: a) L2 normalization for queries and keys along the head dimension, b) per-head learnable scalar value for attention scores, eliminating the traditional sqrt(head\_dimension) scaling factor.

![](https://nickcdryan.com/wp-content/uploads/2024/08/image-3.png?w=1020)

Query-key normalization self-attention equation

(Amusingly, I have not only seen but have actually implemented this paper before (in building [Memorizing Transformers](https://arxiv.org/pdf/2203.08913) we normalize the key values to mitigate model drift interfering with the KNN key value lookup) but I clearly skimmed and only focused on the normalization part of this paper.)

The authors include some interesting analysis of the the temperature terms, hypothesizing that lower entropy distributions of the attention scores improve learning by enabling “winner takes all” computation:

![](https://nickcdryan.com/wp-content/uploads/2024/08/image-4.png?w=1024)

So if you googled the right keywords, I hope you have ended up here and my hunting around will be of some use to you.

WHAT DIDN’T WORK
================

There are a lot of things you could try here. Nothing worked noticeably better than the simple initial idea of per layer, per head, single scalar value that multiplies the attention scores.

*   Scaling post-softmax
*   Scaling by (1,d) temperature values instead of (1,1) per layer per head.
*   Scaling per layer and not per head (heads share one value per layer)
*   Scaling per head and not per layer (layers share one value per head)
*   Wrapping the scaling value in tanh or sigmoid (in order to change the shape of the gradient) seems roughly equivalent
*   Attempts to use entropy, HHI, variance per example, etc. anything where we try to rescale the attention scores based on some statistic about the current data sequence being passed through didn’t work markedly better than simple per-layer-per-head scalers. Scaling the scaling term, adding a bias, etc. did not work.
    *   Example: “sigmoid-entropy-scaling-corrected-bias-.3-scales”, meaning per head per example entropy is calculated from the post-softmax query@key values and then scaled with a sigmoid function with scales initialized at .3 and bias initialized at 0. We then rerun the query@key and softmax using these values to scale the query@key pre-softmax attention scores, s.t. high or low entropy sequences can receive an appropriate “correction” to be made sharper/smoother.
        *   Temperature = (scale1 \* sigmoid(scale2 \* (sequence\_entropy – bias1)))
        *   The idea is scale1 can adjust the magnitude, scale2 adjusts the sharpness of the sigmoid, and bias lets us recenter the function over a reasonable value given the data.
            *   Some work went into trying to help the bias value. Consider that a 512 length sequence is split into 512 examples of 1,2,3,…510,511 token lengths. The entropy of the first 1-length example is always -log(1.0 \* 1).0 = 0, and the last 511-length sequence has max possible entropy of -log(1/511)=~6.2, which is close to what’s observed empirically. Initializing the bias as 0, or scaled linearly from 0-6.2 across increasing length examples, or logarithmicaly from 0-6.2…none of it made much of a difference. 
            *   However this is way too convoluted and involves creating the attention scores, then calculating the entropy per example, then using that value to rescale the values, then computing the new attention scores….too much work for too little gain.

[August 2, 2024](https://nickcdryan.com/2024/08/02/introducing-a-learnable-temperature-value-into-the-self-attention-scores/)