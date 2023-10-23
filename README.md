# Paper Presentation: 🦙 Llama 2: Open Foundation and Fine-Tuned Chat Models

## Presenters:
- Ricky Sun
- Yuning Wu

## Overview

### Context
Large Language Models (LLMs) are advanced AI tools skilled in diverse reasoning tasks, such as programming and creative writing. They interact with users via intuitive chat interfaces and have seen substantial adoption. However, while there have been public LLM releases like BLOOM (Scao et al., 2022), LLaMa-1 (Touvron et al., 2023), and Falcon (Penedo et al., 2023), none match the usability and safety of closed "product" LLMs such as ChatGPT, BARD, and Claude.

### Problems
The challenge lies in developing an LLM that bridges the gap between the performance of public models and the specialized fine-tuning of closed-product LLMs. Additionally, there is a need for transparency in training methodology, given the immense computational requirements and the limited number of entities developing LLMs.

### Llama 2 and Llama 2-Chat
On July 19, 2023, Meta released Llama 2 7B, 13B and 70B, and Llama 2-Chat 7B, 13B, and 70B models. (Llama 2 and Llama 2-Chat 34B models were benchmarked but not released due to red teaming issues). These models are successors of the Llama 1 models released in February 2023. Llama 2-Chat models are fine-tuned from Llama 2 models using supervised fine-tuning methods and Reinforcement Learning with Human Feedback (RLHF) (specifically, rejection sampling and Proximal Policy Optimization (PPO)). Meta's methodology emphasizes enhancing LLM safety, with unique observations made during the development process. They aim to provide a comprehensive description of their fine-tuning approach and safety enhancement techniques.

The image below demonstrates the birth of Llama 2-Chat from a Llama 2 model.
<img width="684" alt="Screenshot 2023-10-23 at 12 19 57 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/82897cdd-e405-4b0f-8f79-c64900b58b4c">

### Llama 2 VS Llama 1

#### Major differences 
- In Pre-Training
  - Llama 2 was trained on 40% more data than Llama 1
  - More robust data cleaning and data mixing
  - Doubled the context length and total tokens processed
- In Fine-Tuning (for Llama2-Chat)
  - More supervised fine-tuning and human annotations.
- In Model Architecture
  - Group Query Attention (GQA) to improve inference scalability for 34B and 70B models.
 
These 2 graphs demonstrate some of the key differences enlisted above.
<img width="684" alt="Screenshot 2023-10-23 at 12 30 32 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/18059a0b-9434-4327-9e2a-a24c559895f6">

<img width="684" alt="Screenshot 2023-10-23 at 12 07 00 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/b8c633bb-83c8-4d91-ad9a-0481b8e36778">

### Llama 2 Performance

#### Academic Benchmarks

<img width="684" alt="Screenshot 2023-10-23 at 1 05 43 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/a6702100-1cdf-4b3b-951e-6ce7295a3568">
<img width="684" alt="Screenshot 2023-10-23 at 1 06 17 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/d54ca38a-d6d8-4155-86a8-28805c526c37">
<img width="684" alt="Screenshot 2023-10-23 at 1 06 36 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/f98a019e-3960-44fc-b041-5de85631fb63">

#### Safety
<img width="684" alt="Screenshot 2023-10-23 at 12 46 42 AM" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/47910316/8593dd65-9d59-45e4-b195-5585fa66fd24">

As shown in the graph above, in safety and helpfulness benchmarks, Llama 2-Chat generally outperformed other open-source models. Safety improvements were made using specialized data annotation, red-teaming, and iterative evaluations. The models are being released to the public, with guidelines and recommendations provided for safe deployment. Meta has also documented their approach in detail to allow for reproducibility and further research by the community.

#### Data
- English CommonCrawl [67%]

five CommonCrawl dumps, ranging from 2017
to 2020, with the CCNet pipeline (Wenzek et al.,
2020). This process deduplicates the data at the
line level, performs language identification with
a fastText linear classifier to remove non-English
pages and filters low quality content with an ngram language model. In addition, we trained a
linear model to classify pages used as references
in Wikipedia v.s. randomly sampled pages, and
discarded pages not classified as references.

- C4 [15%]

During exploratory experiments, we
observed that using diverse pre-processed CommonCrawl datasets improves performance. We thus
included the publicly available C4 dataset (Raffel
et al., 2020) in our data. The preprocessing of C4
also contains deduplication and language identification steps: the main difference with CCNet is
the quality filtering, which mostly relies on heuristics such as presence of punctuation marks or the
number of words and sentences in a webpage.

- Github [4.5%]

We use the public GitHub
dataset available on Google BigQuery. We only
kept projects that are distributed under the Apache,
BSD and MIT licenses. Additionally, we filtered
low quality files with heuristics based on the line
length or proportion of alphanumeric characters,
and removed boilerplate, such as headers, with regular expressions. Finally, we deduplicate the resulting dataset at the file level, with exact matches.

- Wikipedia [4.5%]

We add Wikipedia dumps
from the June-August 2022 period, covering 20 languages, which use either the Latin or Cyrillic
scripts: bg, ca, cs, da, de, en, es, fr, hr, hu, it,
nl, pl, pt, ro, ru, sl, sr, sv, uk. We process the
data to remove hyperlinks, comments and other
formatting boilerplate.

- Gutenberg and Books3 [4.5%]

We include
two book corpora in our training dataset: the Gutenberg Project, which contains books that are in the
public domain, and the Books3 section of ThePile (Gao et al., 2020), a publicly available dataset
for training large language models. We perform
deduplication at the book level, removing books
with more than 90% content overlap.

- ArXiv [2.5%]

We process arXiv Latex files
to add scientific data to our dataset. Following
Lewkowycz et al. (2022), we removed everything
before the first section, as well as the bibliography.
We also removed the comments from the .tex files,
and inline-expanded definitions and macros written
by users to increase consistency across papers.

- Stack Exchange [2%]

We include a dump of
Stack Exchange, a website of high quality questions and answers that covers a diverse set of domains, ranging from computer science to chemistry.
We kept the data from the 28 largest websites, removed the HTML tags from text and sorted the
answers by score (from highest to lowest).

<img width="684" alt="image" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/533992be-7ce7-4700-9e2f-654a7fafa098">


## Architecture Overview
<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/9c6c3468-cb78-4299-a5a4-cc70489022d4">

### Llama (1 & 2)

#### Rotary Embedding (RoPE)

One of the fundamental advancements in LLaMA2 is the adoption of Rotary Position Embedding (RoPE) in place of traditional absolute positional encoding. What sets RoPE apart is its ability to seamlessly integrate explicit relative position dependencies into the self-attention mechanism of the model. This dynamic approach offers several key advantages:
- Flexibility in Sequence Length: Traditional position embeddings often require defining a maximum sequence length, limiting their adaptability. RoPE, on the other hand, is incredibly flexible. It can generate position embeddings on-the-fly for sequences of any length.
- Decaying Inter-Token Dependency: RoPE is smart about modeling the relationship between tokens. As tokens become more distant from each other in a sequence, RoPE naturally reduces their inter-token dependencies. This gradual decay aligns more closely with how humans understand language, where the importance of earlier words tends to diminish.
- Enhanced Self-Attention: RoPE equips the linear self-attention mechanisms with relative position encoding, a feature not present in traditional absolute positional encoding. This enhancement allows for more precise utilization of token embeddings.

<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/da3bcc52-198f-456f-8ba9-96771ae34c76">

#### RMSNorm (Root Mean Square Layer Normalization)

Llama2 adopts Root Mean Square Layer Normalization (RMSNorm), to enhance the transformer architecture by replacing the existing Layer Normalization (LayerNorm). LayerNorm has been beneficial for improving training stability and model convergence, as it re-centers and re-scales input and weight matrix values. However, this improvement comes at the cost of computational overhead, which slows down the network.

$$ \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2} $$

$$ y = \frac{x}{\text{RMS}(x)} \times \gamma + \beta $$

- γ is the scale parameter.
- β is the shift parameter.

RMSNorm, on the other hand, retains the re-scaling invariance property while simplifying the computation. It regulates the combined inputs to a neuron using the root mean square (RMS), providing implicit learning rate adaptation. This makes RMSNorm computationally more efficient than LayerNorm.

$$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i $$

$$ \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 $$

$$ y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta $$

- x is the input vector.
- N is the dimensionality of x.
- μ is the mean of the input.
- $σ^2$ is the variance of the input.
- ϵ is a small constant added for numerical stability.
- γ is the scale parameter.
- β is the shift parameter.

Extensive experiments across various tasks and network architectures show that RMSNorm performs as effectively as LayerNorm while reducing computation time by 7% to 64%.

This custom script first standardizes the input x, by dividing it by its root mean square, thereby making it invariant to scaling changes. The learned weight parameter self.weight is applied to each element in the standardized tensor. This operation adjusts the magnitude of the values based on the learned scaling factor.

#### KV (Key-Value) Caching

Key-Value (KV) caching is a technique used to accelerate the inference process in machine learning models, particularly in autoregressive models like GPT and Llama. In these models, generating tokens one by one is a common practice, but it can be computationally expensive because it repeats certain calculations at each step. To address this, KV caching comes into play. It involves caching the previous Keys and Values, so we don’t need to recalculate them for each new token. This significantly reduces the size of matrices used in calculations, making matrix multiplications faster. The only trade-off is that KV caching requires more GPU memory (or CPU memory if a GPU isn’t used) to store these Key and Value states.

Regarding the code, the KVCache class is responsible for handling this caching. It initializes two tensors, one for keys and one for values, both are initially filled with zeros. The update method is used to update the cache with new Key and Value information while the get method retrieves the cached Key and Value information based on the starting position and sequence length. This information can then be used for efficient attention calculations during token generation.

During inference, the process operates on one token at a time, maintaining a sequence length of one. This means that, for Key, Value, and Query, both the linear layer and rotary embedding exclusively target a single token at a specific position. The attention weights are precomputed and stored for Key and Value as caches, ensuring that these calculations occur only once and their results are cached. The script getmethod retrieves past attention weights for Key and Value up to the current position, extending their length beyond 1. During the scaled dot-product operation, the output size matches the query size, which generate only a single token.

<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/fdccfd26-0efe-402c-99cf-eb7deb9701bd">

#### SwiGLU (Swiss Function + Gated Linear Unit)

SwiGLU, as utilized in LLaMA2 models, is an activation function designed to enhance the performance of the position-wise feed-forward network (FFN) layers in the Transformer architecture.

The definition of SwiGLU is given by the following mathematical expression:

$$ \text{SwiGLU}\left(x, W, V, b, c, \beta\right) = \text{Swish}\_{\beta}\left(xW + b\right) \otimes \left(xV + c\right) $$

Here, x is the input to the neuron, W and V are weight matrices, b and c are bias vectors, and β is a constant. The ⊗ symbol denotes element-wise multiplication, while the Swish function is defined as:

$$ \text{Swish}\_{\beta}\left(x\right) = x \cdot \sigma\left(\beta x\right) $$

where σ is the sigmoid function. The purpose of the Swish function is to introduce non-linearity into the activation function while still allowing for efficient computation.

During the forward pass, the input tensor x is subjected to multi layer of linear transformations. The SwiGLU activation function, applied after first transformation, enhances the expressive power of the model. The final transformation maps the tensor back to its original dimensions. This unique combination of SwiGLU activation and multiple FeedForward layer enhances the performance of the model.

### Unique to Llama 2

#### Attention - Group Query Attention (GQA)

Llama incorporates a technique called grouped-query attention (GQA) to address memory bandwidth challenges during the autoregressive decoding of Transformer models. The primary issue stems from the need to load decoder weights and attention keys/values at each processing step, which consumes excessive memory.

In response, two strategies are introduced: and .

- Multi-query attention (MQA) involves utilizing multiple query heads with a single key/value head, which speeds up decoder inference. However, it has drawbacks such as quality degradation and training instability.

- Grouped-Query attention (GQA), is an evolution of MQA and strikes a balance by using an intermediate number of key-value heads (more than one but fewer than the query heads). The GQA model efficiently breaks the query into n_heads segments like the original multi-head attention, and the key and value are divided into n_kv_headsgroups, enabling multiple key-value heads to share the same query.

By repeating key-value pairs for computational efficiency, the GQA approach optimizes performance while maintaining quality, as evidenced by the code implementation.

The provided code is for implementing grouped query attention (GQA) within the context of an autoregressive decoder using a Transformer model. Notably, during inference, the sequence length (seq_len) is always set to 1.

SelfAttentionis a class that combines mechanism that we have discussed. The key components of this class are as follows:

- Linear transformations are applied to the input tensor for queries (xq), keys (xk), and values (xv). These transformations project the input data into a form suitable for processing.
- The rotary embedding is applied to the query, key, and value tensors using the provided frequency complex number. This step enhances the model’s ability to consider positional information and perform attention computations.
- The key-value pairs (k and v) are cached for efficient memory usage. The cached key-value pairs are retrieved up to current position (start_pos + seq_len)
The query, key, and value tensors are prepared for Grouped-Query attention calculation by repeating key-value pairs n_rep times, where n_rep corresponds to the number of query heads that share the same key-value pair.
- Scaled dot-product attention computation. The attention scores are computed by taking the dot product of the query and key, followed by scaling. Softmax is applied to obtain the final attention scores. During the computation, the output size matches the query size, which is also 1.
- Finally, the module applies a linear transformation (wo) to the output, and the processed output is returned.

<img width="648" alt="image" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/5c5c1e04-7eac-40b8-947a-4a10caf06b30">


#### Attention - Ghost Attention

Summary:
Ghost Attention (GAtt) is a technique to ensure consistent adherence to specific instructions throughout multi-turn dialogues. By artificially attaching instructions to user messages and adjusting the training process, GAtt helps the model maintain attention on crucial instructions, leading to more consistent and context-aware responses.

Potential Improvements:
While GAtt shows promise, it's still in a basic form. There's room for enhancement, such as teaching the model to modify the system message during a conversation.


#### Llama 2 -> Llama 2 Chat -> RLHF & rewarding model

<img width="684" alt="image" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/ce81bc02-37a6-4df4-aa68-5f0bdf3445f1">

## Discussion Question: How can we improve some of these architecture components?

**************************
#### Source code -> Pseudal-code

**Algorithm: DTransformer**

**Input:** `x`, a sequence of token IDs.

**Output:** $P \in (0,1)^{N \times \text{length}(x)}$, where the t-th column of `P` represents $\hat{P̂_θ}(x[t+1]|x[1:t])$.

**Hyperparameters:** $ℓ_{\text{max}}, L, H, d_e, d_{mlp} \in \mathbb{N}$

**Parameters:**
- $W_e \in \mathbb{R}^{d_e \times N}$, $W_p \in \mathbb{R}^{d_e \times ℓ_{\text{max}}}$: the token and rotary positional embedding matrices.
- For each layer `l`:
  - $W_l$, Group Query Attention multi-head attention parameters for layer `l`.
  - $\gamma^1, \beta^1, \gamma^2, \beta^2$: sets of RMS layer-norm parameters.
  - $w^l_{mlp1}, b^l_{mlp1}, w^l_{mlp2}, b^l_{mlp2}$: MLP parameters.
- $\gamma, \beta$: final RMSlayer-norm parameters.
- $W_u \in \mathbb{R}^{N \times d_e}$: the unembedding matrix.

**Algorithm:**
1. $ℓ \leftarrow \text{length}(x)$
2. For each `t` in `ℓ`: $e_t \leftarrow W_e \cdot x[t] + W_p[:,t]$
3. $X \leftarrow [e_1, e_2, ... e_ℓ]$
4. For each `l` from 1 to `L`:
   - For each `t` in `ℓ`:
     - $X{[:,t]} \leftarrow {RMSLayerNorm}(\tilde{X}{[:,t]} | \gamma_l{1}, \beta_l{1})$
     - $X \leftarrow X + \text{GQA-MHAttention}(X, W_l, \text{Mask}[t, :] = [t \leq t'])$$
     - $X{[:,t]} \leftarrow {RMSLayerNorm}(\tilde{X}{[:,t]} | \gamma_l{2}, \beta_l{2})$
     - $X \leftarrow X + w^l_{mlp2} \cdot \text{SwiGLU}(w^l_{mlp1} \tilde{X} + b^l_{mlp1}1^T) + b^l_{mlp2}1^T$
5. For each `t` in `ℓ`: $X[:,t] \leftarrow {RMSLayerNorm}(X[:,t], \gamma, \beta)$
6. Return $P = \text{softmax}(W_u X)$
**************************

### Model Safety and Biases

#### Gender
#### Sexuality & LGBT
#### Carbon

## Critical Analysis & Discussion

- Trained predominantly on English (89.7%)

  - Bias towards English content: This heavy English representation might make the model less reliable or knowledgeable about topics in other languages or from non-English perspectives.
  - Limitations for global safety guideline and usage: For users who are non-native English speakers or who prefer to interact in their native languages, this could lead to less effective results, potentially propagating biases or misunderstandings from English-centric sources.

<img width="684" alt="image" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/f19421d9-7049-4087-b707-37f6a7aec2fa">

- Other open sourced models: Orca & Phi 1 
    - Lack of transparency: Omitting specific results can raise questions about the transparency of the model's development and evaluation. Why were these particular results not highlighted? Was there something specific about them that didn't fit the narrative or expectation? For example, Phi 1 model scored 50 in coding with only 1.3B parameters but was not listed on the Benchmark table.

- safety & helpfulness reward model: false refusal
  - Over-caution: While trying to make the model safer, there's a risk of the model being too cautious, resulting in unnecessary refusals to provide answers that are actually safe and useful.
  - User frustration: Frequent false refusals might lead to user dissatisfaction and mistrust in the system, even when it's trying to act in the user's best interest.

- limitations of human evaluation
  - Subjectivity: Human evaluators bring their own biases and interpretations to the table. Their evaluations might not be consistent across different individuals or cultures.
  - Scale: Given the vast amount of data the model has been trained on, human evaluation can only cover a minuscule fraction. This means the evaluations might miss edge cases or rare problematic outputs.

- Potential misuse
  - Malicious use: Like any powerful tool, there's potential for misuse. This could range from spreading misinformation to more advanced malicious intents like phishing or manipulation.
  - Ethical considerations: The creators and distributors of such models need to think about how they can prevent or at least mitigate the potential for harmful applications.

- responsible use guide is vague and generic
  - Lack of specificity: A vague guide might not provide clear instructions or standards for users, leading to varied interpretations and potential misuse.
  - Accountability: It's essential to have a robust and clear guide to ensure the model's responsible usage and to hold users accountable for potential misuse.

- Benchmarks
  - Page 48 social IQ (llama 1 > llama 2)
<img width="648" alt="image" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/bc21c6cf-6002-44f6-b009-b96114208e53">
  - AQuA-RAT (test for mathematical reasoning: table 24) Orca 13B, exactly the same size, got 27.9 vs 21.7
<img width="648" alt="image" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/fc7800f1-2e64-43ec-a073-6ff8945c9f55">

- sentiment-analysis for (right wing greater than left wing)
  - Potential biases: If the model shows different sentiments towards different political ideologies, it might be perceived as biased, leading to mistrust or misuse. This also raises questions about the neutrality of AI models and the data they're trained on.
  
- Mistral AI

## Discussion Question: Can you think of any potential improvements?

## Paper Citation
- Llama 1 Paper: https://arxiv.org/abs/2302.13971
- Llama 2 Paper: https://arxiv.org/pdf/2307.09288
- Mistral-7b Paper: https://arxiv.org/pdf/2310.06825.pdf
- RMS Normalization: https://arxiv.org/abs/1910.07467  
- Rotary Positional Embedding (RoPE): https://arxiv.org/abs/2104.09864  
- SwiGLU Activation Function: https://paperswithcode.com/method/swiglu
- Group Query Attention Paper: https://arxiv.org/pdf/2305.13245.pdf
- Rotary Position Embedding Paper: https://arxiv.org/pdf/2305.13245.pdf
- MMLU Dataset: https://paperswithcode.com/dataset/mmlu  
- MMLU Paper: https://arxiv.org/abs/2009.03300
- Formal Algorithm of Transformers Paper: https://arxiv.org/pdf/2207.09238.pdf

## Links to Helpful Resources
- Llama 2 Playground: https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat
- Llama 2 Website: https://ai.meta.com/llama/
- Llama 2 Repo: https://github.com/facebookresearch/llama
- Llama 2 Huggingface: https://huggingface.co/docs/transformers/main/model_doc/llama2
- Mistral Website: https://mistral.ai/
- Mistral repo: https://github.com/mistralai
- Mistral huggingface: https://huggingface.co/mistralai/Mistral-7B-v0.1
- RLHF Explained: https://huyenchip.com/2023/05/02/rlhf.html  
- Transformers Explained: https://deepgram.com/learn/visualizing-and-explaining-transformer-models-from-the-ground-up  
- Activation Functions Explained: https://www.geeksforgeeks.org/activation-functions-neural-networks/  
- Understanding Llama2 Architecture: https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7

## Video Overview of Llama 2 Models
https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/71966ca2-4d89-4d9b-954f-952834dc25de







