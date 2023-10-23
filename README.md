# Paper Presentation: 🦙 Llama 2: Open Foundation and Fine-Tuned Chat Models

## Presentors:
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


## Architecture Overview
<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/9c6c3468-cb78-4299-a5a4-cc70489022d4">

### Llama


#### Rotary Embedding (RoPE)

One of the fundamental advancements in LLaMA2 is the adoption of Rotary Position Embedding (RoPE) in place of traditional absolute positional encoding. What sets RoPE apart is its ability to seamlessly integrate explicit relative position dependencies into the self-attention mechanism of the model. This dynamic approach offers several key advantages:
- Flexibility in Sequence Length: Traditional position embeddings often require defining a maximum sequence length, limiting their adaptability. RoPE, on the other hand, is incredibly flexible. It can generate position embeddings on-the-fly for sequences of any length.
- Decaying Inter-Token Dependency: RoPE is smart about modeling the relationship between tokens. As tokens become more distant from each other in a sequence, RoPE naturally reduces their inter-token dependencies. This gradual decay aligns more closely with how humans understand language, where the importance of earlier words tends to diminish.
- Enhanced Self-Attention: RoPE equips the linear self-attention mechanisms with relative position encoding, a feature not present in traditional absolute positional encoding. This enhancement allows for more precise utilization of token embeddings.

<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/da3bcc52-198f-456f-8ba9-96771ae34c76">

#### RMSNorm (Root Mean Square Layer Normalization)

Llama2 adopts Root Mean Square Layer Normalization (RMSNorm), to enhance the transformer architecture by replacing the existing Layer Normalization (LayerNorm). LayerNorm has been beneficial for improving training stability and model convergence, as it re-centers and re-scales input and weight matrix values. However, this improvement comes at the cost of computational overhead, which slows down the network.

<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/317056a3-ceb5-40ac-bc8a-3ce48847ff40">

RMSNorm, on the other hand, retains the re-scaling invariance property while simplifying the computation. It regulates the combined inputs to a neuron using the root mean square (RMS), providing implicit learning rate adaptation. This makes RMSNorm computationally more efficient than LayerNorm.

<img width="684" src="https://github.com/Rundstedtzz/llama2-mistral-presentation/assets/63605514/7a95c485-913a-4b4e-a0f6-3edd2fabbbb6">

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

### Llama 2 uniques

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

#### Attention - Ghost Attention

Summary:
Ghost Attention (GAtt) is a technique to ensure consistent adherence to specific instructions throughout multi-turn dialogues. By artificially attaching instructions to user messages and adjusting the training process, GAtt helps the model maintain attention on crucial instructions, leading to more consistent and context-aware responses.

Potential Improvements:
While GAtt shows promise, it's still in a basic form. There's room for enhancement, such as teaching the model to modify the system message during a conversation.

**************************
Source code + pseudal code
**************************










#### Llama 2 -> Llama 2 Chat -> RLHF

#### Fine-tuning (AB)



### Llama 2 vs other open-sourced models

### Safety & helpfulness

### Meta employees -> Mistral AI -> nice

## Conclusion & Discussion
Answer one or more of the following questions: What was overlooked by the authors? What could have been developed further? Were there any errors? Have others disputed the findings?

## Future Research

## Links to Helpful Resources
- Llama 2 Playground: https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat  
- Llama 1 Paper: https://arxiv.org/abs/2302.13971  
- Transformers Explained: https://deepgram.com/learn/visualizing-and-explaining-transformer-models-from-the-ground-up  
- RMS Normalization: https://arxiv.org/abs/1910.07467  
- Rotary Positional Embedding (RoPE): https://arxiv.org/abs/2104.09864  
- Activation Functions Explained: https://www.geeksforgeeks.org/activation-functions-neural-networks/  
- SwiGLU Activation Function: https://paperswithcode.com/method/swiglu  
- RLHF Explained: https://huyenchip.com/2023/05/02/rlhf.html  
- MMLU Dataset: https://paperswithcode.com/dataset/mmlu  
- MMLU Paper: https://arxiv.org/abs/2009.03300  














