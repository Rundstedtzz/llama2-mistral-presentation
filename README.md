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


## Architectural Overall
Prepare a formal pseudocode description of the proposed model, indicate how it differs from previous models

### Llama 2 vs Llama 1
### Llama 2 uniques

#### Llama 2 -> Llama 2 Chat -> RLHF

#### Fine-tuning (AB)

**************************
Source code + pseudal code
**************************

### Llama 2 vs other open-sourced models

### Safety & helpfulness

### Meta employees -> Mistral AI -> nice

## Conclusion & Discussion
Answer one or more of the following questions: What was overlooked by the authors? What could have been developed further? Were there any errors? Have others disputed the findings?

## Future Research

## Links to Helpful Resources
Llama 2 Playground: https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat  
Llama 1 Paper: https://arxiv.org/abs/2302.13971  
Transformers Explained: https://deepgram.com/learn/visualizing-and-explaining-transformer-models-from-the-ground-up  
RMS Normalization: https://arxiv.org/abs/1910.07467  
Rotary Positional Embedding (RoPE): https://arxiv.org/abs/2104.09864  
Activation Functions Explained: https://www.geeksforgeeks.org/activation-functions-neural-networks/  
SwiGLU Activation Function: https://paperswithcode.com/method/swiglu  
RLHF Explained: https://huyenchip.com/2023/05/02/rlhf.html  
MMLU Dataset: https://paperswithcode.com/dataset/mmlu  
MMLU Paper: https://arxiv.org/abs/2009.03300  














