# Memorized Q&A Simulator

**Memorized Q&A Simulator** is a small-scale, educational demonstration of some core ideas behind Transformer-based language models. In this project, we simulate a toy Transformer Q&A model that "memorizes" a set of explicitly formatted multi‑line Q&A examples. While it does not generalize or truly reason, it serves as a simplified showcase of many underlying concepts found in state-of-the-art LLMs.

<img width="1232" alt="Screenshot 2025-01-17 at 9 32 31 AM" src="https://github.com/user-attachments/assets/d91de9da-2ada-4aff-8503-9411b405d15b" />

<img width="1232" alt="Screenshot 2025-01-17 at 9 32 52 AM" src="https://github.com/user-attachments/assets/d03289f2-8250-4996-923c-33351aa9a635" />


## Overview

The simulator stores several Q&A pairs (including variations and chain-of-thought reasoning examples) and returns a predetermined answer if the user enters a prompt that exactly matches one of the memorized examples. For example, if you enter:

```
Q: If a cat has 4 legs then how many legs do 5 cats have?
A:
```

the model (which is over‑fitted on that example) should return something like:

```
Full response: Q: If a cat has 4 legs then how many legs do 5 cats have?
A: Let's think. One cat has 4 legs, so 5 cats have 20 legs. Therefore, the answer is 20
Extracted answer: 20
```

Additionally, the simulator includes an extra example for humans so that when you ask:

```
Q: How many legs does a human have?
A:
```

it returns "2".

## Transformer Architecture & Logic

Although this project uses a JavaScript simulation (i.e. a memorized dictionary lookup) rather than running the full mathematical machinery of a Transformer, the original Python code (which inspired the simulation) was designed based on the following key ideas:

### 1. **Embeddings & Positional Encoding**
- **Embeddings:**  
  Each token (character) in the input is represented as a vector (a list of numbers). This is like giving every token its own unique "fingerprint."
- **Positional Encoding:**  
  Since Transformers do not have a natural sense of order, we add sinusoidal positional encodings. Think of this as assigning a "timestamp" to every token so that the model knows their order in the sequence.

### 2. **Self-Attention and Multi-Head Attention**
- **Self-Attention:**  
  The model computes attention scores between every pair of tokens. This helps it figure out which parts of the input are most relevant to one another.
- **Multi-Head Attention:**  
  Instead of computing one attention score for each token, the model splits the token representation into several "heads." Each head focuses on a different aspect or relationship in the data. Imagine a team of experts, each looking at the input from a different perspective, then combining their insights together.

### 3. **Feed-Forward Networks and Nonlinear Activations**
- **Feed-Forward Network (FFN):**  
  After attention, each token's representation is passed through a small neural network independently. This network uses nonlinear activation functions (such as GELU) to provide complex transformations.
- **Analogy:**  
  Think of a factory where every item is processed with the same recipe, regardless of its order in a line.

### 4. **Residual Connections and Layer Normalization**
- **Residual (Skip) Connections:**  
  The output of each sub-layer (attention or FFN) is added back to its input. This helps preserve information and facilitates training of very deep models.
- **Layer Normalization:**  
  Each token's vector is normalized (scaled to have mean 0 and variance 1) to stabilize the learning process and maintain consistent gradients during training.

### 5. **Loss Function and Training**
- **Cross-Entropy Loss with Label Smoothing:**  
  The loss function compares the model's predicted probability distribution to the "true" (one-hot) distribution. Label smoothing is applied to regularize the model.
- **Gradient Descent:**  
  The model's parameters are updated iteratively based on the gradient of the loss (using an adaptive learning rate), which is conceptually similar to making small adjustments on each step to minimize error.

## Features & Capabilities

- **Memorization of Q&A Pairs:**  
  The project demonstrates a model that memorizes a small set of Q&A examples (via repetition) and outputs the pre-learned answer when the prompt matches.
  
- **Chain-of-Thought Example:**  
  For advanced questions, the dataset includes chain-of-thought reasoning examples to simulate an explanation process before the final answer.

- **Simple Interactive User Interface:**  
  A minimal single-page app (implemented in HTML, CSS, and JavaScript) allows users to enter a Q&A prompt and receive a corresponding response instantly.

- **Simulated Transformer Components:**  
  Although our live version (in JavaScript) uses a memorized dictionary for demonstration, the underlying design is inspired by the critical building blocks of Transformers:
  - Embeddings and positional encoding
  - Multi-head attention (handling multiple "perspectives")
  - Feed-forward networks with nonlinear activation (GELU)
  - Residual connections and layer normalization for stability

## Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/memorized-qa-simulator.git
   ```

2. **Open index.html in Your Browser**
   - The demo is completely client-side; simply open the HTML file in a modern browser.

3. **Enter Your Prompt**
   - Use the Q&A format exactly. For example:
     ```
     Q: If a cat has 4 legs then how many legs do 5 cats have?
     A:
     ```
   - Or try:
     ```
     Q: How many legs does a human have?
     A:
     ```

4. **Click "Generate Response"**
   - The output panel will show the full response (the memorized Q&A text) and the extracted answer (e.g., "20" for cats, "2" for humans).

## How It Works (Simulation Mode)

Since running a full Python Transformer model directly in the browser (via Pyodide) can be slow and sometimes error-prone, this project also provides a simulation version implemented purely in JavaScript. In this simulation:

- A JavaScript object (dictionary) stores the exact Q&A pairs.
- When the user enters a prompt matching one of these keys, the simulator returns the stored Q&A.
- If there's no match, a default "unknown" message is given.

This approach lets you quickly demonstrate the idea behind a Transformer memorizing examples, without incurring heavy computation in the browser.

## Future Improvements

1. **Full Transformer Implementation in Browser:**
   - Explore using WebAssembly or TensorFlow.js to run a real (even if small) Transformer model on the client-side.

2. **Pre-training and Parameter Loading:**
   - Train the model offline and load pre-trained weights for faster inference.

3. **Improved Input Matching:**
   - Add fuzzy matching so that slight variations in prompt formatting can still produce correct responses.

4. **Enhanced UI:**
   - Integrate interactive visualizations showing attention maps or intermediate representations.

5. **Expanded Dataset:**
   - Incorporate more examples and variations for broader coverage.

## License

This project is provided for educational purposes. Feel free to use, modify, and distribute it as needed.
 
