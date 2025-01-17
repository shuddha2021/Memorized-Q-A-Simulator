#!/usr/bin/env python3
"""
Single-File 'Single-Page' Q&A Model in Python + Flask
=====================================================

This script:
  1. Trains your tiny Transformer Q&A model (autograd + Python).
  2. Embeds the HTML/CSS/JS in a single Python string.
  3. Serves that page with Flask, so you can interact in your browser.

Usage:
  pip install flask autograd
  python app.py
Open your browser at http://127.0.0.1:5000
"""

import sys
import autograd.numpy as np  # Use autograd's numpy
from autograd import grad
from autograd.misc.flatten import flatten
from flask import Flask, request, jsonify

# --------------------------
# MODEL HYPERPARAMETERS
# --------------------------
d_model = 256
num_heads = 8
d_ff = 512
num_layers = 3
learning_rate = 1e-3
num_epochs = 3000
max_gen_length = 50
temperature = 0.1
greedy = True

# We'll store the flattened parameters globally for convenience
flat_params = None
unflatten = None

# --------------------------
# DATASET
# --------------------------
base_dataset = [
    # Basic numerical questions (variations)
    "Q: How many legs does a cat have?\nA: 4\n",
    "Q: What's the number of legs on a cat?\nA: 4\n",
    "Q: How many legs do cats have?\nA: 4\n",
    "Q: Count the legs on a cat.\nA: 4\n",
    "Q: How many days are in a week?\nA: 7\n",
    "Q: What is 2+2?\nA: 4\n",
    "Q: How many hours in a day?\nA: 24\n",
    "Q: How many minutes in an hour?\nA: 60\n",
    "Q: How many months in a year?\nA: 12\n",
    "Q: What is the capital of France?\nA: Paris\n",
    "Q: What color is the sky?\nA: Blue\n",
    # Advanced chain-of-thought examples
    "Q: If a cat has 4 legs then how many legs do 5 cats have?\nA: Let's think. One cat has 4 legs, so 5 cats have 20 legs. Therefore, the answer is 20\n",
    "Q: If a cat has 4 legs then how many legs do 5 cats have?\nA: 20\n",
    "Q: If a cat has 4 legs then how many legs do 5 cats have?\nA: I know one cat has 4 legs. Multiplying 4 by 5 gives 20 legs. So the answer is 20\n",
    "Q: If a cat has 4 legs then how many legs do 5 cats have?\nA: 20\n"
]
# Overfit by repeating
dataset = base_dataset * 30

# --------------------------
# BUILD VOCABULARY
# --------------------------
def build_vocab(dataset):
    vocab = sorted(set("".join(dataset)))
    if "\n" not in vocab:
        vocab.append("\n")
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}
    return char2idx, idx2char

char2idx, idx2char = build_vocab(dataset)
vocab_size = len(char2idx)

def encode(text):
    return np.array([char2idx[ch] for ch in text if ch in char2idx], dtype=np.int32)

def decode(indices):
    return "".join([idx2char[int(idx)] for idx in indices])

# --------------------------
# POSITIONAL ENCODING
# --------------------------
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[np.newaxis, :, :]

# --------------------------
# ACTIVATIONS
# --------------------------
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

# --------------------------
# DENSE & LAYER NORM
# --------------------------
def dense(x, W, b):
    return np.dot(x, W) + b

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

# --------------------------
# SELF-ATTENTION
# --------------------------
def split_heads(x, num_heads):
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)
    return np.transpose(x, (0, 2, 1, 3))

def combine_heads(x):
    batch_size, num_heads, seq_len, head_dim = x.shape
    return np.transpose(x, (0, 2, 1, 3)).reshape(batch_size, seq_len, num_heads * head_dim)

def scaled_dot_product_attention(Q, K, V):
    d_head = Q.shape[-1]
    scores = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) / np.sqrt(d_head)
    attn_weights = softmax(scores, axis=-1)
    return np.matmul(attn_weights, V)

def multi_head_attention(params, x):
    Q = dense(x, params["W_q"], params["b_q"])
    K = dense(x, params["W_k"], params["b_k"])
    V = dense(x, params["W_v"], params["b_v"])
    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)
    attn_out = scaled_dot_product_attention(Q, K, V)
    attn_out = combine_heads(attn_out)
    return dense(attn_out, params["W_o"], params["b_o"])

# --------------------------
# FEED FORWARD
# --------------------------
def feed_forward(params, x):
    x_ff = dense(x, params["W1"], params["b1"])
    x_ff = gelu(x_ff)
    return dense(x_ff, params["W2"], params["b2"])

# --------------------------
# TRANSFORMER BLOCK
# --------------------------
def transformer_block(params, x):
    x_norm = layer_norm(x, params["ln1"]["gamma"], params["ln1"]["beta"])
    attn_out = multi_head_attention(params["mha"], x_norm)
    x = x + attn_out
    x_norm = layer_norm(x, params["ln2"]["gamma"], params["ln2"]["beta"])
    ff_out = feed_forward(params["ffn"], x_norm)
    x = x + ff_out
    return x

# --------------------------
# FORWARD PASS
# --------------------------
def forward(params, x_indices):
    x = params["embedding"][x_indices]
    batch_size, seq_len, _ = x.shape
    x = x + positional_encoding(seq_len, d_model)
    for block in params["transformer_blocks"]:
        x = transformer_block(block, x)
    return dense(x, params["W_final"], params["b_final"])

# --------------------------
# LOSS
# --------------------------
def cross_entropy_loss(logits, targets):
    eps = 0.1  # label smoothing
    batch_size, seq_len, _ = logits.shape
    probs = softmax(logits, axis=-1)
    one_hot = np.zeros_like(logits)
    one_hot[np.arange(batch_size)[:, None], np.arange(seq_len), targets] = 1.0
    smooth_targets = (1.0 - eps) * one_hot + eps / vocab_size
    loss = -np.sum(smooth_targets * np.log(probs + 1e-8)) / (batch_size * seq_len)
    return loss

def model_loss(params, x_indices, y_indices):
    logits = forward(params, x_indices)
    return cross_entropy_loss(logits, y_indices)

# --------------------------
# INIT PARAMS
# --------------------------
import numpy as onp  # for base random
def init_model_params():
    params = {}
    rng = onp.random.RandomState(42)

    # Embedding
    params["embedding"] = rng.normal(0, np.sqrt(2.0 / (vocab_size + d_model)),
                                     (vocab_size, d_model))

    # Final projection
    params["W_final"] = rng.normal(0, np.sqrt(2.0 / (d_model + vocab_size)),
                                   (d_model, vocab_size))
    params["b_final"] = np.zeros(vocab_size)

    # Transformer blocks
    transformer_blocks = []
    for _ in range(num_layers):
        block = {}
        mha = {}
        for name in ["q", "k", "v", "o"]:
            mha[f"W_{name}"] = rng.normal(0, np.sqrt(2.0 / d_model), (d_model, d_model))
            mha[f"b_{name}"] = np.zeros(d_model)
        block["mha"] = mha

        ffn = {}
        ffn["W1"] = rng.normal(0, np.sqrt(2.0 / d_model), (d_model, d_ff))
        ffn["b1"] = np.zeros(d_ff)
        ffn["W2"] = rng.normal(0, np.sqrt(2.0 / d_ff), (d_ff, d_model))
        ffn["b2"] = np.zeros(d_model)
        block["ffn"] = ffn

        block["ln1"] = {"gamma": np.ones((1, d_model)), "beta": np.zeros((1, d_model))}
        block["ln2"] = {"gamma": np.ones((1, d_model)), "beta": np.zeros((1, d_model))}
        transformer_blocks.append(block)
    params["transformer_blocks"] = transformer_blocks
    return params

# --------------------------
# PREPARE TRAIN DATA
# --------------------------
X_data = []
Y_data = []
for text in dataset:
    if not text.endswith("\n"):
        text += "\n"
    indices = encode(text)
    if len(indices) < 2:
        continue
    X_data.append(indices[:-1])
    Y_data.append(indices[1:])

def get_batch(batch_size=1):
    idx = onp.random.randint(0, len(X_data), batch_size)
    x_batch = onp.array([X_data[i] for i in idx])
    y_batch = onp.array([Y_data[i] for i in idx])
    return x_batch, y_batch

# --------------------------
# TRAIN
# --------------------------
params = init_model_params()
flat_params, unflatten = flatten(params)

def loss_flat(theta, x, y):
    return model_loss(unflatten(theta), x, y)

loss_grad = grad(loss_flat)

print("Initializing and training the Q&A model... This may take a while.")
curr_lr = learning_rate
best_loss = float('inf')
patience = 0

for epoch in range(num_epochs):
    x_batch, y_batch = get_batch(batch_size=1)
    current_loss = loss_flat(flat_params, x_batch, y_batch)
    g_flat = loss_grad(flat_params, x_batch, y_batch)
    flat_params = flat_params - curr_lr * g_flat

    if current_loss < best_loss:
        best_loss = current_loss
        patience = 0
    else:
        patience += 1
        if patience > 100:
            curr_lr *= 0.9
            patience = 0

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:04d}/{num_epochs}, Loss: {current_loss:.4f}, LR: {curr_lr:.6f}")

print("\nTraining complete!\n")
params = unflatten(flat_params)

# --------------------------
# INFERENCE
# --------------------------
def sample_with_temperature(logits, temperature=1.0, greedy=False):
    if greedy:
        return int(np.argmax(logits))
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    cum_probs = np.cumsum(probs)
    rand = onp.random.rand()
    return int(np.searchsorted(cum_probs, rand))

def generate_text(params, prompt, max_length=50, temperature=0.1, greedy=False):
    input_ids = encode(prompt)
    for _ in range(max_length):
        x = input_ids[None, :]
        logits = forward(params, x)
        next_logits = logits[0, -1, :]
        next_id = sample_with_temperature(next_logits, temperature, greedy=greedy)
        input_ids = np.append(input_ids, next_id)
        if idx2char[next_id] == "\n":
            break
    return decode(input_ids)

def extract_number_from_text(text):
    if "A:" in text:
        ans_line = text.split("A:")[-1].strip().split("\n")[0].strip()
        try:
            return int(ans_line)
        except ValueError:
            try:
                return float(ans_line)
            except ValueError:
                return ans_line
    return text.strip()

# --------------------------
# FLASK APP
# --------------------------
app = Flask(__name__)

# SINGLE-PAGE HTML (INLINED)
HTML_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Tiny Transformer Q&A Demo</title>
    <style>
        body {
            font-family: sans-serif;
            background: #f0f0f0;
            margin: 0; padding: 0;
        }
        header {
            background: #2c3e50; color: #fff; padding: 1em; text-align: center;
        }
        h1 { margin: 0; }
        #container {
            max-width: 600px;
            margin: 30px auto;
            background: #fff;
            border-radius: 8px;
            padding: 20px;
        }
        label { font-weight: bold; }
        textarea {
            width: 100%;
            height: 100px;
            margin-top: 5px;
            padding: 8px;
            box-sizing: border-box;
            font-family: sans-serif;
        }
        button {
            margin-top: 10px;
            cursor: pointer;
            padding: 8px 16px;
            background: #3498db;
            color: #fff;
            border: none;
            border-radius: 4px;
        }
        button:hover {
            background: #2980b9;
        }
        .response-box {
            margin-top: 20px;
            background: #f8f8f8;
            padding: 10px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .answer {
            font-weight: bold;
            margin-top: 10px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #777;
        }
    </style>
</head>
<body>
    <header>
        <h1>Improved Transformer Q&A</h1>
        <p>A toy model that memorizes small Q&A examples</p>
    </header>
    <div id="container">
        <p><strong>Important:</strong> Type your question in the exact Q&A format. 
        <br>Example:<br>
        <code>Q: How many legs does a cat have?<br>A:</code></p>

        <label for="promptInput">Your Q&A Prompt:</label>
        <textarea id="promptInput" placeholder="Q: How many legs does a cat have?\nA:"></textarea>
        <br>
        <button onclick="askModel()">Ask</button>

        <div class="response-box" id="modelResponse" style="display:none;">
            <div><strong>Full response:</strong></div>
            <div id="fullRespText"></div>
            <div class="answer" id="extractedAnsText"></div>
        </div>
    </div>

    <div class="footer">Copyright &copy; 2023. Tiny Transformer Q&A Demo.</div>

    <script>
    async function askModel() {
        const prompt = document.getElementById("promptInput").value.trim();
        const respBox = document.getElementById("modelResponse");
        const fullRespDiv = document.getElementById("fullRespText");
        const extractedDiv = document.getElementById("extractedAnsText");

        if(!prompt.startsWith("Q:")) {
            alert("Please start your prompt with 'Q:'");
            return;
        }
        if(!prompt.includes("A:")) {
            alert("Please include 'A:' in your prompt.");
            return;
        }

        // Show a loading message
        respBox.style.display = "block";
        fullRespDiv.innerHTML = "Thinking...";
        extractedDiv.innerHTML = "";

        // Send request
        const response = await fetch("/ask_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: prompt })
        });
        const data = await response.json();
        fullRespDiv.innerText = data.full_response;
        extractedDiv.innerText = "Extracted answer: " + data.extracted_answer;
    }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/ask_model", methods=["POST"])
def ask_model():
    req_data = request.get_json()
    user_prompt = req_data.get("prompt", "").strip()
    if not user_prompt.startswith("Q:"):
        return jsonify({
            "full_response": "Error: Prompt must start with 'Q:'",
            "extracted_answer": "N/A"
        })
    if "A:" not in user_prompt:
        user_prompt += "\nA:"
    generated = generate_text(params, user_prompt, max_length=max_gen_length,
                              temperature=temperature, greedy=greedy)
    extracted = extract_number_from_text(generated)
    return jsonify({
        "full_response": generated,
        "extracted_answer": str(extracted)
    })

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
