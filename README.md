# Transformer Model for text translation

This repository contains an implementation of the Transformer model for the task of machine translation from English to Russian. The model is based on the architecture described in [“Attention is All You Need”](https://arxiv.org/abs/1706.03762) and implemented using PyTorch. The code includes training, testing and calculation of the BLEU metric for evaluating translation quality.

## Project Description

- **Objective**: Build a Transformer model to translate text from English to Russian.
- **Dataset**: Used dataset [Helsinki-NLP/news_commentary](https://huggingface.co/datasets/Helsinki-NLP/news_commentary) with en-ru translation pairs.
- **Technologies**: PyTorch, Hugging Face Datasets, Tokenizers, SacreBLEU.
- **Metrics**: BLEU Score for evaluating translation quality.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SUKUNA-AI/Transformer-model.git
   cd Transformer-model
   ```
2. Recommended libraries : `PyTorch`, `datasets`, `tokenizers`, `sacrebleu`, `tqdm`.
   
## Usage

1. model training:
   Run the training script (see `[Test_transformer_model.ipynb]`) with the parameters specified in `get_config()`:
   ```python
   config = get_config()
   train_model(config)
   ```
   The model is saved in the `/content/drive/MyDrive/Colab Notebooks/Transformer` folder.

2. testing the model:
   Use `testing_notebook.ipynb` to load the saved model and perform the translation:
   ```python
   translated_text = translate_sentence(model, src_text, tokenizer_src, tokenizer_tgt, seq_len, device)
   ```

3. Evaluation:
   The BLEU metric is calculated for the validation sample:
   ```python
   bleu_score = corpus_bleu(hypotheses, [references])
   print(f"BLEU Score: {bleu_score.score}")
   ```

## Repository structure

- `Transformer_model_(1).ipynb`: Transformer model training code.
- `Test_transformer_model_model.ipynb`: Code for testing and calculating BLEU.
- `translations_dataframe.csv`: Example of translation results in CSV format.
- `transformer_translate_model09.pt`: The model itself after 10 epochs of training.
- `tokenizer_en.json`: The tokenizer file for the English language
- `tokenizer_en.json`: Russian language tokenizer file.
## Results

- **BLEU Score**: 34.22 (on validation sample after 10 epochs).
- Translation examples are available in `Test_transformer_model.ipynb` and `translations_dataframe.csv`


If you have any questions or suggestions, create an issue in the repository!


---

### Transformer architecture theory based on the article “Attention is All You Need”

### Introduction
The Transformer architecture, proposed in the article [“Attention is All You Need”](https://arxiv.org/abs/1706.03762) by Vaswani et al. in 2017, has replaced recurrent neural networks (RNNs) in sequence processing tasks such as machine translation. The main innovation is the Attention mechanism (Attention), which allows the model to focus on important parts of the input sequence without using recurrent layers.

### Main Components

Transformer consists of two main blocks: **encoder** and **decoder**. Both blocks are a stack of N identical layers (typically N=6).

#### 1. Encoder
The encoder converts an input sequence (e.g., a sentence in English) into a set of hidden representations.

- **Input Embeddings**: Each word is converted into a vector of fixed dimensionality (d_model, e.g. 512).
- **Position coding**: Adds word order information since Transformer does not have a recurrent structure. The sine functions used are:
  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```
- **Self-Attention**: Allows each word in a sequence to take into account the context of other words through queries (Q), keys (K), and values (V).
  
- **Multi-Head Attention**: Allows the model to simultaneously consider different parts of the input sequence. Attention formula:
  ```
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
  ```
  where Q (queries), K (keys), V (values) are linear transformations of the input vector, d_k is the dimensionality of the head.
  
![Multi-Head Attention](https://cdn-uploads.huggingface.co/production/uploads/65f42e30c9940817caaaa8e6/bYyg8WZzq2Koo2rjkezcw.png)
  
- **Full-linked layer (Feed-Forward)**: Applies to each vector independently: `FFN(x) = max(0, xW1 + b1)W2 + b2`.
- **Residual Bounds and Normalization**: Each sublayer (attention and FFN) is accompanied by a residual link (`x + Sublayer(x)`) and a layer normalization (`LayerNorm`).
  

- **Encoder model itself (there can be several encoders, e.g. 8)**:
  
![Encoder](https://i.morioh.com/210424/11eb11de.webp)

#### 2. Decoder
The decoder generates an output sequence (e.g. Russian translation) based on the encoder outputs.

- **Masked multi-headed attention**: Similar to the encoder, but with a mask to prevent “peeking” into future tokens (causal mask).
  
![Masked self-attention](https://habrastorage.org/r/w1560/webt/86/aw/oh/86awoh2nlrcsdh3_hveet0rh3vw.png)

- **Cross-attention**: this is a mechanism in the transformer that allows the decoder to use information from the input sequence processed by the encoder when generating the output sequence.
![cross-attention](https://yastatic.net/s3/education-portal/media/decoder_attention_03be436fa1_b4c942b0ae.webp)

- **Full Layer**: As in encoder.
- **Projection Layer**: Converts the decoder output to logarithms of probabilities by dictionary.

![Decoder Architecture](https://upload.wikimedia.org/wikipedia/commons/5/55/Transformer%2C_one_decoder_block.png)

#### 3. General Architecture
- The encoder and decoder are connected via cross-attention.
- The input and output are augmented with special tokens `[SOS]` (start of sequence) and `[EOS]` (end of sequence).

![Full Transformer](https://www.mdpi.com/sensors/sensors-20-03228/article_deploy/html/images/sensors-20-03228-g003.png)

### How it works from the inside
1. **Tokenization**: The input text is broken down into tokens (words or subwords) and converted into indexes.
2. **Embeddings + positional encoding**: Tokens are converted into vectors to which position information is added.
3. **Encoder**: Passes through N layers, where each layer applies attention and FFN, creating contextualized representations of the input.
4. **Decoder**: Generates output token by token, using masked attention to predict the next word and cross-attention to communicate with the encoder.
5. **Output**: The projection layer outputs the probabilities of the next token, the most probable one is selected (e.g. via `argmax`).

### Advantages
- Parallelization: Lack of recurrence allows the entire sequence to be processed simultaneously.
- Long-term dependencies: Attention mechanism effectively captures connections between distant words.

---

### Implementation, training and testing

### Implementation
My code in two files (`training_notebook.ipynb` and `testing_notebook.ipynb`) implements Transformer for English to Russian translation:
- **Model**: Implemented all components (InputEmbeddings, PositionalEncoding, MultiHeadAttention, Encoder, Decoder, etc.) according to the original architecture.
- **Configuration**: Parameters set in `get_config()`:
  - `batch_size`: 32
  - `num_epochs`: 10
  - `lr`: 2e-4
  - `seq_len`: 570
  - `d_model`: 512
  - `N`: 6 (number of layers)
  - `h`: 8 (number of attention heads)

### Training
- **Dataset**: Used [Helsinki-NLP/news_commentary](https://huggingface.co/datasets/Helsinki-NLP/news_commentary) with 190,104 en-ru pairs.
- **Preprocessing**: Tokenization with `tokenizers` (WordLevel), split into train (70%) and validation (30%).
- **Process**: 
  - 10 epochs of training on GPU (CUDA).
  - Optimizer: Adam (`lr=2e-4`).
  - Loss function: CrossEntropyLoss with `[PAD]` ignored.
  - Progress is displayed via `tqdm`, translation examples are displayed after each epoch.
- **Saving**: The model is saved after each epoch in `/content/drive/MyDrive/Colab Notebooks/Transformer`.

### Testing
- **Model Loading**: Used model after the 10th epoch (`transformer_translate_model09.pt`).
- **Translate**: Implemented `translate_sentence` function to generate token-by-token translation using greedy decoding (`argmax`).
- **Assessment**: 
  - Translation of all sentences from the validation sample.
  - BLEU calculation using `sacrebleu`.

### Metric
- **BLEU Score**: 34.22 (after 10 epochs).
- BLEU measures the similarity of the predicted translation (`hypotheses`) to the reference translation (`references`), given the n-grams (1-4).

#### Translation Examples
1. **Исходный текст**: "Правительствам, которым придется решать задачу повышения расходов, необходимо будет разработать целевые системы социальной защиты..."
- **Эталон**: "Правительства, которые сталкиваются с большими расходами, должны разработать целевые системы социальной защиты..."
- **Предсказание**: "Правительства , которые сталкиваются с большими расходами , должны разработать целевые системы социальной защиты , которые защищают своих бедных людей посредством денежных и денежных переводов ."
2. **Исходный текст**: "Нашей работой по экономическому развитию, сохранению мира, охране окружающей среды и здравоохранению..."
- **Эталон**: "Нашей работой в развитии, сохранении мира, охране окружающей среды и здоровье..."
- **Предсказание**: "Нашей работой в развитии , сохранении мира , охране окружающей среды и здоровье , мы странам и сообществам строить лучшее , более будущее ."

---

### Conclusions

### Analyzing the results
- **BLEU Score (34.22)**: A score of 34.22 indicates moderate translation quality. By comparison, modern models (e.g. Google Translate) achieve 50+ on similar tasks, but your score is good for a basic Transformer implementation after 10 epochs.
- **Translation Quality**: 
  - The model successfully captures the overall meaning and structure of sentences.
  - Errors: word omissions, grammatical inaccuracies, excessive repetition (“money and monetary”).
- **Learning Progress**: Losses decreased from 4.922 (epoch 0) to 2.522 (epoch 9), showing stable learning.

### Limitations
- **Number of epochs**: 10 epochs is not enough for full convergence of the model with this amount of data.
- **Dictionary size**: A limited dictionary can lead to missing rare words.
- **Rad decoding**: Using `argmax` instead of beam search reduces translation diversity and accuracy.

### Future Improvements
1. Increase the number of epochs (e.g. to 20-30) and use a learning rate scheduler.
2. Apply beam search to improve generation.
3. Use pre-trained embeddings (e.g., BERT) to improve quality.
4. Add more data or use augmentation.

