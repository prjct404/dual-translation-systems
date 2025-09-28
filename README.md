
# Dual Translation Systems (Seq2Seq + Transformer)

An educational/research project that builds a machine translation system with **two approaches**:
1) **LSTM**   
2) **Seq2Seq**


---

## 🚀 Features
## 🧹 Preprocessing Pipeline  

Before training embeddings and classifiers, the raw shenasa/English-Persian-Parallel-Dataset was **cleaned, normalized, and tokenized**. This step is essential for Persian (Farsi) text, which often mixes scripts, punctuation styles, and contains noise.  

### Steps Implemented  

1. **Subsampling**  
   - For efficiency, we first sample `N=500,000` rows from the dataset.  

2. **Normalization**  
   - Convert Arabic characters to Persian equivalents (`ي → ی`, `ك → ک`).  
   - Remove zero-width and control characters (`‌`, `‏`, `‪`, …).  
   - Standardize digits: Persian → ASCII (`۱۲۳۴ → 1234`).  
   - Apply Hazm’s `Normalizer` for general Persian text normalization.  

3. **Noise & Artifact Removal**  
   - Strip HTML tags using regex / BeautifulSoup.  
   - Remove non-allowable symbols (extra punctuation, emojis, artifacts).  
   - Collapse multiple spaces into a single space.  

4. **Punctuation Filtering**  
   - Keep only sentence-relevant punctuation (`. ? !`).  
   - Remove unnecessary commas, semicolons, and repetitive symbols.  

5. **Stop-word Removal**  
   - Remove high-frequency Persian stop-words (e.g., "از", "به", "که", "و").  
   - English stop-words are also filtered in case of bilingual content.  

6. **Tokenization**  
   - Text is split into word tokens for downstream embedding and model input.  
   - Uses Hazm tokenizer for Persian, plus Hugging Face tokenizer for BERT-based models.  

### Output  
- `content_clean`: fully preprocessed and normalized text.  
- `tokens`: tokenized version of each sentence, ready for embedding.  

This preprocessing ensures that downstream embedding models (BERT-based or others) work on **clean, consistent, and semantically meaningful text**, leading to improved classification and clustering performance.  



Optional POS/NER analysis (Hazm/SpaCy)

Trainable Seq2Seq LSTM (with/without Attention)

Evaluation using sacreBLEU

---

## 📦 Requirements


- Python ≥ 3.10
- [PyTorch](https://pytorch.org/get-started/locally/)
- Jupyter Notebook/Lab (optional for EDA)

---

## ▶️ How to Run

```bash
# After clone repo
cd dual-translation-systems
source .venv/bin/activate
run jupyter notebook


```

