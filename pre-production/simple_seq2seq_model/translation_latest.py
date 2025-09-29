# Imports
import re
import time
import psutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset, Dataset as HFDataset
import stanza
import pandas as pd
import sacrebleu
from torch.optim.lr_scheduler import StepLR
from transformers import pipeline
import logging

# Set logging for detailed errors
logging.basicConfig(level=logging.INFO)

# Function to monitor memory usage
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024**3:.2f} GB")

# Load Stanza pipelines (run once)
stanza.download('en', processors='tokenize,pos,ner', logging_level='INFO')
stanza.download('fa', processors='tokenize,pos', logging_level='INFO')  # Note: Stanza does not support NER for Persian, so only tokenize and pos
nlp_en = stanza.Pipeline('en', processors='tokenize,pos,ner', verbose=False, logging_level='INFO')
nlp_fa = stanza.Pipeline('fa', processors='tokenize,pos', verbose=False, logging_level='INFO')

# Initialize Hugging Face NER pipeline for Persian
print("Initializing Hugging Face NER pipeline for Persian...")
try:
    nlp_fa_ner = pipeline("ner", model="HooshvareLab/bert-fa-base-uncased-ner-arman", tokenizer="HooshvareLab/bert-fa-base-uncased-ner-arman", device=0 if torch.cuda.is_available() else -1)
    print("Hugging Face NER pipeline initialized!")
except Exception as e:
    print(f"Hugging Face NER initialization failed: {e}")
    nlp_fa_ner = None

# Option to skip POS/NER for faster prototyping
USE_POS_NER = True  # Set to True to include POS/NER, False to skip

# Step 1: Load Dataset
print("Loading dataset...")
start_time = time.time()
train_data = load_dataset("Helsinki-NLP/opus-100", "en-fa", split='train')
val_data = load_dataset("Helsinki-NLP/opus-100", "en-fa", split='validation')
test_data = load_dataset("Helsinki-NLP/opus-100", "en-fa", split='test')

# Use smaller subsets for faster processing
train_data = train_data.select(range(10000))
val_data = val_data.select(range(500))
test_data = test_data.select(range(500))
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
print_memory_usage()

# Debug dataset structure
print("Sample dataset entry:", train_data[0])
print("Dataset features:", train_data.features)

def add_keys(example):
    english = [t['en'] for t in example['translation']]
    persian = [t['fa'] for t in example['translation']]
    return {'english': english, 'persian': persian}

print("Adding keys...")
start_time = time.time()
train_data = train_data.map(add_keys, batched=True, desc="Adding keys (train)")
val_data = val_data.map(add_keys, batched=True, desc="Adding keys (validation)")
test_data = test_data.map(add_keys, batched=True, desc="Adding keys (test)")
print(f"Keys added in {time.time() - start_time:.2f} seconds")
print_memory_usage()
print(f"Original sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Step 2: Cleaning
def clean_english(text):
    if not isinstance(text, str) or pd.isna(text):
        return None
    text = re.sub(r'\s+', ' ', text.strip())
    return text if len(text) > 0 else None

def clean_persian(text):
    if not isinstance(text, str) or pd.isna(text):
        return None
    text = re.sub(r'[\u0640\u200C]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text if len(text) > 0 else None

def apply_cleaning(example):
    english = [clean_english(text) for text in example['english']]
    persian = [clean_persian(text) for text in example['persian']]
    return {'english': english, 'persian': persian}

def process_dataset(data, split_name=""):
    print(f"Cleaning {split_name}...")
    start_time = time.time()
    data = data.map(apply_cleaning, batched=True, desc=f"Cleaning ({split_name})")
    data = data.filter(lambda x: x['english'] is not None and x['persian'] is not None)
    df = data.to_pandas()
    df.drop_duplicates(subset=['english'], inplace=True)
    df.drop_duplicates(subset=['persian'], inplace=True)
    data = HFDataset.from_pandas(df.reset_index(drop=True))
    print(f"Cleaning {split_name} completed in {time.time() - start_time:.2f} seconds")
    print_memory_usage()
    return data

train_data = process_dataset(train_data, "train")
val_data = process_dataset(val_data, "validation")
test_data = process_dataset(test_data, "test")
print(f"After cleaning & dedup: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Step 3: POS & NER Tagging
def process_english(example):
    if not USE_POS_NER:
        example['en_tokens'] = example['english'].split() if example['english'] else []
        example['en_pos'] = ['O'] * len(example['en_tokens'])
        example['en_ner'] = ['O'] * len(example['en_tokens'])
        return example
    try:
        doc = nlp_en(example['english'])
        tokens, pos, ner = [], [], []
        for sent in doc.sentences:
            for word, ent in zip(sent.words, sent.ents if sent.ents else [None] * len(sent.words)):
                tokens.append(word.text)
                pos.append(word.upos)
                ner_tag = 'O'
                if ent is not None and hasattr(ent, 'type'):
                    ner_tag = ent.type
                ner.append(ner_tag)
        if len(tokens) < 3:
            example['english'] = None
        example['en_tokens'] = tokens
        example['en_pos'] = pos
        example['en_ner'] = ner
    except Exception as e:
        print(f"Error processing English: {example['english'][:50]}... {e}")
        example['english'] = None
        example['en_tokens'] = []
        example['en_pos'] = []
        example['en_ner'] = []
    return example

def process_persian(example):
    if not USE_POS_NER:
        example['fa_tokens'] = example['persian'].split() if example['persian'] else []
        example['fa_pos'] = ['O'] * len(example['fa_tokens'])
        example['fa_ner'] = ['O'] * len(example['fa_tokens'])
        return example
    tokens, pos, ner = [], [], []
    try:
        doc = nlp_fa(example['persian'])
        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                pos.append(word.upos)
        # Since Stanza doesn't support Persian NER, use Hugging Face for NER
        if nlp_fa_ner:
            ner_results = nlp_fa_ner(example['persian'])
            ner_tags = ['O'] * len(tokens)
            for entity in ner_results:
                # Simple mapping: approximate token index based on character position
                token_idx = min(len(tokens) - 1, int(entity.get('start', 0) / max(1, len(example['persian']) / len(tokens))))
                ner_tags[token_idx] = entity['entity']
            ner = ner_tags
        else:
            ner = ['O'] * len(tokens)
        if len(tokens) < 3:
            example['persian'] = None
        example['fa_tokens'] = tokens
        example['fa_pos'] = pos
        example['fa_ner'] = ner
    except Exception as e:
        print(f"Error processing Persian: {example['persian'][:50]}... {e}")
        example['persian'] = None
        example['fa_tokens'] = []
        example['fa_pos'] = []
        example['fa_ner'] = []
    return example

def tag_dataset(data, split_name=""):
    start_time = time.time()
    if USE_POS_NER:
        print(f"Tagging English for {split_name}...")
        data = data.map(process_english, batched=False, desc=f"Tagging English ({split_name})")
        print(f"Tagging Persian for {split_name}...")
        data = data.map(process_persian, batched=False, desc=f"Tagging Persian ({split_name})")
    else:
        print(f"Skipping POS/NER tagging for {split_name}...")
        data = data.map(process_english, batched=False, desc=f"Tokenizing English ({split_name})")
        data = data.map(process_persian, batched=False, desc=f"Tokenizing Persian ({split_name})")
    data = data.filter(lambda x: x['english'] is not None and x['persian'] is not None)
    print(f"Tagging {split_name} completed in {time.time() - start_time:.2f} seconds")
    print_memory_usage()
    return data

train_data = tag_dataset(train_data, "train")
val_data = tag_dataset(val_data, "validation")
test_data = tag_dataset(test_data, "test")
print(f"After tagging & filtering: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Step 4: Build Vocabularies
VOCAB_SIZE = 10000
print("Building vocabularies...")
start_time = time.time()
all_en_tokens = [t for ex in train_data for t in ex['en_tokens']]
all_fa_tokens = [t for ex in train_data for t in ex['fa_tokens']]
all_en_pos = [p for ex in train_data for p in ex['en_pos']]
all_fa_pos = [p for ex in train_data for p in ex['fa_pos']]
all_ner = [n for ex in train_data for n in ex['en_ner'] + ex['fa_ner']]
en_token_counter = Counter(all_en_tokens)
fa_token_counter = Counter(all_fa_tokens)
en_pos_set = set(all_en_pos)
fa_pos_set = set(all_fa_pos)
ner_set = set(all_ner)
en_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
en_vocab.update({w: i + 4 for i, (w, _) in enumerate(en_token_counter.most_common(VOCAB_SIZE))})
fa_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
fa_vocab.update({w: i + 4 for i, (w, _) in enumerate(fa_token_counter.most_common(VOCAB_SIZE))})
en_pos_vocab = {'<pad>': 0}
en_pos_vocab.update({p: i + 1 for i, p in enumerate(en_pos_set)})
fa_pos_vocab = {'<pad>': 0}
fa_pos_vocab.update({p: i + 1 for i, p in enumerate(fa_pos_set)})
ner_vocab = {'<pad>': 0, 'O': 1}
ner_vocab.update({n: i + 2 for i, n in enumerate(ner_set - {'O'})})
print(f"Vocab sizes: EN tokens={len(en_vocab)}, FA tokens={len(fa_vocab)}, EN POS={len(en_pos_vocab)}, FA POS={len(fa_pos_vocab)}, NER={len(ner_vocab)}")
print(f"Vocabularies built in {time.time() - start_time:.2f} seconds")
print_memory_usage()

# Step 5: Convert to IDs
def to_ids(example):
    en_token_ids = [[en_vocab['<sos>']] + [en_vocab.get(t, en_vocab['<unk>']) for t in tokens] + [en_vocab['<eos>']] for tokens in example['en_tokens']]
    fa_token_ids = [[fa_vocab['<sos>']] + [fa_vocab.get(t, fa_vocab['<unk>']) for t in tokens] + [fa_vocab['<eos>']] for tokens in example['fa_tokens']]
    en_pos_ids = [[en_pos_vocab['<pad>']] + [en_pos_vocab.get(p, 0) for p in pos] + [en_pos_vocab['<pad>']] for pos in example['en_pos']]
    fa_pos_ids = [[fa_pos_vocab['<pad>']] + [fa_pos_vocab.get(p, 0) for p in pos] + [fa_pos_vocab['<pad>']] for pos in example['fa_pos']]
    en_ner_ids = [[ner_vocab['O']] + [ner_vocab.get(n, ner_vocab['O']) for n in ner] + [ner_vocab['O']] for ner in example['en_ner']]
    fa_ner_ids = [[ner_vocab['O']] + [ner_vocab.get(n, ner_vocab['O']) for n in ner] + [ner_vocab['O']] for ner in example['fa_ner']]
    return {
        'en_token_ids': en_token_ids,
        'fa_token_ids': fa_token_ids,
        'en_pos_ids': en_pos_ids,
        'fa_pos_ids': fa_pos_ids,
        'en_ner_ids': en_ner_ids,
        'fa_ner_ids': fa_ner_ids
    }

print("Converting to IDs...")
start_time = time.time()
train_data = train_data.map(to_ids, batched=True, desc="Converting to IDs (train)")
val_data = val_data.map(to_ids, batched=True, desc="Converting to IDs (validation)")
test_data = test_data.map(to_ids, batched=True, desc="Converting to IDs (test)")
print(f"ID conversion completed in {time.time() - start_time:.2f} seconds")
print_memory_usage()
print(f"Splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Step 7: PyTorch Dataset
class TranslationDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'en_token': torch.tensor(item['en_token_ids'], dtype=torch.long),
            'fa_token': torch.tensor(item['fa_token_ids'], dtype=torch.long),
            'en_pos': torch.tensor(item['en_pos_ids'], dtype=torch.long),
            'fa_pos': torch.tensor(item['fa_pos_ids'], dtype=torch.long),
            'en_ner': torch.tensor(item['en_ner_ids'], dtype=torch.long),
            'fa_ner': torch.tensor(item['fa_ner_ids'], dtype=torch.long),
        }

def collate_fn(batch):
    en_token = nn.utils.rnn.pad_sequence([item['en_token'] for item in batch], batch_first=True, padding_value=0)
    fa_token = nn.utils.rnn.pad_sequence([item['fa_token'] for item in batch], batch_first=True, padding_value=0)
    en_pos = nn.utils.rnn.pad_sequence([item['en_pos'] for item in batch], batch_first=True, padding_value=0)
    fa_pos = nn.utils.rnn.pad_sequence([item['fa_pos'] for item in batch], batch_first=True, padding_value=0)
    en_ner = nn.utils.rnn.pad_sequence([item['en_ner'] for item in batch], batch_first=True, padding_value=0)
    fa_ner = nn.utils.rnn.pad_sequence([item['fa_ner'] for item in batch], batch_first=True, padding_value=0)
    return {
        'en_token': en_token, 'fa_token': fa_token,
        'en_pos': en_pos, 'fa_pos': fa_pos,
        'en_ner': en_ner, 'fa_ner': fa_ner,
    }

print("Creating DataLoaders...")
start_time = time.time()
train_ds = TranslationDataset(train_data)
val_ds = TranslationDataset(val_data)
test_ds = TranslationDataset(test_data)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn, num_workers=4)
print(f"DataLoaders created in {time.time() - start_time:.2f} seconds")
print_memory_usage()

# Step 8: Seq2Seq LSTM Model
class Encoder(nn.Module):
    def __init__(self, en_vocab_size, en_pos_size, ner_size, embed_dim=128, hidden_dim=256, dropout=0.1, num_layers=1):
        super().__init__()
        self.token_embed = nn.Embedding(en_vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(en_pos_size, 32) if USE_POS_NER else nn.Embedding(2, 32)
        self.ner_embed = nn.Embedding(ner_size, 32) if USE_POS_NER else nn.Embedding(2, 32)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim + 32 + 32, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, token, pos, ner):
        te = self.token_embed(token)
        pe = self.pos_embed(pos)
        ne = self.ner_embed(ner)
        x = torch.cat([te, pe, ne], dim=-1)
        x = self.dropout(x)
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, fa_vocab_size, fa_pos_size, ner_size, embed_dim=128, hidden_dim=256, dropout=0.1, num_layers=1):
        super().__init__()
        self.token_embed = nn.Embedding(fa_vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(fa_pos_size, 32) if USE_POS_NER else nn.Embedding(2, 32)
        self.ner_embed = nn.Embedding(ner_size, 32) if USE_POS_NER else nn.Embedding(2, 32)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim + 32 + 32, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, fa_vocab_size)

    def forward(self, token, pos, ner, hidden, cell):
        te = self.token_embed(token)
        pe = self.pos_embed(pos)
        ne = self.ner_embed(ner)
        x = torch.cat([te, pe, ne], dim=-1)
        x = self.dropout(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_token, src_pos, src_ner, trg_token, trg_pos, trg_ner, teacher_forcing_ratio=0.5):
        batch_size = src_token.shape[0]
        trg_len = trg_token.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        hidden, cell = self.encoder(src_token, src_pos, src_ner)
        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(src_token.device)
        input = trg_token[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            pos_input = trg_pos[:, t-1].unsqueeze(1)
            ner_input = trg_ner[:, t-1].unsqueeze(1)
            output, hidden, cell = self.decoder(input, pos_input, ner_input, hidden, cell)
            outputs[:, t-1] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg_token[:, t].unsqueeze(1) if teacher_force else top1
        return outputs

# Instantiate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Instantiating model...")
start_time = time.time()
encoder = Encoder(len(en_vocab), len(en_pos_vocab), len(ner_vocab), embed_dim=128, hidden_dim=256, num_layers=1).to(device)
decoder = Decoder(len(fa_vocab), len(fa_pos_vocab), len(ner_vocab), embed_dim=128, hidden_dim=256, num_layers=1).to(device)
model = Seq2Seq(encoder, decoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
pad_idx = fa_vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
print(f"Model instantiated in {time.time() - start_time:.2f} seconds")
print_memory_usage()

# Step 9: Training and Evaluation
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        src_token = batch['en_token'].to(device)
        trg_token = batch['fa_token'].to(device)
        src_pos = batch['en_pos'].to(device)
        trg_pos = batch['fa_pos'].to(device)
        src_ner = batch['en_ner'].to(device)
        trg_ner = batch['fa_ner'].to(device)
        optimizer.zero_grad()
        output = model(src_token, src_pos, src_ner, trg_token, trg_pos, trg_ner)
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg_token[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)} processed")
            print_memory_usage()
    print(f"Epoch training completed in {time.time() - start_time:.2f} seconds")
    return epoch_loss / len(loader)

def eval_epoch(model, loader, criterion, device, fa_vocab):
    model.eval()
    epoch_loss = 0
    hyps, refs = [], []
    idx2word = {v: k for k, v in fa_vocab.items()}
    start_time = time.time()
   
    with torch.no_grad():
        for batch in loader:
            src_token = batch['en_token'].to(device)
            trg_token = batch['fa_token'].to(device)
            src_pos = batch['en_pos'].to(device)
            trg_pos = batch['fa_pos'].to(device)
            src_ner = batch['en_ner'].to(device)
            trg_ner = batch['fa_ner'].to(device)
            output = model(src_token, src_pos, src_ner, trg_token, trg_pos, trg_ner, teacher_forcing_ratio=0)
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg_token[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            hidden, cell = model.encoder(src_token, src_pos, src_ner)
            inputs = trg_token[:, 0].unsqueeze(1)
            eos_idx = fa_vocab['<eos>']
            max_len = trg_token.shape[1]
            for t in range(1, max_len):
                pos_input = trg_pos[:, t-1].unsqueeze(1)
                ner_input = trg_ner[:, t-1].unsqueeze(1)
                output, hidden, cell = model.decoder(inputs, pos_input, ner_input, hidden, cell)
                pred = output.argmax(2)
                inputs = pred
                if (pred == eos_idx).all():
                    break
            for i in range(src_token.shape[0]):
                hyp_tokens = [idx2word.get(tok.item(), '<unk>') for tok in inputs[i] if tok.item() not in {0, fa_vocab['<sos>'], fa_vocab['<eos>']}]
                ref_tokens = [idx2word.get(tok.item(), '<unk>') for tok in trg_token[i, 1:] if tok.item() not in {0, fa_vocab['<eos>']}]
                hyps.append(' '.join(hyp_tokens))
                refs.append([' '.join(ref_tokens)])
    bleu = sacrebleu.corpus_bleu(hyps, refs).score
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    print_memory_usage()
    return epoch_loss / len(loader), bleu

# Test preprocessing (print sample)
print("Sample processed example:")
sample = train_data[0]
print(f"English: {sample['english']}")
print(f"Persian: {sample['persian']}")
print(f"EN tokens: {sample['en_tokens']}")
print(f"FA tokens: {sample['fa_tokens']}")
print(f"EN POS: {sample['en_pos']}")
print(f"FA POS: {sample['fa_pos']}")
print(f"EN NER: {sample['en_ner']}")
print(f"FA NER: {sample['fa_ner']}")

# Train
NUM_EPOCHS = 3
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_bleu = eval_epoch(model, val_loader, criterion, device, fa_vocab)
    scheduler.step()
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val BLEU={val_bleu:.2f}, LR={scheduler.get_last_lr()[0]:.6f}")
    print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds")
    print_memory_usage()

# Save
print("Saving model and vocabularies...")
torch.save(model.state_dict(), 'seq2seq_lstm.pth')
torch.save({'en_vocab': en_vocab, 'fa_vocab': fa_vocab, 'en_pos_vocab': en_pos_vocab,
            'fa_pos_vocab': fa_pos_vocab, 'ner_vocab': ner_vocab}, 'vocabs.pth')

# Test
print("Testing model...")
test_loss, test_bleu = eval_epoch(model, test_loader, criterion, device, fa_vocab)
print(f"Test Loss={test_loss:.4f}, Test BLEU={test_bleu:.2f}")
