import torch
import torch.nn as nn
import stanza
from transformers import pipeline
import re

# Initialize Stanza pipelines for preprocessing
stanza.download('en', processors='tokenize,pos,ner', logging_level='INFO')
stanza.download('fa', processors='tokenize,pos', logging_level='INFO')
nlp_en = stanza.Pipeline('en', processors='tokenize,pos,ner', verbose=False, logging_level='INFO')
nlp_fa = stanza.Pipeline('fa', processors='tokenize,pos', verbose=False, logging_level='INFO')

# Initialize Hugging Face NER pipeline for Persian
try:
    nlp_fa_ner = pipeline("ner", model="HooshvareLab/bert-fa-base-uncased-ner-arman", 
                         tokenizer="HooshvareLab/bert-fa-base-uncased-ner-arman", 
                         device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Hugging Face NER initialization failed: {e}")
    nlp_fa_ner = None

# Load vocabularies
vocab_data = torch.load('vocabs.pth')
en_vocab = vocab_data['en_vocab']
fa_vocab = vocab_data['fa_vocab']
en_pos_vocab = vocab_data['en_pos_vocab']
fa_pos_vocab = vocab_data['fa_pos_vocab']
ner_vocab = vocab_data['ner_vocab']

# Model architecture (must match training script)
class Encoder(nn.Module):
    def __init__(self, en_vocab_size, en_pos_size, ner_size, embed_dim=128, hidden_dim=256, dropout=0.1, num_layers=1):
        super().__init__()
        self.token_embed = nn.Embedding(en_vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(en_pos_size, 32)
        self.ner_embed = nn.Embedding(ner_size, 32)
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
        self.pos_embed = nn.Embedding(fa_pos_size, 32)
        self.ner_embed = nn.Embedding(ner_size, 32)
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
    
    def forward(self, src_token, src_pos, src_ner, max_len=50):
        batch_size = src_token.shape[0]
        device = src_token.device
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = []
        hidden, cell = self.encoder(src_token, src_pos, src_ner)
        
        input = torch.tensor([[fa_vocab['<sos>']]] * batch_size, device=device)
        pos = torch.tensor([[fa_pos_vocab['<pad>']]] * batch_size, device=device)
        ner = torch.tensor([[ner_vocab['O']]] * batch_size, device=device)
        
        for _ in range(max_len):
            output, hidden, cell = self.decoder(input, pos, ner, hidden, cell)
            pred = output.argmax(2)
            outputs.append(pred.squeeze(1))
            input = pred
            if (pred == fa_vocab['<eos>']).all():
                break
                
        return torch.stack(outputs, dim=1)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
encoder = Encoder(len(en_vocab), len(en_pos_vocab), len(ner_vocab)).to(device)
decoder = Decoder(len(fa_vocab), len(fa_pos_vocab), len(ner_vocab)).to(device)
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load('seq2seq_lstm.pth'))
model.eval()

def clean_english(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r'\s+', ' ', text.strip())
    return text if len(text) > 0 else None

def process_english(text):
    text = clean_english(text)
    if not text:
        return None, None, None
    
    doc = nlp_en(text)
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
        return None, None, None
        
    token_ids = [en_vocab['<sos>']] + [en_vocab.get(t, en_vocab['<unk>']) for t in tokens] + [en_vocab['<eos>']]
    pos_ids = [en_pos_vocab['<pad>']] + [en_pos_vocab.get(p, 0) for p in pos] + [en_pos_vocab['<pad>']]
    ner_ids = [ner_vocab['O']] + [ner_vocab.get(n, ner_vocab['O']) for n in ner] + [ner_vocab['O']]
    
    return token_ids, pos_ids, ner_ids

def translate_sentence(sentence):
    # Preprocess input
    token_ids, pos_ids, ner_ids = process_english(sentence)
    if token_ids is None:
        return "Invalid input sentence"
    
    # Convert to tensors
    src_token = torch.tensor([token_ids], dtype=torch.long).to(device)
    src_pos = torch.tensor([pos_ids], dtype=torch.long).to(device)
    src_ner = torch.tensor([ner_ids], dtype=torch.long).to(device)
    
    # Generate translation
    with torch.no_grad():
        output = model(src_token, src_pos, src_ner)
    
    # Convert output to text
    idx2word = {v: k for k, v in fa_vocab.items()}
    translated_tokens = [idx2word.get(tok.item(), '<unk>') for tok in output[0] 
                        if tok.item() not in {0, fa_vocab['<sos>'], fa_vocab['<eos>']}]
    
    return ' '.join(translated_tokens)

# Example usage
if __name__ == "__main__":
    test_sentences = [
        "Hello, how are you today?",
        "I am going to the market.",
        "This is a beautiful day."
    ]
    
    for sentence in test_sentences:
        translation = translate_sentence(sentence)
        print(f"English: {sentence}")
        print(f"Persian: {translation}\n")
