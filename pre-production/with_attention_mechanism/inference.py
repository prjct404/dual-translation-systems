import torch
import re
import stanza
import argparse
from collections import Counter

# Load Stanza pipelines for preprocessing
stanza.download('en', processors='tokenize,pos,ner', logging_level='INFO')
stanza.download('fa', processors='tokenize', logging_level='INFO')
nlp_en = stanza.Pipeline('en', processors='tokenize,pos,ner', verbose=False, logging_level='INFO')
nlp_fa = stanza.Pipeline('fa', processors='tokenize', verbose=False, logging_level='INFO')

# Model classes (copied from translation_task.py to ensure compatibility)
class Attention(torch.nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = torch.nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = torch.nn.Linear(dec_hid_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return torch.nn.functional.softmax(attention, dim=1)

class Encoder(torch.nn.Module):
    def __init__(self, en_vocab_size, en_pos_size, ner_size, embed_dim=256, hidden_dim=512, dropout=0.5, num_layers=2):
        super().__init__()
        self.token_embed = torch.nn.Embedding(en_vocab_size, embed_dim)
        self.pos_embed = torch.nn.Embedding(en_pos_size, 64)
        self.ner_embed = torch.nn.Embedding(ner_size, 64)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(embed_dim + 64 + 64, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.fc_hidden = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.num_layers = num_layers
    def forward(self, token, pos, ner):
        te = self.token_embed(token)
        pe = self.pos_embed(pos)
        ne = self.ner_embed(ner)
        x = torch.cat([te, pe, ne], dim=-1)
        x = self.dropout(x)
        outputs, (hidden, cell) = self.lstm(x)
        reduced_h = []
        reduced_c = []
        for layer in range(self.num_layers):
            fwd_h = hidden[2 * layer: 2 * layer + 1]
            bwd_h = hidden[2 * layer + 1: 2 * layer + 2]
            combined_h = torch.cat((fwd_h, bwd_h), dim=2)
            reduced_h.append(torch.tanh(self.fc_hidden(combined_h)))
            fwd_c = cell[2 * layer: 2 * layer + 1]
            bwd_c = cell[2 * layer + 1: 2 * layer + 2]
            combined_c = torch.cat((fwd_c, bwd_c), dim=2)
            reduced_c.append(torch.tanh(self.fc_cell(combined_c)))
        hidden = torch.cat(reduced_h, dim=0)
        cell = torch.cat(reduced_c, dim=0)
        return outputs, hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, fa_vocab_size, embed_dim=256, hidden_dim=512, dropout=0.5, num_layers=2):
        super().__init__()
        self.token_embed = torch.nn.Embedding(fa_vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = torch.nn.Linear(hidden_dim, fa_vocab_size)
    def forward(self, token, hidden, cell, encoder_outputs, mask):
        embedded = self.dropout(self.token_embed(token))
        a = self.attention(hidden[-1], encoder_outputs, mask)
        context = a.unsqueeze(1).bmm(encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src_token, src_pos, src_ner, max_len=50):
        batch_size = src_token.shape[0]
        fa_vocab_size = self.decoder.fc_out.out_features
        encoder_outputs, hidden, cell = self.encoder(src_token, src_pos, src_ner)
        mask = (src_token != 0)
        outputs = torch.zeros(batch_size, max_len, fa_vocab_size).to(src_token.device)
        input_token = torch.ones(batch_size, 1, dtype=torch.long).to(src_token.device) * 2 # <sos>
        preds = input_token.clone()
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t-1] = output
            pred = output.unsqueeze(1).argmax(2)
            preds = torch.cat((preds, pred), dim=1)
            input_token = pred
            if (pred == 3).all(): # <eos>
                break
        return preds

# Cleaning functions (from translation_task.py)
def clean_english(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r'\s+', ' ', text.strip())
    return text if len(text) > 0 else None

def clean_persian(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r'[\u0640\u200C]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text if len(text) > 0 else None

# Preprocessing function for English input
def process_english(text, nlp_en, use_pos_ner=True):
    text = clean_english(text)
    if text is None:
        return None, None, None
    try:
        doc = nlp_en(text)
        tokens, pos, ner = [], [], []
        for sent in doc.sentences:
            for word, ent in zip(sent.words, sent.ents if sent.ents else [None] * len(sent.words)):
                tokens.append(word.text)
                pos.append(word.upos if use_pos_ner else 'O')
                ner_tag = 'O'
                if use_pos_ner and ent is not None and hasattr(ent, 'type'):
                    ner_tag = ent.type
                ner.append(ner_tag)
        if len(tokens) < 3:
            return None, None, None
        return tokens, pos, ner
    except Exception as e:
        print(f"Error processing English: {text[:50]}... {e}")
        return None, None, None

# Convert tokens to IDs
def tokens_to_ids(tokens, pos, ner, en_vocab, en_pos_vocab, ner_vocab):
    if tokens is None:
        return None, None, None
    token_ids = [en_vocab['<sos>']] + [en_vocab.get(t, en_vocab['<unk>']) for t in tokens] + [en_vocab['<eos>']]
    pos_ids = [en_pos_vocab['<pad>']] + [en_pos_vocab.get(p, en_pos_vocab['<pad>']) for p in pos] + [en_pos_vocab['<pad>']]
    ner_ids = [ner_vocab['O']] + [ner_vocab.get(n, ner_vocab['O']) for n in ner] + [ner_vocab['O']]
    return token_ids, pos_ids, ner_ids

# Main inference function
def translate_sentence(sentence, model, en_vocab, fa_vocab, en_pos_vocab, ner_vocab, device, max_len=50, use_pos_ner=True):
    model.eval()
    tokens, pos, ner = process_english(sentence, nlp_en, use_pos_ner)
    if tokens is None:
        return "Error: Unable to process input sentence."
    token_ids, pos_ids, ner_ids = tokens_to_ids(tokens, pos, ner, en_vocab, en_pos_vocab, ner_vocab)
    if token_ids is None:
        return "Error: Unable to convert tokens to IDs."
    
    # Convert to tensors and add batch dimension
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    pos_tensor = torch.tensor([pos_ids], dtype=torch.long).to(device)
    ner_tensor = torch.tensor([ner_ids], dtype=torch.long).to(device)
    
    # Perform inference
    with torch.no_grad():
        preds = model(token_tensor, pos_tensor, ner_tensor, max_len)
    
    # Convert predictions to text
    idx2word = {v: k for k, v in fa_vocab.items()}
    pred_ids = preds[0].tolist()
    try:
        eos_pos = pred_ids.index(fa_vocab['<eos>'])
        pred_ids = pred_ids[:eos_pos]
    except ValueError:
        pass
    translation = [idx2word.get(tok, '<unk>') for tok in pred_ids[1:]] # Skip <sos>
    return ' '.join(translation)

def main():
    parser = argparse.ArgumentParser(description="Translate English to Persian using trained Seq2Seq model")
    parser.add_argument('--model_path', type=str, default='seq2seq_lstm.pth', help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, default='vocabs.pth', help='Path to vocabularies')
    parser.add_argument('--sentence', type=str, help='English sentence to translate')
    parser.add_argument('--no_pos_ner', action='store_true', help='Disable POS/NER features')
    args = parser.parse_args()

    # Load vocabularies
    vocab_data = torch.load(args.vocab_path)
    en_vocab = vocab_data['en_vocab']
    fa_vocab = vocab_data['fa_vocab']
    en_pos_vocab = vocab_data['en_pos_vocab']
    ner_vocab = vocab_data['ner_vocab']
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(len(en_vocab), len(en_pos_vocab), len(ner_vocab)).to(device)
    decoder = Decoder(len(fa_vocab)).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    
    # Load trained model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Translate
    if args.sentence:
        translation = translate_sentence(
            args.sentence, 
            model, 
            en_vocab, 
            fa_vocab, 
            en_pos_vocab, 
            ner_vocab, 
            device, 
            use_pos_ner=not args.no_pos_ner
        )
        print(f"Input: {args.sentence}")
        print(f"Translation: {translation}")
    else:
        print("Please provide a sentence to translate using --sentence")

if __name__ == "__main__":
    main()