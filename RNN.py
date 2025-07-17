import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd

file = pd.read_csv("100_Unique_QA_Dataset.csv")

# Tokenize
def tokanize(text):
    text = text.lower()
    text = text.replace("?", "")
    text = text.replace("'", "")
    return text.split()

# Vocabulary
vocab = {'<UNK>': 0}

def build_vocab(row):
    tokanize_question = tokanize(row['question']) 
    tokanize_answer = tokanize(row['answer']) 
    merge_token = tokanize_question + tokanize_answer
    for token in merge_token:
        if token not in vocab:
            vocab[token] = len(vocab)

file.apply(build_vocab, axis=1)

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokanize(text)]

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question_indices = text_to_indices(self.df.iloc[index]['question'], self.vocab)
        answer_indices = text_to_indices(self.df.iloc[index]['answer'], self.vocab)
        return torch.tensor(question_indices), torch.tensor(answer_indices)

datasets = QADataset(file, vocab)
Data_Loader = DataLoader(datasets, batch_size=1, shuffle=True)

# ----------------------------
# ✅ FIXED RNN (your structure)
# ----------------------------
class myRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, question):
        embedded_question = self.embedding(question)         # (B, T, 50)
        output, hidden = self.rnn(embedded_question)         # hidden: (1, B, 64)
        final_hidden = hidden.squeeze(0)                     # (B, 64)
        output = self.fc(final_hidden)                       # (B, vocab_size)
        return output

model = myRNN(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# ✅ FIXED TRAINING LOOP
# ----------------------------
epochs = 20
for epoch in range(epochs):
    total_loss = 0.0
    for question, answer in Data_Loader:
        optimizer.zero_grad()
        
        outputs = model(question)       # (B, vocab_size)
        target = answer[:, 0]           # ✅ Use first token of answer as label (B,)
        
        loss = criterion(outputs, target)  # ✅ Fix loss shape mismatch
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")


def pridict(model, question, threshold=0.5):
    # Convert question to indices
    numrical_question = text_to_indices(question, vocab)
    
    # Make tensor and add batch dimension
    question_tensor = torch.tensor(numrical_question).unsqueeze(0)
    
    # Model prediction
    output = model(question_tensor)
    
    # Convert logits to probabilities
    probs = nn.functional.softmax(output, dim=1)
    
    # Get max probability and index
    value, index = torch.max(probs, dim=1)
    
    # Check if below threshold
    if value.item() < threshold:
        print("I don't know.")
        return

    # Convert vocab index to word
    idx2word = {v: k for k, v in vocab.items()}  # reverse vocab
    predicted_word = idx2word.get(index.item(), "<UNK>")
    
    print(predicted_word)


pridict(model, "What is the freezing point of water in Fahrenheit?")