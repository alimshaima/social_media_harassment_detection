import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 1]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, filepath=None):
        """Load and preprocess the dataset with BALANCED examples"""
        
        # BALANCED DATASET
        sample_data = {
            'text': [
                # HARASSMENT EXAMPLES (1)
                "You are so stupid and worthless",
                "Go kill yourself you pathetic loser",
                "Nobody likes you, you should disappear",
                "You're such a failure at everything",
                "I hope you get hit by a car",
                "You're too ugly to be seen in public",
                "Fuck you bitch",
                "Everyone hates you",
                "You should just die",
                "What a stupid idiot you are",
                "Go jump off a bridge",
                "You're worthless garbage",
                "Kill yourself",
                "You bitch",
                "You are stupid",
                
                # NON-HARASSMENT EXAMPLES (0)
                "I love this beautiful day",
                "Great job on the project!",
                "Looking forward to our meeting tomorrow",
                "Thanks for your help with this",
                "Have a wonderful day!",
                "Congratulations on your achievement!",
                "I feel so stupid today",
                "This game is fucking awesome",
                "I'm going to kill this workout",
                "This traffic is killing me",
                "You're crazy good at this!",
                "I hate how good you are",
                "Bro you're killing it today",
                "That movie was stupid funny",
                "My computer is so stupid",
                "This is driving me crazy",
                "Omg I'm dying of laughter",
                "You're a beast at this game",
                "Hi",
                "Hello",
                "How are you",
                "hw r u",
                "Good morning"
            ],
            'label': [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Save raw data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/raw_data.csv', index=False)
        
        # Apply text cleaning
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Save processed data
        df.to_csv('data/processed_data.csv', index=False)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Harassment: {df['label'].sum()} samples")
        print(f"   Non-Harassment: {len(df) - df['label'].sum()} samples")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        X = df['cleaned_text']
        y = df['label']
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)