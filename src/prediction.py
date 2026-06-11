import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class HarassmentPredictor:
    def __init__(self, model_path='models/harassment_model.pkl', 
                 vectorizer_path='models/vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.loaded_successfully = False
        
        # SAFE GREETINGS (NEVER flag these)
        self.safe_greetings = {
            'hi', 'hii', 'hiii', 'hiiii', 'hiiiii', 'hiiiiii', 
            'hey', 'heyy', 'heyyy', 'hello', 'hola', 'yo', 
            'sup', 'whatsup', 'wassup', 'hiya', 'howdy',
            'greetings', 'good morning', 'good afternoon', 'good evening'
        }
        
        self.safe_how_are_you = {
            'how are you', 'how r u', 'how ru', 'how are u', 
            'how u doing', 'how you doing', 'how do you do', 
            'howre you', 'how r u doing', 'hw r u', 'hw r u doing'
        }
        
        self.safe_responses = {
            'good', 'great', 'fine', 'ok', 'okay', 'alright', 
            'not bad', 'doing well', 'im good', 'i am good'
        }
        
        # COMPREHENSIVE symbol mapping
        self.symbol_map = {
            '*': '', '!': 'i', '@': 'a', '$': 's', '%': '',
            '&': 'and', '?': '', '.': '', ',': '', ';': '',
            ':': '', '-': '', '_': '', '+': '', '=': '',
            '~': '', '`': '', "'": '', '"': '',
            '|': 'i', '\\': '', '/': '', '(': '', ')': '',
            '{': '', '}': '', '[': '', ']': '',
            '#': '', '^': '',
            '$$': 'ss', '$$$': 'sss', '@@': 'aa', '!!': 'ii',
            '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
            '5': 's', '6': 'g', '7': 't', '8': 'a', '9': 'g',
            '<': '', '>': '', '=': '',
        }
        
        # Special patterns for leetspeak
        self.special_patterns = {
            # F-word variations
            'f8ck': 'fuck', 'f8': 'fu', 'f8k': 'fuk', 'f8ker': 'fucker',
            'f$$k': 'fuck', 'f$$k off': 'fuck off', 'f$$k you': 'fuck you',
            'f$ck': 'fuck', 'f$%k': 'fuck', 'f*ck': 'fuck', 'f@ck': 'fuck',
            'f#ck': 'fuck', 'f^ck': 'fuck', 'f&ck': 'fuck',
            'f*ck you': 'fuck you',
            'f*** you': 'fuck you', 
            'f**k you': 'fuck you',
            'f**k': 'fuck',
            'ill murder you': 'i will murder you',
            "i'll murder you": 'i will murder you',
            "iill murder you": 'i will murder you',
            
            # Hoe/Whore variations
            'hoe': 'whore',
            'ho': 'whore',
            'h0e': 'whore',
            'h@e': 'whore',
            
            # "You" variations
            'you stupid': 'you are stupid',
            'you r stupid': 'you are stupid',
            'u stupid': 'you are stupid',
            'you r garbage': 'you are garbage',
            'u r garbage': 'you are garbage',
            'you r trash': 'you are trash',
            'u r trash': 'you are trash',
            'you r worthless': 'you are worthless',
            'u r worthless': 'you are worthless',
            'you hoe': 'you whore',
            'you ho': 'you whore',
            'u hoe': 'you whore',
            
            # B-word variations
            'b8ch': 'bitch', 'b8tch': 'bitch', 'b$$ch': 'bitch',
            'b$tch': 'bitch', 'b!tch': 'bitch', 'b@tch': 'bitch',
            'b1tch': 'bitch', 'b#tch': 'bitch',
            
            # Sh-word variations
            'sh8t': 'shit', 'sh$$t': 'shit', 'sh$t': 'shit',
            'sh!t': 'shit', 'sh@t': 'shit', 'sh1t': 'shit',
            '$hut': 'shut',
            
            # Ass variations
            '@$$': 'ass', '@$': 'as', '@ss': 'ass',
            'ur': 'your', 'u r': 'your',
            
            # Full phrases
            '$hut ur @$$': 'shut your ass',
            'shut ur ass': 'shut your ass',
            'shut your @$$': 'shut your ass',
            
            # C-word variations
            'c8nt': 'cunt', 'c$$nt': 'cunt', 'c$nt': 'cunt',
            'c!nt': 'cunt', 'c@nt': 'cunt', 'c1nt': 'cunt',
            
            # Wh-word variations
            'wh8re': 'whore', 'wh$$re': 'whore', 'wh$re': 'whore',
            'wh!re': 'whore', 'wh@re': 'whore',
            
            # Sl-word variations
            'sl8t': 'slut', 'sl$$t': 'slut', 'sl$t': 'slut',
            'sl!t': 'slut', 'sl@t': 'slut',
            
            # N-word variations
            'n8gga': 'nigga', 'n8gger': 'nigger', 'n$$gga': 'nigga',
            'n$gga': 'nigga', 'n!gga': 'nigga', 'n@gga': 'nigga',
            
            # Bastard variations
            'b4st4rd': 'bastard', 'b4stard': 'bastard', 'ba5tard': 'bastard',
            'b@stard': 'bastard', 'b*stard': 'bastard', 'b$$tard': 'bastard',
            
            # Randi variations
            'r4ndi': 'randi', 'r@ndi': 'randi', 'r*ndi': 'randi',
            'r$$ndi': 'randi', 'r$ndi': 'randi',
        }
        
        # COMPLETE OFFENSIVE WORDS LIST
        self.offensive_variations = {
            # Core profanity
            'fuck': ['fuck', 'fuk', 'fck', 'f*ck', 'f@ck', 'f#ck', 'fuuck', 
                     'f8ck', 'fuc8', 'f8k', 'f$$k', 'f$ck', 'f$%k', 'f u c k', 'f.u.c.k',
                     'fuck off', 'fuck you', 'f*ck you', 'f*** you', 'f**k you'],
            'bitch': ['bitch', 'b!tch', 'b@tch', 'b1tch', 'biatc', 'bytch', 
                      'biatch', 'b*tch', 'b8tch', 'b8ch', 'b$$ch', 'b$tch', 'b i t c h'],
            'whore': ['whore', 'wh0re', 'wh@re', 'wh*re', 'whoore', 'wh8re', 
                      'wh$$re', 'wh$re', 'w h o r e', 'hoe', 'ho', 'h0e', 'h@e'],
            'slut': ['slut', 'sl*t', 'sl@t', 'sl0t', 'slt', 'sl8t', 'sl$$t', 'sl$t', 's l u t'],
            'cunt': ['cunt', 'c*nt', 'c@nt', 'c0nt', 'kunt', 'c8nt', 'c$$nt', 'c$nt', 'c u n t'],
            'asshole': ['asshole', 'a$$hole', '@sshole', 'ashole', 'assh0le', 'a s s h o l e'],
            'ass': ['ass', '@$$', '@$', 'a$$', 'a s s'],
            'bastard': ['bastard', 'b@stard', 'b*stard', 'b4stard', 'b4st4rd', 
                        'ba5tard', 'b$$tard', 'b a s t a r d', 'bastrd', 'bastid', 'bastad'],
            'dick': ['dick', 'd*ck', 'd@ck', 'd1ck', 'd!ck', 'd8ck', 'd$$ck', 'd i c k'],
            'pussy': ['pussy', 'p*ssy', 'p@ssy', 'pu$$y', 'p8ssy', 'p u s s y'],
            'cock': ['cock', 'c*ck', 'c@ck', 'c0ck', 'c8ck', 'c$$ck', 'c o c k'],
            'shit': ['shit', 'sh*t', 'sh@t', 'sh1t', 'sh8t', 'sh$$t', 'sh$t', 's h i t'],
            'piss': ['piss', 'p*ss', 'p@ss', 'p1ss', 'p i s s'],
            'damn': ['damn', 'd@mn', 'd*amn', 'd a m n'],
            
            # Insults
            'shut up': ['shut up', 'shut ur mouth', 'shut your mouth', 'shut it'],
            'shut your ass': ['shut your ass', 'shut ur @$$', '$hut ur @$$', 'shut ur ass'],
            
            # Murder/death threats
            'murder': ['ill murder you', "i'll murder you", 'i will murder you', 'gonna murder you'],
            'kill': ['i will kill you', 'gonna kill you', 'going to kill you'],
            
            # "Freak" as harassment
            'freak': ['freak', 'fr3ak', 'fr34k', 'f r e a k', 'freak!', 'freaK'],
            
            # South Asian slurs
            'randi': ['randi', 'r4ndi', 'r@ndi', 'r*ndi', 'r$$ndi', 'r$ndi', 'r a n d i', 'randii', 'randy'],
            'chutiya': ['chutiya', 'chutia', 'ch*tiya', 'ch@tiya', 'c h u t i y a'],
            'bhosdi': ['bhosdi', 'bhosri', 'bhosdike', 'bhosdika', 'b h o s d i'],
            'madarchod': ['madarchod', 'madarchodh', 'maderchod', 'm a d a r c h o d'],
            'behenchod': ['behenchod', 'behenchodo', 'bhenchod', 'b e h e n c h o d'],
            
            # Racial slurs
            'nigga': ['nigga', 'n1gga', 'n!gga', 'n@gga', 'n*gga', 'niger', 
                      'n8gga', 'n$$gga', 'n$gga', 'n i g g a'],
            'nigger': ['nigger', 'n1gger', 'n!gger', 'n@gger', 'n8gger', 'n i g g e r'],
            'faggot': ['faggot', 'fag', 'fagg0t', 'f@ggot', 'f*ggot', 'f8ggot', 'f a g g o t'],
            'retard': ['retard', 'r3tard', 'r*tard', 'r@tard', 'r8tard', 'r e t a r d'],
            'tranny': ['tranny', 'tr*nny', 'tr@nny', 'trany', 't r a n n y'],
            'chink': ['chink', 'ch1nk', 'ch!nk', 'c h i n k'],
            'spic': ['spic', 'sp1c', 'sp!c', 's p i c'],
            'kike': ['kike', 'k1ke', 'k!ke', 'k i k e'],
            'gook': ['gook', 'g00k', 'g*ok', 'g@ok', 'g o o k'],
        }
        
        # SAFE WORDS
        self.safe_words = {
            'duck', 'luck', 'truck', 'stuck', 'pluck', 'cluck',
            'pitch', 'switch', 'ditch', 'stitch', 'snitch', 'glitch', 'witch',
            'feature', 'future', 'culture', 'picture', 'nature',
            'track', 'crack', 'black', 'stack', 'pack',
            'click', 'brick', 'stick', 'thick', 'quick',
            'pick', 'kick', 'sick', 'wick', 'trick',
            'block', 'clock', 'lock', 'rock', 'shock',
            'spark', 'park', 'mark', 'dark',
        }
        
        # Context patterns for 'freak' - when it's NOT harassment
        self.freak_safe_contexts = [
            'freak accident', 'freak weather', 'freak storm', 'freak wave',
            'freak occurrence', 'freak event', 'freak coincidence',
            'control freak', 'neat freak', 'health freak', 'fitness freak',
            'freak of nature', 'freak show', 'circus freak',
            'i am a freak', 'im a freak', 'i\'m a freak',
        ]
        
        # Insult patterns
        self.insult_patterns = [
            'you are stupid', 'you are a loser', 'nobody likes you',
            'everyone hates you', 'you are worthless', 'you are pathetic',
            'you are garbage', 'kill yourself', 'go die', 'you should die',
            'you bitch', 'fuck you', 'fuck off', 'you are a freak', 
            'you\'re a freak', 'you freak', 'what a freak', 'such a freak',
            'shut up', 'shut your mouth', 'shut your ass', 'shut ur ass',
            'f*ck you', 'f*** you', 'f**k you',
            'ill murder you', "i'll murder you", 'i will murder you',
            'i will kill you', 'gonna kill you', 'going to kill you',
            'you are garbage', 'you are trash', 'you are worthless',
            # NEW ADDITIONS
            'you stupid', 'you r stupid', 'u stupid',
            'you hoe', 'you ho', 'u hoe',
            'you whore', 'u whore',
            'stupid hoe', 'stupid ho',
            'you r garbage', 'u r garbage',
            'you r trash', 'u r trash',
            'you r worthless', 'u r worthless',
        ]
        
        self.load_or_create_model()
    
    def normalize_text(self, text):
        """Aggressive text normalization - handles ALL symbols"""
        if not text:
            return text
        
        original_text = text.lower()
        text = original_text
        
        # FIRST: Apply special patterns
        for pattern, replacement in self.special_patterns.items():
            text = text.replace(pattern, replacement)
        
        # SECOND: Handle abbreviations
        text = re.sub(r'\bur\b', 'your', text)
        text = re.sub(r'\bu r\b', 'your', text)
        text = re.sub(r'\bu\b', 'you', text)
        text = re.sub(r'\br\b', 'are', text)
        
        # THIRD: Apply general symbol mapping
        for symbol, replacement in self.symbol_map.items():
            text = text.replace(symbol, replacement)
        
        # FOURTH: Remove any remaining non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # FIFTH: Handle repeated letters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # SIXTH: Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_safe_word(self, text):
        """Check if text contains safe words"""
        text_lower = text.lower()
        normalized = self.normalize_text(text_lower)
        
        for safe_word in self.safe_words:
            if re.search(r'\b' + re.escape(safe_word) + r'\b', normalized):
                return True
        return False
    
    def is_freak_safe_context(self, text):
        """Check if 'freak' is used in a non-harassing context"""
        text_lower = text.lower()
        
        for context in self.freak_safe_contexts:
            if context in text_lower:
                return True
        return False
    
    def contains_offensive_word(self, text):
        """Check if text contains offensive words"""
        text_lower = text.lower()
        
        # First check if it's a safe word
        if self.is_safe_word(text_lower):
            return []
        
        # Normalize the text aggressively
        normalized = self.normalize_text(text_lower)
        
        detected = []
        
        # Special handling for 'freak'
        if 'freak' in normalized or 'freak' in text_lower:
            if not self.is_freak_safe_context(text_lower):
                if 'freak' not in detected:
                    detected.append('freak')
        
        # Check each offensive word
        for word, variations in self.offensive_variations.items():
            if word == 'freak':
                continue
            
            # Check normalized text
            if word in normalized:
                if re.search(r'\b' + re.escape(word) + r'\b', normalized):
                    if not self.is_safe_word(word):
                        if word not in detected:
                            detected.append(word)
                        continue
            
            # Check variations
            for variation in variations:
                if variation in text_lower:
                    if not self.is_safe_word(variation):
                        if word not in detected:
                            detected.append(word)
                        break
                
                # Check normalized variation
                norm_var = self.normalize_text(variation)
                if norm_var and norm_var in normalized:
                    if not self.is_safe_word(norm_var):
                        if word not in detected:
                            detected.append(word)
                        break
        
        return detected
    
    def contains_insult(self, text):
        """Check if text contains insult patterns"""
        text_lower = text.lower()
        normalized = self.normalize_text(text_lower)
        
        for insult in self.insult_patterns:
            if insult in normalized:
                return True, insult
            # Check without spaces
            if insult.replace(' ', '') in normalized.replace(' ', ''):
                return True, insult
        return False, None
    
    def is_safe_greeting(self, text):
        """Check if text is a friendly greeting"""
        text_lower = text.lower().strip()
        if text_lower in self.safe_greetings:
            return True
        return False
    
    def is_safe_question(self, text):
        """Check if text is asking 'how are you'"""
        text_lower = text.lower().strip()
        if text_lower in self.safe_how_are_you:
            return True
        return False
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.loaded_successfully = True
                print("✅ Model loaded successfully!")
            else:
                print("⚠️ Creating new model...")
                self.create_model()
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Creating new model...")
            self.create_model()
    
    def create_model(self):
        """Create a new model with training data"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        texts = [
            # Harassment examples (1)
            'you are stupid', 'kill yourself', 'fuck you', 'you bitch',
            'nobody likes you', 'you should die', 'bitch', 'whore', 'slut',
            'nigga', 'nigger', 'faggot', 'retard', 'cunt', 'asshole',
            'fuck', 'shit', 'dick', 'pussy', 'cock', 'f*ck', 'b1tch',
            'f8ck', 'b8tch', 'n8gga', 'bastard', 'b@stard',
            'you are a freak', 'you freak', 'what a freak',
            'randi', 'chutiya', 'bhosdi', 'madarchod',
            'fuck off', 'f$$k off', 'f$ck off',
            '$hut ur @$$', 'shut your ass', 'shut ur ass',
            'f*ck you', 'ill murder you', 'i will murder you',
            'you are garbage', 'you are trash',
            # NEW ADDITIONS
            'you stupid', 'you r stupid', 'u stupid',
            'you hoe', 'you ho', 'stupid hoe',
            'you r garbage', 'u r garbage',
            
            # Safe examples (0)
            'hi', 'hello', 'how are you', 'good morning', 'hw r u',
            'i feel stupid', 'this is killing me', 'you are crazy good',
            'great job', 'thanks', 'good', 'fine', 'hey', 'whats up',
            'how r u', 'hiiii', 'heyyy', 
            'freak accident', 'control freak', 'neat freak',
            'feature', 'future', 'culture', 'duck', 'luck', 'truck'
        ]
        
        labels = [1] * 48 + [0] * 28
        
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(random_state=42, class_weight='balanced')
        self.model.fit(X, labels)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        self.loaded_successfully = True
        print("✅ Model created successfully!")
    
    def predict(self, text):
        """Predict if text is harassment"""
        try:
            if not text or len(text.strip()) <= 1:
                return {
                    'text': text,
                    'is_harassment': False,
                    'confidence': 0.99,
                    'harassment_probability': 0.01,
                    'detection_method': 'empty_text'
                }
            
            text_original = text
            text_lower = text.lower().strip()
            normalized_text = self.normalize_text(text_lower)
            
            # 1. Check safe greetings
            if self.is_safe_greeting(text_lower):
                return {
                    'text': text_original,
                    'is_harassment': False,
                    'confidence': 0.99,
                    'harassment_probability': 0.01,
                    'detection_method': 'safe_greeting'
                }
            
            # 2. Check safe questions
            if self.is_safe_question(text_lower):
                return {
                    'text': text_original,
                    'is_harassment': False,
                    'confidence': 0.99,
                    'harassment_probability': 0.01,
                    'detection_method': 'safe_question'
                }
            
            # 3. Check for insults (HIGH PRIORITY)
            is_insult, insult_text = self.contains_insult(text_lower)
            if is_insult:
                return {
                    'text': text_original,
                    'normalized_text': normalized_text,
                    'is_harassment': True,
                    'confidence': 0.98,
                    'harassment_probability': 0.98,
                    'detection_method': 'insult_detected',
                    'insult': insult_text
                }
            
            # 4. Check for offensive words
            offensive_words = self.contains_offensive_word(text_lower)
            if offensive_words:
                return {
                    'text': text_original,
                    'normalized_text': normalized_text,
                    'is_harassment': True,
                    'confidence': 0.95,
                    'harassment_probability': 0.95,
                    'detection_method': 'offensive_word_detected',
                    'offensive_words': offensive_words
                }
            
            # 5. Use ML model as fallback
            if self.loaded_successfully:
                vec = self.vectorizer.transform([normalized_text])
                proba = self.model.predict_proba(vec)[0]
                harassment_prob = float(proba[1])
                is_harassment = harassment_prob > 0.60
                
                return {
                    'text': text_original,
                    'normalized_text': normalized_text,
                    'is_harassment': is_harassment,
                    'confidence': max(harassment_prob, 1-harassment_prob),
                    'harassment_probability': harassment_prob,
                    'detection_method': 'ml_model'
                }
            
            return {
                'text': text_original,
                'is_harassment': False,
                'confidence': 0.70,
                'harassment_probability': 0.30,
                'detection_method': 'default_safe'
            }
                
        except Exception as e:
            return {
                'text': text,
                'is_harassment': False,
                'confidence': 0.0,
                'harassment_probability': 0.0,
                'detection_method': 'error',
                'error': str(e)
            }
    
    def predict_batch(self, texts):
        """Predict for multiple texts"""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    predictor = HarassmentPredictor()
    
    test_cases = [
        # CRITICAL - Must be HARASSMENT
        ("kill yourself", True),
        ("you stupid", True),
        ("you r stupid", True),
        ("you stupid hoe", True),
        ("you hoe", True),
        ("stupid hoe", True),
        ("you r garbage", True),
        ("you are garbage", True),
        ("f*ck you", True),
        ("ill murder you", True),
        ("bastard", True),
        
        # Safe - Must be SAFE
        ("hi", False),
        ("how are you", False),
        ("control freak", False),
    ]
    
    print("\n" + "="*70)
    print("COMPREHENSIVE HARASSMENT DETECTION TEST")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        result = predictor.predict(text)
        actual = result['is_harassment']
        is_correct = (actual == expected)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        expected_str = "HARASS" if expected else "SAFE"
        actual_str = "HARASS" if actual else "SAFE"
        print(f"{status} '{text:25}' -> {actual_str:6} (expected: {expected_str:6}) | {result['detection_method']}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️ Some tests failed")
    print("="*70)