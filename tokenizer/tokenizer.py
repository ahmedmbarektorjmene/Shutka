"""
Custom tokenizer implementations for CPU-based training
Supports both byte-level and BPE tokenization
"""
import torch
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple
import json
import os


class SimpleByteTokenizer:
    """
    Simple byte-level tokenizer - maps each byte to a token ID
    Efficient for CPU training, no external dependencies
    """
    def __init__(self):
        # Reserve special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Byte tokens: 0-255 (but we reserve 0-3 for special tokens)
        # So we map bytes 0-251 to tokens 4-255, and bytes 252-255 to unk
        self.byte_to_token = {i: i + 4 for i in range(252)}
        self.token_to_byte = {v: k for k, v in self.byte_to_token.items()}
        
        # Add special tokens
        self.vocab_size = 256
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = [self.bos_token_id]
        for byte_val in text.encode('utf-8'):
            if byte_val < 252:
                tokens.append(self.byte_to_token[byte_val])
            else:
                tokens.append(self.unk_token_id)
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        bytes_list = []
        for tid in token_ids:
            if tid == self.bos_token_id or tid == self.pad_token_id:
                continue
            if tid == self.eos_token_id:
                break
            if tid in self.token_to_byte:
                bytes_list.append(self.token_to_byte[tid])
            elif tid == self.unk_token_id:
                bytes_list.append(0)  # Replace unknown with null byte
            else:
                # Skip invalid token IDs
                continue
        try:
            # Decode with error handling to replace invalid sequences
            decoded = bytes(bytes_list).decode('utf-8', errors='replace')
            # Remove replacement characters that can cause encoding issues
            decoded = decoded.replace('\ufffd', '')
            return decoded
        except Exception:
            return ""
    
    def save(self, path: str):
        """Save tokenizer state"""
        state = {
            'byte_to_token': self.byte_to_token,
            'token_to_byte': self.token_to_byte,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(state, f)
    
    def load(self, path: str):
        """Load tokenizer state"""
        with open(path, 'r') as f:
            state = json.load(f)
        self.byte_to_token = {int(k): v for k, v in state['byte_to_token'].items()}
        self.token_to_byte = {int(k): v for k, v in state['token_to_byte'].items()}
        self.vocab_size = state['vocab_size']


class BytePairTokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implemented from scratch
    More efficient than byte-level for code data
    """
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize with byte-level vocabulary
        self.word_to_tokens: Dict[str, List[int]] = {}
        self.token_to_word: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Initialize base vocabulary with bytes and special tokens
        self._init_base_vocab()
    
    def _init_base_vocab(self):
        """Initialize base vocabulary with bytes"""
        # Special tokens
        self.token_to_word[0] = '<pad>'
        self.token_to_word[1] = '<unk>'
        self.token_to_word[2] = '<bos>'
        self.token_to_word[3] = '<eos>'
        
        # Byte tokens
        for i in range(256):
            if i + 4 < self.vocab_size:
                self.token_to_word[i + 4] = bytes([i]).decode('latin-1')
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from texts"""
        word_freqs = Counter()
        for text in texts:
            # Split by whitespace and punctuation
            words = re.findall(r'\S+', text)
            word_freqs.update(words)
        return dict(word_freqs)
    
    def _get_pair_freqs(self, word_freqs: Dict[str, int]) -> Counter:
        """Get frequencies of adjacent byte pairs"""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            # Convert word to bytes
            byte_word = word.encode('utf-8')
            for i in range(len(byte_word) - 1):
                pair = (byte_word[i], byte_word[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def train(self, texts: List[str], num_merges: Optional[int] = None):
        """
        Train BPE tokenizer on texts
        If num_merges is None, merge until vocab_size is reached
        """
        if num_merges is None:
            num_merges = self.vocab_size - 256 - 4  # Reserve space for base vocab
        
        # Get word frequencies
        word_freqs = self._get_word_freqs(texts)
        
        # Initialize word_to_tokens: each word as list of byte IDs
        for word in word_freqs:
            byte_word = word.encode('utf-8')
            self.word_to_tokens[word] = [b + 4 for b in byte_word]
        
        # Perform merges
        for merge_idx in range(num_merges):
            if len(self.token_to_word) >= self.vocab_size:
                break
            
            # Find most frequent pair
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            
            best_pair = pair_freqs.most_common(1)[0][0]
            
            # Create new token
            new_token_id = len(self.token_to_word)
            if new_token_id >= self.vocab_size:
                break
            
            new_token = self.token_to_word[best_pair[0] + 4] + self.token_to_word[best_pair[1] + 4]
            self.token_to_word[new_token_id] = new_token
            self.merges.append((self.token_to_word[best_pair[0] + 4], 
                               self.token_to_word[best_pair[1] + 4]))
            
            # Update word_to_tokens
            for word in list(self.word_to_tokens.keys()):
                tokens = self.word_to_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and 
                        tokens[i] == best_pair[0] + 4 and 
                        tokens[i + 1] == best_pair[1] + 4):
                        new_tokens.append(new_token_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                self.word_to_tokens[word] = new_tokens
    
    def _apply_bpe(self, word: str) -> List[int]:
        """Apply BPE to a single word"""
        if word in self.word_to_tokens:
            return self.word_to_tokens[word]
        
        # Fallback: byte-level encoding
        byte_word = word.encode('utf-8')
        return [b + 4 if b < 252 else self.unk_token_id for b in byte_word]
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = [self.bos_token_id]
        
        # Split text into words (simple whitespace split)
        words = re.findall(r'\S+|\s+', text)
        for word in words:
            if word.isspace():
                # Encode whitespace as bytes
                for byte_val in word.encode('utf-8'):
                    if byte_val < 252:
                        tokens.append(byte_val + 4)
                    else:
                        tokens.append(self.unk_token_id)
            else:
                tokens.extend(self._apply_bpe(word))
        
        tokens.append(self.eos_token_id)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        text_parts = []
        for tid in token_ids:
            if tid == self.bos_token_id or tid == self.pad_token_id:
                continue
            if tid == self.eos_token_id:
                break
            if tid in self.token_to_word:
                text_parts.append(self.token_to_word[tid])
            else:
                text_parts.append('<unk>')
        
        # Join and decode
        try:
            return ''.join(text_parts).encode('latin-1').decode('utf-8', errors='replace')
        except:
            return ''.join(text_parts)
    
    def save(self, path: str):
        """Save tokenizer state"""
        state = {
            'vocab_size': self.vocab_size,
            'token_to_word': {str(k): v for k, v in self.token_to_word.items()},
            'word_to_tokens': {k: v for k, v in self.word_to_tokens.items()},
            'merges': self.merges
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load tokenizer state"""
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        self.vocab_size = state['vocab_size']
        self.token_to_word = {int(k): v for k, v in state['token_to_word'].items()}
        self.word_to_tokens = state['word_to_tokens']
        self.merges = state['merges']
