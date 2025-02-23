import sys
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple
import socket
import threading
from queue import Queue

class LuckAwareAttention:
    def __init__(self, socket_path="/tmp/luck.sock", max_adjustment=0.005):
        self.socket_path = socket_path
        self.max_adjustment = max_adjustment
        self.luck_values = Queue()
        self.running = True
        
        # Start luck collector thread
        self.collector_thread = threading.Thread(target=self._collect_luck_values)
        self.collector_thread.daemon = True
        self.collector_thread.start()
    
    def _collect_luck_values(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
        except:
            print("No luck socket found, running without luck values", file=sys.stderr)
            return
            
        while self.running:
            try:
                data = sock.recv(16)
                if data:
                    luck_value = float(data.decode().strip())
                    self.luck_values.put(luck_value)
            except:
                continue

    def modify_attention(self, attention_probs):
        if self.luck_values.empty():
            return attention_probs
            
        try:
            luck_value = self.luck_values.get_nowait()
            adjustment = (luck_value / 6000.0) * self.max_adjustment  # Using 6000 as per luck generator
            
            # Create subtle modifications
            noise = torch.normal(1.0, adjustment, 
                               attention_probs.shape,
                               device=attention_probs.device)
            modified = attention_probs * noise
            
            # Renormalize
            return modified / modified.sum(dim=-1, keepdim=True)
        except:
            return attention_probs

class ModifiedBertSelfAttention(torch.nn.Module):
    """Wrapper to modify BERT's self attention with luck values"""
    def __init__(self, original_self_attention, luck_attention):
        super().__init__()
        self.original = original_self_attention
        self.luck = luck_attention
        
    def forward(self, *args, **kwargs):
        # Get original attention outputs
        outputs = self.original(*args, **kwargs)
        
        # Modify attention probs if they're in the outputs
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attention_probs = outputs[1]
            modified_probs = self.luck.modify_attention(attention_probs)
            return (outputs[0], modified_probs) + outputs[2:]
        
        return outputs

class ManPageCommandHelper:
    def __init__(self):
        # Initialize tiny BERT
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.luck = LuckAwareAttention()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Patch the model's attention mechanism
        self._patch_attention()
    
    def _patch_attention(self):
        """Patch BERT's attention layers with luck-aware attention"""
        for layer in self.model.encoder.layer:
            # Wrap the self attention with our modifier
            original_self = layer.attention.self
            layer.attention.self = ModifiedBertSelfAttention(original_self, self.luck)
    
    def _extract_key_sections(self, man_content: str) -> List[str]:
        """Extract relevant sections from man page"""
        sections = []
        current_section = []
        
        for line in man_content.split('\n'):
            # More flexible indentation detection
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent > 2:  # Any indented line might be important
                current_section.append(stripped)
            else:
                if current_section and any(word in ' '.join(current_section).lower() for word in ['column', 'list', 'display', 'output', 'format']):
                    sections.append(' '.join(current_section))
                current_section = []
                
            # Debug first few lines we see
            if len(sections) == 0 and len(current_section) == 1:
                print(f"Processing line: '{line}'", file=sys.stderr)
        
        return sections
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text with BERT tokenizer"""
        inputs = self.tokenizer(text, 
                              truncation=True, 
                              max_length=512, 
                              return_tensors="pt")
        return inputs.to(self.device)
    
    def generate_command(self, man_content: str, prompt: str) -> str:
        """Generate command based on man page content and prompt"""
        # Extract key sections
        sections = self._extract_key_sections(man_content)
        print(f"Found {len(sections)} sections", file=sys.stderr)
        
        if not sections:
            print("Example sections:", file=sys.stderr)
            print(man_content[:500], file=sys.stderr)
            return "no sections"  # Default if no sections found

        # Encode prompt
        prompt_encoded = self._encode_text(prompt)
        with torch.no_grad():
            prompt_embedding = self.model(**prompt_encoded).last_hidden_state.mean(dim=1)
        
        # Process each section
        best_score = -float('inf')
        best_command = ""
        
        for section in sections:
            # Encode section
            section_encoded = self._encode_text(section)
            with torch.no_grad():
                section_embedding = self.model(**section_encoded).last_hidden_state.mean(dim=1)
            
            # Calculate similarity
            similarity = torch.cosine_similarity(prompt_embedding, section_embedding)
            
            if similarity > best_score:
                best_score = similarity
                # Extract command from section (first word or first line)
                command = section.split('\n')[0].split()[0]
                best_command = command
        
        return best_command

def main():
    # Read man page content from stdin
    man_content = sys.stdin.read()
    
    # Get prompt from command line argument
    if len(sys.argv) != 2:
        print("Usage: man -P cat command | python script.py 'prompt'", file=sys.stderr)
        sys.exit(1)
    prompt = sys.argv[1]
    
    # Initialize and run helper
    helper = ManPageCommandHelper()
    command = helper.generate_command(man_content, prompt)
    
    # Output just the command
    print(command)

if __name__ == "__main__":
    main()

