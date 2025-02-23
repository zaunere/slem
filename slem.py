import sys
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple
import socket
import threading
from queue import Queue
import re

class CommandNode:
    def __init__(self, command: str, description: str = "", parent=None):
        self.command = command
        self.description = description
        self.parent = parent
        self.children = []
        self.flags = {}  # flag -> description
        self.examples = []

class ManPageParser:
    def __init__(self):
        self.root = None
        self.current_section = None
        self.current_command = None

    def parse(self, content: str) -> CommandNode:
        lines = content.split('\n')
        self.root = CommandNode("root")
        self.current_section = None
        
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(stripped)
            
            # Skip empty lines
            if not stripped:
                continue

            # Look for command name in first few non-empty lines
            if not self.root.command or self.root.command == "root":
                cmd_match = re.match(r'^([\w-]+)\s*[-\(].*', stripped)
                if cmd_match:
                    cmd = cmd_match.group(1)
                    if cmd not in ['NAME', 'SYNOPSIS']:
                        self.root = CommandNode(cmd)
                        self.current_command = self.root
                        print(f"Found command: {cmd}", file=sys.stderr)
                        continue

            # Command examples section
            if indent == 0 and 'example' in stripped.lower():
                self.current_section = 'examples'
                continue

            # Options/flags section
            if indent == 0 and any(x in stripped.lower() for x in ['options', 'flags', 'arguments']):
                self.current_section = 'options'
                continue

            # Process based on current section
            if indent > 0:  # Indented content belongs to current section
                if self.current_section == 'main':
                    # Look for command patterns
                    cmd_match = re.match(r'\s*(\w+)\s*-\s*(.+)', stripped)
                    if cmd_match:
                        cmd, desc = cmd_match.groups()
                        self.root = CommandNode(cmd, desc)
                        self.current_command = self.root
                    
                elif self.current_section == 'options':
                    # Look for flag patterns
                    flag_match = re.match(r'\s*(-[-\w]+)\s*(.+)', stripped)
                    if flag_match and self.current_command:
                        flag, desc = flag_match.groups()
                        self.current_command.flags[flag] = desc

                elif self.current_section == 'examples':
                    # Look for command examples
                    if self.current_command and any(x in stripped for x in [self.current_command.command, '-']):
                        self.current_command.examples.append(stripped)

        return self.root

class LuckAwareAttention:
    def __init__(self, socket_path="/tmp/luck.sock", max_adjustment=0.005):
        self.socket_path = socket_path
        self.max_adjustment = max_adjustment
        self.luck_values = Queue()
        self.running = True
        self.socket_available = False
        
        try:
            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            test_sock.connect(self.socket_path)
            test_sock.close()
            self.socket_available = True
            
            self.collector_thread = threading.Thread(target=self._collect_luck_values)
            self.collector_thread.daemon = True
            self.collector_thread.start()
        except:
            print("Luck socket not available - running without luck enhancement", file=sys.stderr)

    def _collect_luck_values(self):
        if not self.socket_available:
            return
            
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        
        while self.running:
            try:
                data = sock.recv(16)
                if data:
                    luck_value = float(data.decode().strip())
                    self.luck_values.put(luck_value)
            except:
                continue

class ManPageCommandHelper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.luck = LuckAwareAttention()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.parser = ManPageParser()

    def _encode_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, 
                              truncation=True, 
                              max_length=512, 
                              return_tensors="pt")
        return inputs.to(self.device)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        encoded1 = self._encode_text(text1)
        encoded2 = self._encode_text(text2)
        
        with torch.no_grad():
            embed1 = self.model(**encoded1).last_hidden_state.mean(dim=1)
            embed2 = self.model(**encoded2).last_hidden_state.mean(dim=1)
            sim = torch.cosine_similarity(embed1, embed2)
            return float(sim)

    def _find_best_matches(self, prompt: str, node: CommandNode) -> List[Tuple[float, str, str]]:
        matches = []
        
        # Check flags descriptions
        for flag, desc in node.flags.items():
            sim = self._calculate_similarity(prompt, desc)
            if sim > 0.3:  # Threshold can be adjusted
                matches.append((sim, flag, desc))
        
        # Check examples
        for example in node.examples:
            sim = self._calculate_similarity(prompt, example)
            if sim > 0.3:
                matches.append((sim, example, "example"))
        
        # Sort by similarity score
        matches.sort(reverse=True)
        return matches

    def generate_command(self, man_content: str, prompt: str) -> str:
        # Parse man page into structured format
        command_tree = self.parser.parse(man_content)
        print(f"Parsed man page for: {command_tree.command}", file=sys.stderr)
        
        # Find best matching sections
        matches = self._find_best_matches(prompt, command_tree)
        
        if not matches:
            print("No strong matches found", file=sys.stderr)
            return command_tree.command  # Return basic command if no good matches
            
        # Get top matches
        print("\nTop matches:", file=sys.stderr)
        for score, match, desc in matches[:3]:
            print(f"Score {score:.3f}: {match} - {desc}", file=sys.stderr)
        
        # Build command based on best matches
        best_match = matches[0]
        if best_match[2] == "example":
            # Return the example command directly
            return best_match[1].split('#')[0].strip()  # Remove any comments
        else:
            # Construct command from matched flags
            cmd_parts = [command_tree.command]
            cmd_parts.extend(m[1] for m in matches[:2] if m[0] > 0.4)  # Use top 2 flags if they're good matches
            return ' '.join(cmd_parts)

def main():
    if len(sys.argv) != 2:
        print("Usage: man command | col -b | python script.py 'prompt'", file=sys.stderr)
        sys.exit(1)

    # Read all content, strip any pager escapes
    man_content = sys.stdin.read()
    print(f"\nFirst few lines received:", file=sys.stderr)
    for line in man_content.split('\n')[:5]:
        print(f"  {line}", file=sys.stderr)
    prompt = sys.argv[1]
    
    helper = ManPageCommandHelper()
    command = helper.generate_command(man_content, prompt)
    print(command)

if __name__ == "__main__":
    main()

