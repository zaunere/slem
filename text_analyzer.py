import sys
import re
from typing import Dict, List, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class WHStructure:
    """Represents core WH question structure"""
    what: Dict[str, Any] = field(default_factory=lambda: {'action': None, 'target': None})
    how: Dict[str, Any] = field(default_factory=lambda: {'format': None, 'order': None, 'method': None})
    where: Dict[str, Any] = field(default_factory=lambda: {'source': None, 'destination': None})
    when: Dict[str, Any] = field(default_factory=lambda: {'timing': None, 'sequence': None})
    who: Dict[str, Any] = field(default_factory=lambda: {'owner': None, 'access': None})

class TextAnalyzer:
    """Analyzes text using WH questions followed by Yes/No refinement"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.structure = WHStructure()
        
        # Common patterns
        self.action_words = {
            'show', 'list', 'find', 'sort', 'display', 'output', 'print',
            'format', 'group', 'filter', 'search'
        }
        self.format_indicators = {
            'in', 'by', 'per', 'using', 'with', 'as', 'like'
        }
        self.sequence_words = {
            'one', 'each', 'every', 'all', 'per', 'by'
        }
    
    def analyze(self, content: str, query: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        if self.debug:
            print("\n=== Analysis Start ===", file=sys.stderr)
            print(f"Query: {query}", file=sys.stderr)
        
        # Build query understanding
        self._analyze_wh(query)
        
        if self.debug:
            print("\nWH Analysis:", file=sys.stderr)
            self._debug_structure()
        
        # Find matching patterns in content
        refinements = self._refine_structure(content, query)
        
        if self.debug:
            print("\nRefinements:", file=sys.stderr)
            for key, value in refinements.items():
                print(f"  {key}: {value}", file=sys.stderr)
            print("\n=== Analysis End ===", file=sys.stderr)
        
        return {
            'structure': self.structure,
            'refinements': refinements
        }
    
    def _analyze_wh(self, query: str) -> None:
        """Analyze query using WH patterns"""
        words = query.lower().split()
        
        # WHAT: Find primary action and target
        for i, word in enumerate(words):
            if word in self.action_words:
                self.structure.what['action'] = word
                # Look ahead for target
                if i + 1 < len(words):
                    self.structure.what['target'] = words[i + 1]
                break
        
        # HOW: Look for format/method patterns
        format_parts = []
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for sequence patterns like "one per line"
            if word in self.sequence_words and i + 2 < len(words):
                if words[i + 1] in self.format_indicators:
                    format_parts.extend(words[i:i+3])
                    i += 3
                    continue
            
            # Check for format indicators
            if word in self.format_indicators and i + 1 < len(words):
                format_parts.extend(words[i:i+2])
                i += 2
                continue
                
            i += 1
        
        if format_parts:
            self.structure.how['format'] = ' '.join(format_parts)
    
    def _refine_structure(self, content: str, query: str) -> Dict[str, Any]:
        """Refine WH structure with pattern matching"""
        refinements = {}
        
        # Find action methods if we have an action
        if self.structure.what['action']:
            methods = self._find_action_methods(content, query)
            if methods:
                refinements['action_methods'] = methods
        
        # Find format options if we have a format
        if self.structure.how['format']:
            formats = self._find_format_options(content, query)
            if formats:
                refinements['format_options'] = formats
        
        return refinements
    
    def _find_action_methods(self, content: str, query: str) -> List[Dict[str, Any]]:
        """Find methods matching our action"""
        methods = []
        action = self.structure.what['action']
        target = self.structure.what['target']
        
        if not action:
            return methods
        
        # Look for relevant sections
        relevant_lines = []
        for line in content.split('\n'):
            line = line.lower()
            # Must contain action and ideally target
            if action in line and (not target or target in line):
                relevant_lines.append(line)
        
        # Score each line
        for line in relevant_lines:
            score = 0
            # Base score for having the action
            score += 1
            
            # Bonus for having the target
            if target and target in line:
                score += 1
            
            # Bonus for similar words to query
            query_words = set(query.lower().split())
            line_words = set(line.split())
            common_words = query_words & line_words
            score += len(common_words) * 0.5
            
            if score >= 2:  # Only keep good matches
                methods.append({
                    'description': line.strip(),
                    'score': score,
                    'matches': list(common_words)
                })
        
        # Sort by score
        methods.sort(key=lambda x: x['score'], reverse=True)
        return methods
    
    def _find_format_options(self, content: str, query: str) -> List[Dict[str, Any]]:
        """Find format options matching our requirements"""
        options = []
        format_desc = self.structure.how['format']
        
        if not format_desc:
            return options
        
        format_words = set(format_desc.lower().split())
        query_words = set(query.lower().split())
        
        # Look for format-related content
        for line in content.split('\n'):
            line = line.lower()
            score = 0
            matches = []
            
            # Look for key format words
            for word in format_words:
                if word in line:
                    score += 1
                    matches.append(word)
            
            # Bonus for phrase matches
            if format_desc in line:
                score += 3
                matches.append(format_desc)
            
            # Bonus for query word matches
            common_words = query_words & set(line.split())
            score += len(common_words) * 0.5
            
            if score > 0:
                options.append({
                    'description': line.strip(),
                    'score': score,
                    'matches': matches
                })
        
        # Sort by score
        options.sort(key=lambda x: x['score'], reverse=True)
        return options
    
    def _debug_structure(self) -> None:
        """Show debug info about current structure"""
        print("  WHAT:", file=sys.stderr)
        print(f"    Action: {self.structure.what['action']}", file=sys.stderr)
        print(f"    Target: {self.structure.what['target']}", file=sys.stderr)
        print("  HOW:", file=sys.stderr)
        print(f"    Format: {self.structure.how['format']}", file=sys.stderr)
        print(f"    Order: {self.structure.how['order']}", file=sys.stderr)
        print(f"    Method: {self.structure.how['method']}", file=sys.stderr)
