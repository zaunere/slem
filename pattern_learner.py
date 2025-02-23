import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import string

class SimplePatternLearner:
    """Learns patterns without external dependencies"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.common_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        # Common command-line action verbs
        self.action_verbs = {
            'show', 'display', 'list', 'print', 'format', 'sort', 'order',
            'find', 'search', 'filter', 'group', 'count', 'create', 'remove'
        }
    
    def analyze_text(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        # Clean text
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        words = text.split()
        
        # Remove common words and single characters
        return [w for w in words if w not in self.common_words and len(w) > 1]
    
    def find_action_patterns(self, text: str) -> List[Tuple[str, str]]:
        """Find verb-object patterns"""
        patterns = []
        words = text.lower().split()
        
        for i, word in enumerate(words):
            if word in self.action_verbs:
                # Look for object after verb
                for j in range(i + 1, min(i + 4, len(words))):
                    if words[j] not in self.common_words:
                        patterns.append((word, words[j]))
                        break
        return patterns
    
    def find_parameter_patterns(self, text: str) -> Dict[str, Set[str]]:
        """Find parameter patterns like 'by size' or 'with time'"""
        params = defaultdict(set)
        
        # Look for common parameter patterns
        for match in re.finditer(r'(by|with|in|as)\s+(\w+)', text.lower()):
            prep, param = match.groups()
            if param not in self.common_words:
                params[prep].add(param)
        
        return dict(params)
    
    def find_flag_patterns(self, text: str) -> Dict[str, List[str]]:
        """Find patterns in flag descriptions"""
        patterns = defaultdict(list)
        
        # Look for flag descriptions
        flag_matches = re.finditer(r'-(\w+)\s+([^-\n]+)', text)
        for match in flag_matches:
            flag = match.group(1)
            desc = match.group(2).lower()
            
            # Find action words
            actions = [word for word in desc.split() 
                      if word in self.action_verbs]
            
            # Find parameters
            params = re.findall(r'(?:by|with|in|as)\s+(\w+)', desc)
            
            if actions or params:
                patterns[flag] = {
                    'actions': actions,
                    'parameters': params,
                    'description': desc
                }
        
        return dict(patterns)
    
    def learn_patterns(self, content: str, intent: str) -> Dict[str, object]:
        """Learn patterns from content, guided by intent"""
        if self.debug:
            print("\n=== Pattern Learning Start ===", file=sys.stderr)
            print(f"Processing intent: {intent}", file=sys.stderr)
        
        # Analyze intent
        intent_terms = self.analyze_text(intent)
        intent_actions = self.find_action_patterns(intent)
        intent_params = self.find_parameter_patterns(intent)
        
        if self.debug:
            print("\nIntent analysis:", file=sys.stderr)
            print(f"  Terms: {intent_terms}", file=sys.stderr)
            print(f"  Actions: {intent_actions}", file=sys.stderr)
            print(f"  Parameters: {intent_params}", file=sys.stderr)
        
        # Analyze content
        content_patterns = self.find_flag_patterns(content)
        
        if self.debug:
            print("\nContent patterns:", file=sys.stderr)
            for flag, data in content_patterns.items():
                print(f"\nFlag: -{flag}", file=sys.stderr)
                print(f"  Actions: {data['actions']}", file=sys.stderr)
                print(f"  Parameters: {data['parameters']}", file=sys.stderr)
                print(f"  Description: {data['description']}", file=sys.stderr)
        
        # Find relationships between intent and content
        relationships = self._find_relationships(intent_terms, intent_actions,
                                              intent_params, content_patterns)
        
        if self.debug:
            print("\nRelationships found:", file=sys.stderr)
            for rel_type, items in relationships.items():
                print(f"  {rel_type}: {items}", file=sys.stderr)
            print("\n=== Pattern Learning End ===", file=sys.stderr)
        
        return {
            'intent': {
                'terms': intent_terms,
                'actions': intent_actions,
                'parameters': intent_params
            },
            'content': content_patterns,
            'relationships': relationships
        }
    
    def _find_relationships(self, intent_terms, intent_actions, 
                          intent_params, content_patterns):
        """Find relationships between intent and content patterns"""
        relationships = defaultdict(list)
        
        # Look for phrase matches in intent
        intent_str = ' '.join(intent_terms)
        
        for flag, data in content_patterns.items():
            score = 0
            matches = []
            desc = data['description']
            
            # Check for exact phrase matches - highest priority
            if any(phrase in desc for phrase in [
                'one file per line',
                'one per line',
                'list by lines',
                'list by columns'
            ]):
                score += 5
                matches.append('exact_phrase')
            
            # Check for consecutive term matches
            terms_found = []
            for term in intent_terms:
                if term in desc:
                    terms_found.append(term)
            
            if len(terms_found) > 1:
                # Check if terms appear in same order
                desc_words = desc.split()
                term_positions = [desc_words.index(t) for t in terms_found if t in desc_words]
                if term_positions == sorted(term_positions):
                    score += len(terms_found) * 2
                    matches.append(f"consecutive_terms:{','.join(terms_found)}")
            
            # Check for display/format flags when intent is about display
            if any(w in intent_str for w in ['show', 'display', 'list', 'format']):
                if any(w in desc for w in ['format', 'display', 'list by', 'output']):
                    score += 2
                    matches.append('display_flag')
            
            # Action matches are good but not primary
            for intent_action, intent_obj in intent_actions:
                if intent_action in data['actions']:
                    score += 1
                    matches.append(f"action:{intent_action}")
                
                # Object matches are important
                if intent_obj in desc:
                    score += 2
                    matches.append(f"object:{intent_obj}")
            
            # Individual term matches are lowest priority
            for term in intent_terms:
                if term in desc and 'term:'+term not in matches:
                    score += 0.5
                    matches.append(f"term:{term}")
            
            if score > 0:
                relationships['matches'].append({
                    'flag': flag,
                    'score': score,
                    'matches': matches
                })
        
        # Sort by score
        relationships['matches'].sort(key=lambda x: x['score'], reverse=True)
        return dict(relationships)
