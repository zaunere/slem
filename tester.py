#!/usr/bin/env python3

import sys
from text_analyzer import TextAnalyzer

def main():
    # Get query from command line
    if len(sys.argv) != 2:
        print("Usage: cat content.txt | python3 test_analyzer.py 'your query'", file=sys.stderr)
        sys.exit(1)
    query = sys.argv[1]
    
    # Read content from stdin
    content = sys.stdin.read()
    
    # Create analyzer with debug on
    analyzer = TextAnalyzer(debug=True)
    
    # Run analysis
    results = analyzer.analyze(content, query)
    
    # Show key findings
    print("\nKey Findings:")
    
    # Show WH structure matches
    structure = results['structure']
    if structure.what['action']:
        print(f"\nAction identified: {structure.what['action']}")
        print(f"Target identified: {structure.what['target']}")
    
    if structure.how['format']:
        print(f"\nFormat requested: {structure.how['format']}")
    
    # Show refined options
    refinements = results['refinements']
    if 'format_options' in refinements:
        print("\nPossible format matches:")
        for opt in refinements['format_options'][:3]:  # Show top 3
            print(f"  Score {opt['score']}: {opt['description']}")
            print(f"    Matched terms: {', '.join(opt['matches'])}")
    
    if 'action_methods' in refinements:
        print("\nPossible methods:")
        for method in refinements['action_methods'][:3]:  # Show top 3
            print(f"  {method['description']}")

if __name__ == "__main__":
    main()