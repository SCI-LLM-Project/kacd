#!/bin/bash

# Extract the first Python function
sed -n '/def rag/,/^\S/p' prompts.py | sed '$d' > default_prompt.tmp

# Extract the second Python function
sed -n '/def reduce/,/^\S/p' prompts.py | sed '$d' > context_prompt.tmp

# Generate the diff
diff -u --color=always default_prompt.tmp context_prompt.tmp

# Clean up
rm default_prompt.tmp context_prompt.tmp


