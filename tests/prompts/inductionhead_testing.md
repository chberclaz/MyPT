# Code pattern induction (model's been training on lots of code)

python generate.py --model_name unified_v1_llama_gold \
 --prompt "def add(a, b):
return a + b

def subtract(a, b):
return" \
 --temperature 0

# 3-word password, all lowercase (test if case was the issue)

python generate.py --model_name unified_v1_llama_gold \
 --prompt "The code is: red blue green. Please repeat the code: red blue" \
 --temperature 0

# 3-word password, mixcase (test if case was the issue)

python generate.py --model_name unified_v1_llama_gold \
 --prompt "The code is: red BLue grEEn. Please repeat the code: red BLue" \
 --temperature 0

# Number sequence completion

python generate.py --model_name unified_v1_llama_gold \
 --prompt "1 2 3 4 5. 1 2 3 4" \
 --temperature 0

# Dialogue-style retrieval (trained on threaded conversations)

python generate.py --model_name unified_v1_llama_gold \
 --prompt "User: My name is John.
Assistant: Your name is" \
 --temperature 0

python generate.py --model_name unified_v1_llama_gold \
 --prompt "A B C D. A B C" \
 --temperature 0

python generate.py --model_name unified_v1_llama_gold \
 --prompt "The password is: elephant Test hound. Please repeat the password: elephant" \
 --temperature 0

python generate.py --model_name unified_v1_llama_gold \
 --prompt "The Capital city of Germany is" \
 --temperature 0

python generate.py --model_name unified_v1_llama_gold \
--prompt "10 + 4 = 1" \
--temperature 0

python generate.py --model_name unified_v1_llama_gold \
 --prompt "User: Du bist MyPT.
Assistant: Ich bin" \
 --temperature 0

python generate.py --model_name unified_v1_llama_gold \
--prompt "write a java multiply function: " \
--temperature 0
