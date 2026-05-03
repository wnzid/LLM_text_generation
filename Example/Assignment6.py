import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

#  Load GPT-2 model
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")


input_ids = tokenizer("Rome is the capital", return_tensors="pt").input_ids
print(input_ids)
# Output: tensor([[  49,  462,  318,  262, 3139]])

# Decode the tokens back into text
for t in input_ids[0]:
    print(t, "\t:", tokenizer.decode(t))

"""    
Output:
tensor(49) 	    : R
tensor(462) 	: ome
tensor(318) 	:  is
tensor(262) 	:  the
tensor(3139) 	:  capital
"""

# Feeding Input to the Model
outputs = gpt2(input_ids)
print(outputs.logits.shape)
# Output: torch.Size([1, 5, 50257])

#Finding the Most Likely Next Token
final_logits = gpt2(input_ids).logits[0, -1]  # The last set of logits
most_likely_token_id = final_logits.argmax()
print(most_likely_token_id)
# Output: tensor(286)
most_likely_token = tokenizer.decode(most_likely_token_id)
print(most_likely_token)
# Output: of

# Exploring Other Token Candidates
top10_logits = torch.topk(final_logits, 10)
for index in top10_logits.indices:
    print(tokenizer.decode(index))
"""
# Output:
 of
 city
 and
,
.
 in
 for
 that
 capital
 on
"""

# Converting Logits to Probabilities
top10 = torch.topk(final_logits.softmax(dim=0), 10)
for value, index in zip(top10.values, top10.indices):
    print(f"{tokenizer.decode(index):<10} {value.item():.1%}")

"""
Output:
 of        71.8%
 city      14.4%
 and       3.4%
,          2.8%
.          1.0%
 in        0.8%
 for       0.8%
 that      0.3%
 capital   0.2%
 on        0.2%
"""

#Generating text using Greedy Decoding
output_ids = gpt2.generate(input_ids, max_new_tokens=20)
decoded_text = tokenizer.decode(output_ids[0])
print("Output IDs:", output_ids)
print(f"Generated text: {decoded_text}")

"""
Output:
Output IDs: tensor([[  49,  462,  318,  262, 3139,  286,  262, 7993, 8065,   11,  290,  262,
         3139,  286,  262, 7993, 8065,   13,  632,  318,  262, 3139,  286,  262,
         7993]])
Generated text: Rome is the capital of the Roman Empire, and the capital of the Roman Empire. It is the capital of the Roman
"""

#Generating text using Beam Search
beam_output = gpt2.generate(
    input_ids,
    num_beams=5,
    max_new_tokens=20,
)
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
"""
Output:
Rome is the capital of the Roman Empire. 
It is the capital of the Roman Empire. 
It is the capital of the
"""

# Text generation using Repetition Penalty
beam_output = gpt2.generate(
    input_ids,
    num_beams=5,
    repetition_penalty=1.2,
    max_new_tokens=20,
)
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

"""
Output:
Rome is the capital city of the Roman Empire. 
It is home to the largest Roman population in the world, and is
"""

# Text generation using Temperature
sampling_output = gpt2.generate(
    input_ids,
    do_sample=True,
    temperature=0.2,
    max_length=20,
    repetition_penalty=1.2,
    top_k=0,
)
print(tokenizer.decode(sampling_output[0], skip_special_tokens=True))

# Text generation Top-K Sampling
sampling_output = gpt2.generate(
    input_ids,
    do_sample=True,
    max_length=40,
    top_p=0.94,
    top_k=10,
    repetition_penalty=1.2,    
)
print(tokenizer.decode(sampling_output[0], skip_special_tokens=True))

# Top-P (Nucleus) Sampling
sampling_output = gpt2.generate(
    input_ids,
    do_sample=True,
    max_length=40,
    top_p=0.94,
    top_k=0,
)
print(tokenizer.decode(sampling_output[0], skip_special_tokens=True))


