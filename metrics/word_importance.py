from collections import Counter
import torch




def calculate_word_importance(text, model, tokenizer):
    # Tokenize and get model output
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the token ids and attention mask
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]
    
    # Get the last hidden states
    last_hidden_states = outputs.last_hidden_state[0]
    
    # Perform max pooling across the hidden states
    pooled_output, indices = torch.max(last_hidden_states, dim=0)
    
    # Count the occurrences of each index
    index_counts = Counter(indices.tolist())
    
    # Calculate importance percentages
    total_counts = sum(index_counts.values())
    importance_percentages = [100.0 * count / total_counts for count in index_counts.values()]
    
    # Create a list of (token, importance) tuples
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_importance = list(zip(tokens, importance_percentages))
    
    # Remove special tokens (like [CLS] and [SEP]) and their importances
    word_importance = [(token, imp) for token, imp in word_importance if token not in ('[CLS]', '[SEP]', '[PAD]')]
    
    return word_importance


def get_word_importance_dict(text, model, tokenizer):
    word_importance = calculate_word_importance(text, model, tokenizer)
    return {word: importance for word, importance in word_importance}