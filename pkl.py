from transformers import AutoTokenizer

# Load tokenizer cho MiniLM
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Kiểm tra max token
max_length = tokenizer.model_max_length
print(f"Mô hình hỗ trợ tối đa: {max_length} tokens")
