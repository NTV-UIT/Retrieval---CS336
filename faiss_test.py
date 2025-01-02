import faiss

# Đường dẫn đến file FAISS
index_path = r"F:\HK1 2024-2025\Retrieval---CS336\vi_vectorstores\vi_db_faiss\index.faiss"

# Load FAISS index
index = faiss.read_index(index_path)

# Kiểm tra thông tin của index
print("Số vector trong index:", index.ntotal)
print("Kích thước vector (d):", index.d)

# Nếu muốn kiểm tra các vector, truy vấn một vector (giả sử vector 0)
if index.ntotal > 0:
    vector = index.reconstruct(0)
    print("Vector đầu tiên trong index:", vector)
