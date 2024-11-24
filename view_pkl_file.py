import joblib

train_matrix_path = r"classifier\baseline.pkl"
# output_file_path = r"matrix\vocabulary.txt"
    

train_matrix = joblib.load(train_matrix_path)
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     f.write(repr(train_matrix))

print(train_matrix)