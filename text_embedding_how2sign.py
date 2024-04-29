import csv
from InstructorEmbedding import INSTRUCTOR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_sentences_from_csv(filename, prefix):
    sentences = []
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            sentences.append([f"{prefix}:"] + [row[6]])  # Assuming the "SENTENCE" column is at index 6
    return sentences

csv_filename = r"C:\Users\lalas\Downloads\how2sign_val.csv"
prefix = "Represent conversational speech senetence for clustering"
sentences = extract_sentences_from_csv(csv_filename, prefix)

print(len(sentences))

model = INSTRUCTOR('hkunlp/instructor-large')

embeddings = model.encode(sentences)

print(embeddings.shape)

kmeans = KMeans(n_clusters=5)
# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, perplexity=30, metric="cosine", init="random")  # Adjust perplexity here
embeddings_tsne = tsne.fit_transform(embeddings)
labels = kmeans.fit_predict(embeddings)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],c=labels, cmap='viridis')
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()
