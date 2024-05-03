import csv
from InstructorEmbedding import INSTRUCTOR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def extract_sentences_from_csv(filename, prefix):
    sentences = []
    filenames = []
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            filenames.append(row[3])
            sentences.append([f"{prefix}:"] + [row[6]])  
    return sentences, filenames

csv_filename = "/home/lala/other/Repos/git/simu_wrist_har/data/how2sign/train/ann.csv"
prefix = "Represent conversational speech senetence for clustering"
sentences, filenames = extract_sentences_from_csv(csv_filename, prefix)
#print(sentences)
#print(filenames)
#print(len(sentences),len(filenames))

model = INSTRUCTOR('hkunlp/instructor-large')

embeddings = model.encode(sentences)
#print(embeddings.shape[0])
numpy_array = np.array(filenames)
#print(numpy_array.shape[0])

#kmeans = KMeans(n_clusters=5)
# Reduce dimensionality using t-SNE
#tsne = TSNE(n_components=2, perplexity=30, metric="cosine", init="random")  # Adjust perplexity here
#embeddings_tsne = tsne.fit_transform(embeddings)
#labels = kmeans.fit_predict(embeddings)

# Plot the clusters
#plt.figure(figsize=(8, 6))
#plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],c=labels, cmap='viridis')
#plt.title('t-SNE Visualization of Clusters')
#plt.xlabel('t-SNE Dimension 1')
#plt.ylabel('t-SNE Dimension 2')
#plt.colorbar(label='Cluster')
#plt.show()
new_directory = csv_filename.replace('ann.csv', 'text_processed/')
for i in range (1,numpy_array.shape[0]):
    np.save(new_directory+numpy_array[i]+".npy",embeddings[i])