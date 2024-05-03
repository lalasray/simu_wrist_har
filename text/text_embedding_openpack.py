import sklearn.cluster
from InstructorEmbedding import INSTRUCTOR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

model = INSTRUCTOR('hkunlp/instructor-large')

openpack_sentences = [
    ['Represent packing activity sentence for clustering','Picking'],
    ['Represent packing activity sentence for clustering','Relocate Item Label'],
    ['Represent packing activity sentence for clustering','Assemble Box'],
    ['Represent packing activity sentence for clustering','Insert Items'],
    ['Represent packing activity sentence for clustering','Close Box'],
    ['Represent packing activity sentence for clustering','Attach Box Label'],
    ['Represent packing activity sentence for clustering','Scan Label'],
    ['Represent packing activity sentence for clustering','Attach Shipping Label'],
    ['Represent packing activity sentence for clustering','Put on Back Table'],
    ['Represent packing activity sentence for clustering','Fill out Order']
    ]

embeddings = model.encode(openpack_sentences)
labels = np.arange(10)
tsne = TSNE(n_components=2, perplexity=5, metric="cosine", init="random")  # Adjust perplexity here
embeddings_tsne = tsne.fit_transform(embeddings)
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis')

# Annotate each point with its corresponding label
for i, label in enumerate(labels):
    plt.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], str(label), color='black', fontsize=8)

plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()