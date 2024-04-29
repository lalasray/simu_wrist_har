import sklearn.cluster
from InstructorEmbedding import INSTRUCTOR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

model = INSTRUCTOR('hkunlp/instructor-large')

sentences = [['Represent human fitness activity for clustering: ','The body is lowered at the hips from a standing position and then stands back up to complete a repetition. Hands are push in front for balancing.'],
             ['Represent human fitness activity for clustering: ','Legs extended back and balancing the straight body on hands and toes. The arms are flexed to alower and raise the body. Repetitions are counted when the body returns to the starting position.'],
             ['Represent human fitness activity for clustering: ','From a sitting pisition, the weights are pressed upwards until the arms are straight and the weights touch above the head.'],
             ['Represent human fitness activity for clustering: ',"One leg is positioned forward with knee bent and foot flat on the ground while the other leg is positioned behind. The position of the legs is repeatedly swapped."],
             ['Represent human fitness activity for clustering: ','Slightly bent knees, hips pushed back, chest and head up. With elbows at a 60-degree angle, the dumbbells are raised up from the back muscles.'],
             ['Represent human fitness activity for clustering: ','Abdominal exercise done by lying on the back and lifting the torso with arms behind the head.'],
             ['Represent human fitness activity for clustering: ','The weight is brought overhead, extending the arms straight. Keeping the shoulders still, the elbows are slowly bent, lowering the weight behind the head to where the arms are just lower than 90 degrees, elbows pointing forward.'],
             ['Represent human fitness activity for clustering: ','Bicep curls with weights. Arms are alternated in lifting up the weight with the rest of the body remaining still.'],
             ['Represent human fitness activity for clustering: ','Sitting with a dumbbell in each hand and straight back. Slowly lifting the weights out to the side until the arms are parallel with the floor.'],
             ['Represent human fitness activity for clustering: ','Starting with the arms on the side and the legs brought together. By a jump into the air, simultaneously the legs are spread and the hands are pushed up to touch overhead. A repetition is completed with another jump returning to the starting position.'],
             ['Represent human fitness activity for clustering: ','Squats'],
             ['Represent human fitness activity for clustering: ','Push-ups'],
             ['Represent human fitness activity for clustering: ','Dumbbell shoulder presses'],
             ['Represent human fitness activity for clustering: ','Lunges'],
             ['Represent human fitness activity for clustering: ','Standing dumbbell rows'],
             ['Represent human fitness activity for clustering: ','Sit-ups'],
             ['Represent human fitness activity for clustering: ','Dumbbell tricep extensions'],
             ['Represent human fitness activity for clustering: ','Bicep curls'],
             ['Represent human fitness activity for clustering: ','Sitting dumbbell lateral raises'],
             ['Represent human fitness activity for clustering: ','Jumping jacks']
             ]
embeddings = model.encode(sentences)
#clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=2)
#clustering_model.fit(embeddings)
#cluster_assignment = clustering_model.labels_
#print(cluster_assignment)
print(embeddings.shape)

arr1 = np.arange(10)
arr2 = np.arange(10)
labels = np.concatenate((arr1, arr2))
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