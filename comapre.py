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
prefix = "Represent conversational speech for clustering"
sentences = extract_sentences_from_csv(csv_filename, prefix)

print(len(sentences))

model = INSTRUCTOR('hkunlp/instructor-large')

embeddings_asl = model.encode(sentences)


sentences_mmfit = [['Represent human activity sentence for clustering: ','The body is lowered at the hips from a standing position and then stands back up to complete a repetition. Hands are push in front for balancing.'],
             ['Represent human activity sentence for clustering: ','Legs extended back and balancing the straight body on hands and toes. The arms are flexed to alower and raise the body. Repetitions are counted when the body returns to the starting position.'],
             ['Represent human activity sentence for clustering: ','From a sitting pisition, the weights are pressed upwards until the arms are straight and the weights touch above the head.'],
             ['Represent human activity sentence for clustering: ',"One leg is positioned forward with knee bent and foot flat on the ground while the other leg is positioned behind. The position of the legs is repeatedly swapped."],
             ['Represent human activity sentence for clustering: ','Slightly bent knees, hips pushed back, chest and head up. With elbows at a 60-degree angle, the dumbbells are raised up from the back muscles.'],
             ['Represent human activity sentence for clustering: ','Abdominal exercise done by lying on the back and lifting the torso with arms behind the head.'],
             ['Represent human activity sentence for clustering: ','The weight is brought overhead, extending the arms straight. Keeping the shoulders still, the elbows are slowly bent, lowering the weight behind the head to where the arms are just lower than 90 degrees, elbows pointing forward.'],
             ['Represent human activity sentence for clustering: ','Bicep curls with weights. Arms are alternated in lifting up the weight with the rest of the body remaining still.'],
             ['Represent human activity sentence for clustering: ','Sitting with a dumbbell in each hand and straight back. Slowly lifting the weights out to the side until the arms are parallel with the floor.'],
             ['Represent human activity sentence for clustering: ','Starting with the arms on the side and the legs brought together. By a jump into the air, simultaneously the legs are spread and the hands are pushed up to touch overhead. A repetition is completed with another jump returning to the starting position.'],
             ['Represent human activity sentence for clustering: ','Squats'],
             ['Represent human activity sentence for clustering: ','Push-ups'],
             ['Represent human activity sentence for clustering: ','Dumbbell shoulder presses'],
             ['Represent human activity sentence for clustering: ','Lunges'],
             ['Represent human activity sentence for clustering: ','Standing dumbbell rows'],
             ['Represent human activity sentence for clustering: ','Sit-ups'],
             ['Represent human activity sentence for clustering: ','Dumbbell tricep extensions'],
             ['Represent human activity sentence for clustering: ','Bicep curls'],
             ['Represent human activity sentence for clustering: ','Sitting dumbbell lateral raises'],
             ['Represent human activity sentence for clustering: ','Jumping jacks']
             ]
embeddings_sentence = model.encode(sentences_mmfit)

tsne = TSNE(n_components=2, perplexity=5, metric="cosine", init="random")  # Adjust perplexity here
embeddings_tsne_sentence = tsne.fit_transform(embeddings_sentence)
embeddings_tsne_asl = tsne.fit_transform(embeddings_asl)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne_sentence[:, 0], embeddings_tsne_sentence[:, 1])
plt.scatter(embeddings_tsne_asl[:, 0], embeddings_tsne_asl[:, 1])
plt.title('t-SNE Visualization of Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()
