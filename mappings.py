# Read the mappings file and create a dictionary of the mappings
# Read mapping file

# %%
mappings_file = 'mappings.txt'
mappings_dict = {}
with open(mappings_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line != '':
            line = line.split(': ')
            mappings_dict[line[0]] = line[1]

# %%
# Read the csv file of the embeddings
import pandas as pd

df = pd.read_csv('glove_embed.csv', header=None)

class_to_index = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
           'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
           'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
           'bottle', 'bowl', 'can', 'cup', 'plate',
           'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
           'clock', 'keyboard', 'lamp', 'telephone', 'television',
           'bed', 'chair', 'couch', 'table', 'wardrobe',
           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
           'bear', 'leopard', 'lion', 'tiger', 'wolf',
           'bridge', 'castle', 'house', 'road', 'skyscraper',
           'cloud', 'forest', 'mountain', 'plain', 'sea',
           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
           'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
           'cra', 'lobster', 'snail', 'spider', 'worm',
           'baby', 'boy', 'girl', 'man', 'woman',
           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
           'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
           'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
           'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

# %%
import numpy as np

embeddings = []

# Iterate through the mappings dictionary
for key, value in mappings_dict.items():
    # Check if the value string matches with the initial part class_to_index
    if value in class_to_index:
        # Append row at class_to_index index to the embeddings array
        # embeddings = embeddings.append(np.array(df.iloc[class_to_index.index(value)]))
        embeddings.append(np.array(df.iloc[class_to_index.index(value)]))


# %%

embeddings = np.array(embeddings)

pd.DataFrame(embeddings).to_csv('embeddings_sorted.csv', index=False, header =  None)

# %%
