#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from models import train_neural_network, evaluate_model
from utils import DataProcessor, CustomDataset

#%%
file_list = [
    'power-gb-train.tsv',
    'power-ua-train.tsv'
]

processor = DataProcessor()

raw_data = processor.load_data(
    folder_path="data/power/",
    file_list=file_list,
    text_head='text_en'
)

train_dev_raw, test_raw = processor.split_data(raw_data, test_size=0.2)
train_raw, dev_raw = processor.split_data(train_dev_raw, test_size=0.2)

#%%
print("Prepare data encoder...")
train_texts = [tup[2] for tup in train_raw]
train_encoder = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(1,3))
train_encoder.fit(train_texts)
print("Prepare data...")
train_dataset = CustomDataset(train_raw, train_encoder)
dev_dataset = CustomDataset(dev_raw, train_encoder)
test_dataset = CustomDataset(test_raw, train_encoder)


# %%

print("Train model...")
model = train_neural_network(
    train_data=train_dataset,
    dev_data=dev_dataset,
    num_classes=2,
    hidden_size=64,
    num_epochs=20,
    early_stop_patience=5,
)


# %%
precision, recall, f1 = evaluate_model(model, test_dataset)

print(precision, recall, f1)


# %%
