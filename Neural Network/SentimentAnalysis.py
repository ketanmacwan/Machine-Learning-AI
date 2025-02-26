# Import necessary libraries
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import pipeline
from datasets import load_dataset
import tensorflow as tf

# Step 1: Load the IMDb dataset
dataset = load_dataset("imdb")

# Step 2: Load pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Prepare TensorFlow datasets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Use 1000 examples for faster training
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))    # Use 1000 examples for testing

tf_train_dataset = train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    batch_size=8,
    shuffle=True
)

tf_test_dataset = test_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    batch_size=8,
    shuffle=False
)

# Step 5: Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=model.compute_loss,
              metrics=["accuracy"])

# Step 6: Train the model
model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=3)

model = TFBertForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_model")

# Use the model for inference
sentences = ["I loved this movie!", "This was terrible."]
inputs = tokenizer(sentences, return_tensors="tf", padding=True, truncation=True)
outputs = model(inputs)
predictions = tf.argmax(outputs.logits, axis=-1)
print(predictions.numpy())  # Predicted labels