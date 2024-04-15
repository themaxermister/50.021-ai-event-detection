import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, create_optimizer

OUTPUT_DIR = 'output'
DATASET_DIR = "data/full_maven_with_category.csv" # File location of the model training data

VAL_RATIO = 0.3
DATA_COLUMNS = ['title', 'word_count', 'character_count', 'bigrams', 'lemma', 'pos', 'tag', 'dep', 'label', 'context_score', 'trigger_words']
LABEL_COLUMN = 'category'

def load_dataset(file_location, target_column):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_location)
    labels = sorted(list(df[target_column].unique()))
    print(len(labels))

    #convert labels in target column to numbers
    df[target_column] = df[target_column].apply(lambda x: labels.index(x))

    return df, labels

class BERTModel:
    def __init__(self, model_checkpoint, max_length, tokenizer, label_list, epochs=3, batch_size=8):
        self.model_checkpoint = model_checkpoint
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
    
        
    def preprocess_function(self, input_data):
        titles = input_data['title'].tolist()

        # Tokenize each title separately
        tokenized_inputs = self.tokenizer(titles, padding=True, truncation=True, max_length=self.max_length)

        # Convert lists to TensorFlow tensors
        input_ids = tf.constant(tokenized_inputs['input_ids'])
        attention_mask = tf.constant(tokenized_inputs['attention_mask'])
        labels = tf.constant(input_data[LABEL_COLUMN].tolist())

        # Create a TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices(({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }, labels))

        return dataset

    def train(self, encoded_train):
        self.model.resize_token_embeddings(len(tokenizer))
        
        batches_per_epoch = len(encoded_train)
        total_train_steps = int(batches_per_epoch * self.EPOCHS)
        
        optimizer, _ = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
        self.model.compile(optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(encoded_train.batch(self.BATCH_SIZE), epochs=self.EPOCHS)
    
    def evaluate(self, encoded_test):
        bert_loss, bert_acc = self.model.evaluate(encoded_test)
        return bert_loss, bert_acc
        

if __name__ == "__main__":
    df, label_list = load_dataset(DATASET_DIR, LABEL_COLUMN)
    train_data, test_data = train_test_split(df, test_size=0.25, stratify=df[LABEL_COLUMN])
    
    model_checkpoint = "distilbert-base-uncased"
    max_length = 128
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    bert_model = BERTModel(model_checkpoint, max_length, tokenizer, label_list)
    
    encoded_train = bert_model.preprocess_function(train_data)
    encoded_test = bert_model.preprocess_function(test_data)
    
    bert_model.train(encoded_train)
    bert_loss, bert_acc = bert_model.evaluate(encoded_test)
    
    print(f"BERT Loss: {bert_loss}, BERT Accuracy: {bert_acc}")
    
    