import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import datasets
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import BertConfig
import random
import logging
import tensorflow as tf
from tensorflow import keras

# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 5 # Batch-size for pretraining the model
MAX_EPOCHS = 200  # Maximum number of epochs to train the model for
LEARNING_RATE = 2e-5  # Learning rate for training the model

MODEL_CHECKPOINT = "bert-base-cased"  # Name of pretrained model from Model Hub

# tokenized_dataset_train = datasets.load_from_disk('./wikiTokenizedTrain.hf')
tokenized_dataset_valid = datasets.load_from_disk('./wikiTokenizedValid.hf')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

collater = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB, return_tensors="tf"
)

valid = tokenized_dataset_valid.to_tf_dataset(
    columns=["input_ids", "token_type_ids", "attention_mask"],
    label_cols=["labels", "next_sentence_label"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
    collate_fn=collater,
)

del tokenized_dataset_valid

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
valid = valid.with_options(options)


config = BertConfig.from_pretrained('bert-base-cased')
print(tf.config.list_physical_devices())
from transformers import TFBertForPreTraining


try:
  # Specify an invalid GPU device
    with tf.device('/device:GPU:1'):
        print(os.environ['TF_CPP_MIN_LOG_LEVEL'])
        model = TFBertForPreTraining(config)

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        model.compile(optimizer=optimizer)

        model.fit(valid,epochs=MAX_EPOCHS,callbacks = [tf.keras.callbacks.ModelCheckpoint( filepath='./bert-base-HG-checkpoints-B32/{epoch:02d}.ckpt')])
except RuntimeError as e:
    print(e)
