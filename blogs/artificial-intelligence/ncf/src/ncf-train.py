import sys
import pandas as pd
import tensorflow as tf
import logging
#tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import ndcg_at_k

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 100
BATCH_SIZE = 28

SEED = 42
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

train_file = "./train.csv"
test_file = "./test.csv"

data = NCFDataset(train_file=train_file, test_file=test_file, seed=SEED)


model = NCF (
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=8,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=1,
    seed=SEED
)

print('Started training')

with Timer() as train_time:
    model.fit(data)
print("Took {} seconds for training.".format(train_time))

model.save('.')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

with Timer() as test_time:
    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item) 
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))
        # break

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()]#.drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time))


eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
print("NDCG:\t%f" % eval_ndcg)