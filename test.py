from tqdm import tqdm

from generator import CSVBatchGenerator

valid_files = '/home/daniel/DeepSpeech/data/readylingua-en/readylingua-en-dev.csv'
valid_batches = 1
batch_size = 16

data_valid = CSVBatchGenerator(valid_files, n_batches=valid_batches,
                               batch_size=batch_size)

for _ in tqdm(range(len(data_valid))):
    batch_inputs, _ = next(data_valid)