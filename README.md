# MultiSentimentRecognition

This tool helps to prepare data and make sentiment classification.
It contains 4 parts:



## Preprocessing

Here are some functions for some data preparation.
**remove_url(text)**
text: str - input text
_output_: str - text without URL
**remove_special_characters(text, drop_digit=True, drop_cyrillic=True)**
text: str - input text
drop_digit: boolean - flag of deleting digits
drop_cyrillic: boolean - flag of deleting Cyrillic symbols
_output_: str - text
**remove_extra_spaces(text)**
text: str - input text
_output_: str - text
**create_lemma(text, tokenizer, lemmatizer, punctuations, stop_words, get_lemma)**
text: str - input text
tokenizer - used tokenizer from nltk
lemmatizer - used lemmatizer from nltk
punctuations: list - list of punctuation to delete from output
stop_words: list - list of words to delete from output
get_lemma: boolean - flag of making lemmas
output: list - list of tokens or lemmas

## Machine Translation
**translation(text, target_lang, use_list, get_large_trans)**
text: str or list - input text
target_lang: str or list - if string, this language (or 'auto') is used for all input. If list, it must be the same length as text input
use_list: boolean - flag of having several input texts
get_large_trans: boolean - flag of getting translations for text which is more than 5000 characters
_output_: str or list - translated text
## Embeddings
**encode(word)**
word: str - word
_output_: int - idx for word in embedding model if word exist in this model, else idx for 'UNK'
**make_emb(emb_vectors, sent, emb_type, min_length, max_length)**
emb_vectors - used embedding model
sent: str or list - raw sentence or list of tokens
emb_type: str - type of returned embeddings
min_length: int - minimal length of sentence
max_length: int - maximal amount of words in sentence
_output_: list - if emb_type = 'mean' then returns mean embeddings of every word of size [N], N refers to size of each embedding length. If emb_type = 'sequence' then returns list of idx of every word in sentence in embedding model of size [M], M refers to length of sentence

## LSTM
**create_nn(use_embed_layer, embed_size, hidden_size, drop_rate, sequence_length, layer_type, bidir_flg, num_layers)**
use_embed_layer: boolean - flag of using nn.Embedding layer. If False then nn.Linear is used
embed_size: int - size of embed layer
hidden_size: int - size of hidden layer
drop_rate: float - rate of dropout in nn.LSTM
sequence_length: int - length of sentences in batches
layer_type: str - type of recurrent layer, whether 'RNN', 'LSTM' or 'GRU'
bidir_flg: boolean - flag of using bidirectional recurrent layer
num_layers: int - number of recurrent layers
_output_ - Neural Network

**data_preprocess(df, text_column, label_column, datatype_column, datatype, fillna, shuffle_flg)**
df: pd.DataFrame - input dataframe
text_column: str - name of text column
label_column: str - name of label column
datatype_column: str - name of column with data type
datatype: str - name of data type which should be left in output
fillna: 'drop' or list - way of filling NaN. If 'drop' then all missing data is dropped. If list then it should contain one int for label column and one string for text and data type column
shuffle_flg: boolean - flag of shuffling the dataframe
_output_: pd.DataFrame - output dataframe
**create_embeddings(df, text_column, label_column, model, emb_type, min_length, max_length, train_test_rate, datatype_column, test_type)**
df: pd.DataFrame - input dataframe
text_column: str - name of text column
label_column: str - name of label column
model - embedding model
emb_type - type of returned embeddings as in **make_emb** 
min_length: int - minimal length of sentence
max_length: int - maximal amount of words in sentence
train_test_rate: float - rate of spllit for train and test dataframes
datatype_column: str - name of column with data type
test_type: str - if 'all' then dataset splits as in train_test_rate. If another then set data type is used in test dataframe and other data types are used in train dataset
_output_:pd.Dataframe - output dataframe

**get_collator(max_len, text_column, label_column)**
max_len: int - max sentence length in batch
text_column: str - name of text column
label_column: str - name of label column
_output_: Collator - collator for making batches
 
**create_dataloaders_kfold(df, n_splits, batch_size, collate_fn, get_one, num_iter)**
df: pd.Dataframe - input dataframe
n_splits: int - number of k-fold splits. If 1 then there is no cross-validation
batch_size: int - batch size
collate_fn: Collator - Collator from **get_collator**
get_one: boolean - flag of returning all cross validation iteration. If False then returns only 1 iteration
num_iter: int - number of returned iteration if get_one = False
_output_: Dataloader

**training(model, x_column, y_column, criterion, optimizer,  scheduler, num_epochs, loaders, hyperparameters, df_result)**
model - neural network model
x_column: str - name of text column
y_column: str - name of label column
criterion - loss function
optimizer - torch optimizer
scheduler - torch scheduler
num_epochs: int - number of epochs
loaders: Dataloader - dataloaders from **create_dataloaders_kfold**
hyperparameters: list - list of hyperparameters to add to output dataframe
df_result: pd.DataFrame - dataframe to add hyperparameters ans results of training (accuracy, f1-score and loss for train and test)
_output_: pd.DataFrame - dataframe with hyperparameters ans results of training (accuracy, f1-score and loss for train and test)
