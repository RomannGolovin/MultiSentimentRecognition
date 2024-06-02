# MultiSentimentRecognition

This tool helps to prepare data and make sentiment classification.
It contains 4 parts:

## Preprocessing
Here are some functions for some data preparation.
**remove_url(text)**
text: str - input text <br />
_output_: str - text without URL <br />

**remove_special_characters(text, drop_digit=True, drop_cyrillic=True)**
text: str - input text <br />
drop_digit: boolean - flag of deleting digits <br />
drop_cyrillic: boolean - flag of deleting Cyrillic symbols <br />
_output_: str - text <br />

**remove_extra_spaces(text)**
text: str - input text <br />
_output_: str - text <br />

**create_lemma(text, tokenizer, lemmatizer, punctuations, stop_words, get_lemma)**
text: str - input text <br />
tokenizer - used tokenizer from nltk <br />
lemmatizer - used lemmatizer from nltk <br />
punctuations: list - list of punctuation to delete from output <br />
stop_words: list - list of words to delete from output <br />
get_lemma: boolean - flag of making lemmas <br />
output: list - list of tokens or lemmas <br />

## Machine Translation
Here are some functions for some automatic translation for texts and lists of texts.
**translation(text, target_lang, use_list, get_large_trans)**
text: str or list - input text <br />
target_lang: str or list - if string, this language (or 'auto') is used for all input. If list, it must be the same length as text input <br />
use_list: boolean - flag of having several input texts <br />
get_large_trans: boolean - flag of getting translations for text which is more than 5000 characters <br />
_output_: str or list - translated text <br />

## Embeddings
Here are some functions for using embeddinf.
**encode(word)**
word: str - word <br />
_output_: int - idx for word in embedding model if word exist in this model, else idx for 'UNK' <br />

**make_emb(emb_vectors, sent, emb_type, min_length, max_length)**
emb_vectors - used embedding model <br />
sent: str or list - raw sentence or list of tokens <br />
emb_type: str - type of returned embeddings <br />
min_length: int - minimal length of sentence <br />
max_length: int - maximal amount of words in sentence <br />
_output_: list - if emb_type = 'mean' then returns mean embeddings of every word of size [N], N refers to size of each  <br />embedding length. If emb_type = 'sequence' then returns list of idx of every word in sentence in embedding model of size [M], M refers to length of sentence <br />

## LSTM
Here are some functions for using neural network models.
**create_nn(use_embed_layer, embed_size, hidden_size, drop_rate, sequence_length, layer_type, bidir_flg, num_layers)**
use_embed_layer: boolean - flag of using nn.Embedding layer. If False then nn.Linear is used <br />
embed_size: int - size of embed layer <br />
hidden_size: int - size of hidden layer <br />
drop_rate: float - rate of dropout in nn.LSTM <br />
sequence_length: int - length of sentences in batches <br />
layer_type: str - type of recurrent layer, whether 'RNN', 'LSTM' or 'GRU' <br />
bidir_flg: boolean - flag of using bidirectional recurrent layer <br />
num_layers: int - number of recurrent layers <br />
_output_ - Neural Network <br />

**data_preprocess(df, text_column, label_column, datatype_column, datatype, fillna, shuffle_flg)**
df: pd.DataFrame - input dataframe <br />
text_column: str - name of text column <br />
label_column: str - name of label column <br />
datatype_column: str - name of column with data type <br />
datatype: str - name of data type which should be left in output <br />
fillna: 'drop' or list - way of filling NaN. If 'drop' then all missing data is dropped. If list then it should contain one int for label column and one string for text and data type column <br />
shuffle_flg: boolean - flag of shuffling the dataframe <br />
_output_: pd.DataFrame - output dataframe <br />

**create_embeddings(df, text_column, label_column, model, emb_type, min_length, max_length, train_test_rate, datatype_column, test_type)**
df: pd.DataFrame - input dataframe <br />
text_column: str - name of text column <br />
label_column: str - name of label column <br />
model - embedding model <br />
emb_type - type of returned embeddings as in **make_emb**  <br />
min_length: int - minimal length of sentence <br />
max_length: int - maximal amount of words in sentence <br />
train_test_rate: float - rate of spllit for train and test dataframes <br />
datatype_column: str - name of column with data type <br />
test_type: str - if 'all' then dataset splits as in train_test_rate. If another then set data type is used in test dataframe and other data types are used in train dataset <br />
_output_:pd.Dataframe - output dataframe <br />

**get_collator(max_len, text_column, label_column)**
max_len: int - max sentence length in batch <br />
text_column: str - name of text column <br />
label_column: str - name of label column <br />
_output_: Collator - collator for making batches <br />
 
**create_dataloaders_kfold(df, n_splits, batch_size, collate_fn, get_one, num_iter)**
df: pd.Dataframe - input dataframe <br />
n_splits: int - number of k-fold splits. If 1 then there is no cross-validation <br />
batch_size: int - batch size <br />
collate_fn: Collator - Collator from **get_collator** <br />
get_one: boolean - flag of returning all cross validation iteration. If False then returns only 1 iteration <br />
num_iter: int - number of returned iteration if get_one = False <br />
_output_: Dataloader <br />

**training(model, x_column, y_column, criterion, optimizer,  scheduler, num_epochs, loaders, hyperparameters, df_result)**
model - neural network model <br />
x_column: str - name of text column <br />
y_column: str - name of label column <br />
criterion - loss function <br />
optimizer - torch optimizer <br />
scheduler - torch scheduler <br />
num_epochs: int - number of epochs <br />
loaders: Dataloader - dataloaders from **create_dataloaders_kfold** <br />
hyperparameters: list - list of hyperparameters to add to output dataframe <br />
df_result: pd.DataFrame - dataframe to add hyperparameters ans results of training (accuracy, f1-score and loss for train and test) <br />
_output_: pd.DataFrame - dataframe with hyperparameters ans results of training (accuracy, f1-score and loss for train and test) <br />
