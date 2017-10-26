
# 语言翻译

在此项目中，你将了解神经网络机器翻译这一领域。你将用由英语和法语语句组成的数据集，训练一个序列到序列模型（sequence to sequence model），该模型能够将新的英语句子翻译成法语。

## 获取数据

因为将整个英语语言内容翻译成法语需要大量训练时间，所以我们提供了一小部分的英语语料库。



```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## 探索数据

研究 view_sentence_range，查看并熟悉该数据的不同部分。



```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## 实现预处理函数

### 文本到单词 id

和之前的 RNN 一样，你必须首先将文本转换为数字，这样计算机才能读懂。在函数 `text_to_ids()` 中，你需要将单词中的 `source_text` 和 `target_text` 转为 id。但是，你需要在 `target_text` 中每个句子的末尾，添加 `<EOS>` 单词 id。这样可以帮助神经网络预测句子应该在什么地方结束。


你可以通过以下代码获取  `<EOS> ` 单词ID：

```python
target_vocab_to_int['<EOS>']
```

你可以使用 `source_vocab_to_int` 和 `target_vocab_to_int` 获得其他单词 id。



```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_sentences = [sentence for sentence in source_text.split('\n')]
    target_sentences = [sentence + ' <EOS>' for sentence in target_text.split('\n')]
    
    source_id_text = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source_sentences]
    target_id_text = [[target_vocab_to_int[word] for word in sentence.split()] for sentence in target_sentences]
    
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed


### 预处理所有数据并保存

运行以下代码单元，预处理所有数据，并保存到文件中。



```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# 检查点

这是你的第一个检查点。如果你什么时候决定再回到该记事本，或需要重新启动该记事本，可以从这里继续。预处理的数据已保存到磁盘上。


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### 检查 TensorFlow 版本，确认可访问 GPU

这一检查步骤，可以确保你使用的是正确版本的 TensorFlow，并且能够访问 GPU。



```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


## 构建神经网络

你将通过实现以下函数，构建出要构建一个序列到序列模型所需的组件：

- `model_inputs`
- `process_decoding_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### 输入

实现 `model_inputs()` 函数，为神经网络创建 TF 占位符。该函数应该创建以下占位符：

- 名为 “input” 的输入文本占位符，并使用 TF Placeholder 名称参数（等级（Rank）为 2）。
- 目标占位符（等级为 2）。
- 学习速率占位符（等级为 0）。
- 名为 “keep_prob” 的保留率占位符，并使用 TF Placeholder 名称参数（等级为 0）。

在以下元祖（tuple）中返回占位符：（输入、目标、学习速率、保留率）



```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    input_ = tf.placeholder(tf.int32, [None, None], 'input')
    target_ = tf.placeholder(tf.int32, [None, None], 'target')
    lr_ = tf.placeholder(tf.float32, None, 'lr')
    keep_prob_ = tf.placeholder(tf.float32, None, 'keep_prob')
    return input_, target_, lr_, keep_prob_

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### 处理解码输入

使用 TensorFlow 实现 `process_decoding_input`，以便删掉 `target_data` 中每个批次的最后一个单词 ID，并将 GO ID 放到每个批次的开头。


```python
def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    a = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    b = tf.fill([batch_size, 1], target_vocab_to_int['<GO>'])
    c = tf.concat([b, a], 1)
    
    return c

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)
```

    Tests Passed


### 编码

实现 `encoding_layer()`，以使用 [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn) 创建编码器 RNN 层级。


```python
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)

    _, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
    return final_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### 解码 - 训练

使用 [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) 和 [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) 创建训练分对数（training logits）。将 `output_fn` 应用到 [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) 输出上。


```python
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    dec_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    outputs_train, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                                 dec_fn_train, 
                                                                 dec_embed_input, 
                                                                 sequence_length, 
                                                                 scope=decoding_scope)
    logits_train = output_fn(outputs_train)
    logits = tf.nn.dropout(logits_train, keep_prob)
    return logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### 解码 - 推论

使用 [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) 和 [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) 创建推论分对数（inference logits）。


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    dec_fn_infer = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, 
                                                                  encoder_state, 
                                                                  dec_embeddings, 
                                                                  start_of_sequence_id, 
                                                                  end_of_sequence_id, 
                                                                  maximum_length, 
                                                                  vocab_size)
    dropouted = tf.contrib.rnn.DropoutWrapper(dec_cell, keep_prob)
    logits_infer, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dropouted, dec_fn_infer, 
                                                                sequence_length=maximum_length, 
                                                                scope=decoding_scope)
    
    return logits_infer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### 构建解码层级

实现 `decoding_layer()` 以创建解码器 RNN 层级。

- 使用 `rnn_size` 和 `num_layers` 创建解码 RNN 单元。
- 使用 [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) 创建输出函数，将输入，也就是分对数转换为类分对数（class logits）。
- 使用 `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` 函数获取训练分对数。
- 使用 `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` 函数获取推论分对数。

注意：你将需要使用 [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) 在训练和推论分对数间分享变量。


```python
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    dec_cell = tf.contrib.rnn.MultiRNNCell([dropout] * num_layers)
    
    max_target_sentence_length = max([len(sentence) for sentence in source_int_text])
    
    with tf.variable_scope('decoding_layer') as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
    
    with tf.variable_scope('decoding_layer') as decoding_scope:
        logits_train = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                                            sequence_length, decoding_scope, output_fn, keep_prob)
    
    with tf.variable_scope('decoding_layer', reuse=True) as decoding_scope:
        logits_infer = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, 
                                            target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], 
                                            max_target_sentence_length, 
                                            vocab_size, decoding_scope, output_fn, keep_prob)
    
    return logits_train, logits_infer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### 构建神经网络

应用你在上方实现的函数，以：

- 向编码器的输入数据应用嵌入。
- 使用 `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)` 编码输入。
- 使用 `process_decoding_input(target_data, target_vocab_to_int, batch_size)` 函数处理目标数据。
- 向解码器的目标数据应用嵌入。
- 使用 `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)` 解码编码的输入数据。


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
    
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.truncated_normal([target_vocab_size, dec_embedding_size], stddev=0.01))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    logits_train, logits_infer = decoding_layer(dec_embed_input, dec_embeddings, 
                                                enc_state, target_vocab_size, 
                                                sequence_length, rnn_size, num_layers, 
                                                target_vocab_to_int, keep_prob)
    
    return logits_train, logits_infer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## 训练神经网络

### 超参数

调试以下参数：

- 将 `epochs` 设为 epoch 次数。
- 将 `batch_size` 设为批次大小。
- 将 `rnn_size` 设为 RNN 的大小。
- 将 `num_layers` 设为层级数量。
- 将 `encoding_embedding_size` 设为编码器嵌入大小。
- 将 `decoding_embedding_size` 设为解码器嵌入大小
- 将 `learning_rate` 设为训练速率。
- 将 `keep_probability` 设为丢弃保留率（Dropout keep probability）。


```python
# Number of Epochs
epochs = 2
# Batch Size
batch_size = 100
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 150
decoding_embedding_size = 150
# Learning Rate
learning_rate = 0.01
# Dropout Keep Probability
keep_probability = 0.9
```

### 构建图表

使用你实现的神经网络构建图表。


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
# save_path = './save'

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
```

### 训练

利用预处理的数据训练神经网络。如果很难获得低损失值，请访问我们的论坛，看看其他人是否遇到了相同的问题。


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/1378 - Train Accuracy:  0.317, Validation Accuracy:  0.354, Loss:  5.881
    Epoch   0 Batch    1/1378 - Train Accuracy:  0.262, Validation Accuracy:  0.318, Loss:  5.740
    Epoch   0 Batch    2/1378 - Train Accuracy:  0.173, Validation Accuracy:  0.239, Loss:  4.219
    Epoch   0 Batch    3/1378 - Train Accuracy:  0.107, Validation Accuracy:  0.225, Loss:  4.473
    Epoch   0 Batch    4/1378 - Train Accuracy:  0.243, Validation Accuracy:  0.322, Loss:  4.042
    Epoch   0 Batch    5/1378 - Train Accuracy:  0.277, Validation Accuracy:  0.348, Loss:  3.921
    Epoch   0 Batch    6/1378 - Train Accuracy:  0.239, Validation Accuracy:  0.325, Loss:  3.861
    Epoch   0 Batch    7/1378 - Train Accuracy:  0.301, Validation Accuracy:  0.352, Loss:  3.758
    Epoch   0 Batch    8/1378 - Train Accuracy:  0.295, Validation Accuracy:  0.350, Loss:  3.621
    Epoch   0 Batch    9/1378 - Train Accuracy:  0.275, Validation Accuracy:  0.360, Loss:  3.675
    Epoch   0 Batch   10/1378 - Train Accuracy:  0.303, Validation Accuracy:  0.380, Loss:  3.551
    Epoch   0 Batch   11/1378 - Train Accuracy:  0.318, Validation Accuracy:  0.382, Loss:  3.494
    Epoch   0 Batch   12/1378 - Train Accuracy:  0.289, Validation Accuracy:  0.340, Loss:  3.407
    Epoch   0 Batch   13/1378 - Train Accuracy:  0.320, Validation Accuracy:  0.382, Loss:  3.473
    Epoch   0 Batch   14/1378 - Train Accuracy:  0.297, Validation Accuracy:  0.375, Loss:  3.495
    Epoch   0 Batch   15/1378 - Train Accuracy:  0.327, Validation Accuracy:  0.385, Loss:  3.451
    Epoch   0 Batch   16/1378 - Train Accuracy:  0.303, Validation Accuracy:  0.365, Loss:  3.300
    Epoch   0 Batch   17/1378 - Train Accuracy:  0.359, Validation Accuracy:  0.390, Loss:  3.237
    Epoch   0 Batch   18/1378 - Train Accuracy:  0.348, Validation Accuracy:  0.385, Loss:  3.151
    Epoch   0 Batch   19/1378 - Train Accuracy:  0.303, Validation Accuracy:  0.390, Loss:  3.318
    Epoch   0 Batch   20/1378 - Train Accuracy:  0.290, Validation Accuracy:  0.388, Loss:  3.275
    Epoch   0 Batch   21/1378 - Train Accuracy:  0.353, Validation Accuracy:  0.403, Loss:  3.095
    Epoch   0 Batch   22/1378 - Train Accuracy:  0.364, Validation Accuracy:  0.410, Loss:  3.076
    Epoch   0 Batch   23/1378 - Train Accuracy:  0.340, Validation Accuracy:  0.424, Loss:  3.175
    Epoch   0 Batch   24/1378 - Train Accuracy:  0.356, Validation Accuracy:  0.421, Loss:  3.059
    Epoch   0 Batch   25/1378 - Train Accuracy:  0.353, Validation Accuracy:  0.425, Loss:  3.107
    Epoch   0 Batch   26/1378 - Train Accuracy:  0.376, Validation Accuracy:  0.425, Loss:  2.865
    Epoch   0 Batch   27/1378 - Train Accuracy:  0.328, Validation Accuracy:  0.436, Loss:  3.066
    Epoch   0 Batch   28/1378 - Train Accuracy:  0.371, Validation Accuracy:  0.443, Loss:  2.899
    Epoch   0 Batch   29/1378 - Train Accuracy:  0.385, Validation Accuracy:  0.451, Loss:  2.795
    Epoch   0 Batch   30/1378 - Train Accuracy:  0.369, Validation Accuracy:  0.456, Loss:  2.797
    Epoch   0 Batch   31/1378 - Train Accuracy:  0.385, Validation Accuracy:  0.444, Loss:  2.669
    Epoch   0 Batch   32/1378 - Train Accuracy:  0.380, Validation Accuracy:  0.458, Loss:  2.707
    Epoch   0 Batch   33/1378 - Train Accuracy:  0.369, Validation Accuracy:  0.445, Loss:  2.619
    Epoch   0 Batch   34/1378 - Train Accuracy:  0.401, Validation Accuracy:  0.468, Loss:  2.647
    Epoch   0 Batch   35/1378 - Train Accuracy:  0.423, Validation Accuracy:  0.462, Loss:  2.415
    Epoch   0 Batch   36/1378 - Train Accuracy:  0.393, Validation Accuracy:  0.440, Loss:  2.512
    Epoch   0 Batch   37/1378 - Train Accuracy:  0.317, Validation Accuracy:  0.408, Loss:  2.467
    Epoch   0 Batch   38/1378 - Train Accuracy:  0.350, Validation Accuracy:  0.438, Loss:  2.400
    Epoch   0 Batch   39/1378 - Train Accuracy:  0.352, Validation Accuracy:  0.437, Loss:  2.387
    Epoch   0 Batch   40/1378 - Train Accuracy:  0.379, Validation Accuracy:  0.450, Loss:  2.421
    Epoch   0 Batch   41/1378 - Train Accuracy:  0.397, Validation Accuracy:  0.420, Loss:  2.196
    Epoch   0 Batch   42/1378 - Train Accuracy:  0.320, Validation Accuracy:  0.340, Loss:  2.183
    Epoch   0 Batch   43/1378 - Train Accuracy:  0.352, Validation Accuracy:  0.347, Loss:  2.089
    Epoch   0 Batch   44/1378 - Train Accuracy:  0.274, Validation Accuracy:  0.341, Loss:  2.170
    Epoch   0 Batch   45/1378 - Train Accuracy:  0.266, Validation Accuracy:  0.345, Loss:  2.116
    Epoch   0 Batch   46/1378 - Train Accuracy:  0.301, Validation Accuracy:  0.348, Loss:  2.019
    Epoch   0 Batch   47/1378 - Train Accuracy:  0.357, Validation Accuracy:  0.407, Loss:  2.021
    Epoch   0 Batch   48/1378 - Train Accuracy:  0.307, Validation Accuracy:  0.372, Loss:  1.940
    Epoch   0 Batch   49/1378 - Train Accuracy:  0.264, Validation Accuracy:  0.368, Loss:  2.139
    Epoch   0 Batch   50/1378 - Train Accuracy:  0.370, Validation Accuracy:  0.390, Loss:  1.911
    Epoch   0 Batch   51/1378 - Train Accuracy:  0.358, Validation Accuracy:  0.440, Loss:  1.903
    Epoch   0 Batch   52/1378 - Train Accuracy:  0.338, Validation Accuracy:  0.444, Loss:  1.868
    Epoch   0 Batch   53/1378 - Train Accuracy:  0.379, Validation Accuracy:  0.447, Loss:  1.811
    Epoch   0 Batch   54/1378 - Train Accuracy:  0.328, Validation Accuracy:  0.365, Loss:  1.750
    Epoch   0 Batch   55/1378 - Train Accuracy:  0.259, Validation Accuracy:  0.345, Loss:  1.792
    Epoch   0 Batch   56/1378 - Train Accuracy:  0.252, Validation Accuracy:  0.352, Loss:  1.667
    Epoch   0 Batch   57/1378 - Train Accuracy:  0.371, Validation Accuracy:  0.460, Loss:  1.671
    Epoch   0 Batch   58/1378 - Train Accuracy:  0.345, Validation Accuracy:  0.440, Loss:  1.687
    Epoch   0 Batch   59/1378 - Train Accuracy:  0.347, Validation Accuracy:  0.372, Loss:  1.676
    Epoch   0 Batch   60/1378 - Train Accuracy:  0.306, Validation Accuracy:  0.375, Loss:  1.560
    Epoch   0 Batch   61/1378 - Train Accuracy:  0.301, Validation Accuracy:  0.380, Loss:  1.648
    Epoch   0 Batch   62/1378 - Train Accuracy:  0.334, Validation Accuracy:  0.380, Loss:  1.615
    Epoch   0 Batch   63/1378 - Train Accuracy:  0.351, Validation Accuracy:  0.384, Loss:  1.547
    Epoch   0 Batch   64/1378 - Train Accuracy:  0.272, Validation Accuracy:  0.377, Loss:  1.529
    Epoch   0 Batch   65/1378 - Train Accuracy:  0.313, Validation Accuracy:  0.375, Loss:  1.577
    Epoch   0 Batch   66/1378 - Train Accuracy:  0.373, Validation Accuracy:  0.379, Loss:  1.455
    Epoch   0 Batch   67/1378 - Train Accuracy:  0.344, Validation Accuracy:  0.384, Loss:  1.435
    Epoch   0 Batch   68/1378 - Train Accuracy:  0.396, Validation Accuracy:  0.470, Loss:  1.485
    Epoch   0 Batch   69/1378 - Train Accuracy:  0.410, Validation Accuracy:  0.470, Loss:  1.505
    Epoch   0 Batch   70/1378 - Train Accuracy:  0.384, Validation Accuracy:  0.463, Loss:  1.534
    Epoch   0 Batch   71/1378 - Train Accuracy:  0.390, Validation Accuracy:  0.464, Loss:  1.349
    Epoch   0 Batch   72/1378 - Train Accuracy:  0.398, Validation Accuracy:  0.458, Loss:  1.402
    Epoch   0 Batch   73/1378 - Train Accuracy:  0.439, Validation Accuracy:  0.461, Loss:  1.325
    Epoch   0 Batch   74/1378 - Train Accuracy:  0.384, Validation Accuracy:  0.462, Loss:  1.397
    Epoch   0 Batch   75/1378 - Train Accuracy:  0.374, Validation Accuracy:  0.468, Loss:  1.390
    Epoch   0 Batch   76/1378 - Train Accuracy:  0.356, Validation Accuracy:  0.457, Loss:  1.468
    Epoch   0 Batch   77/1378 - Train Accuracy:  0.414, Validation Accuracy:  0.467, Loss:  1.356
    Epoch   0 Batch   78/1378 - Train Accuracy:  0.406, Validation Accuracy:  0.476, Loss:  1.356
    Epoch   0 Batch   79/1378 - Train Accuracy:  0.442, Validation Accuracy:  0.470, Loss:  1.316
    Epoch   0 Batch   80/1378 - Train Accuracy:  0.381, Validation Accuracy:  0.486, Loss:  1.422
    Epoch   0 Batch   81/1378 - Train Accuracy:  0.453, Validation Accuracy:  0.491, Loss:  1.302
    Epoch   0 Batch   82/1378 - Train Accuracy:  0.412, Validation Accuracy:  0.503, Loss:  1.336
    Epoch   0 Batch   83/1378 - Train Accuracy:  0.383, Validation Accuracy:  0.468, Loss:  1.305
    Epoch   0 Batch   84/1378 - Train Accuracy:  0.421, Validation Accuracy:  0.480, Loss:  1.293
    Epoch   0 Batch   85/1378 - Train Accuracy:  0.402, Validation Accuracy:  0.500, Loss:  1.305
    Epoch   0 Batch   86/1378 - Train Accuracy:  0.530, Validation Accuracy:  0.524, Loss:  1.243
    Epoch   0 Batch   87/1378 - Train Accuracy:  0.441, Validation Accuracy:  0.497, Loss:  1.264
    Epoch   0 Batch   88/1378 - Train Accuracy:  0.445, Validation Accuracy:  0.502, Loss:  1.269
    Epoch   0 Batch   89/1378 - Train Accuracy:  0.443, Validation Accuracy:  0.487, Loss:  1.389
    Epoch   0 Batch   90/1378 - Train Accuracy:  0.424, Validation Accuracy:  0.484, Loss:  1.276
    Epoch   0 Batch   91/1378 - Train Accuracy:  0.421, Validation Accuracy:  0.507, Loss:  1.285
    Epoch   0 Batch   92/1378 - Train Accuracy:  0.433, Validation Accuracy:  0.526, Loss:  1.254
    Epoch   0 Batch   93/1378 - Train Accuracy:  0.469, Validation Accuracy:  0.544, Loss:  1.316
    Epoch   0 Batch   94/1378 - Train Accuracy:  0.463, Validation Accuracy:  0.497, Loss:  1.234
    Epoch   0 Batch   95/1378 - Train Accuracy:  0.439, Validation Accuracy:  0.479, Loss:  1.245
    Epoch   0 Batch   96/1378 - Train Accuracy:  0.486, Validation Accuracy:  0.480, Loss:  1.210
    Epoch   0 Batch   97/1378 - Train Accuracy:  0.470, Validation Accuracy:  0.514, Loss:  1.248
    Epoch   0 Batch   98/1378 - Train Accuracy:  0.463, Validation Accuracy:  0.528, Loss:  1.198
    Epoch   0 Batch   99/1378 - Train Accuracy:  0.462, Validation Accuracy:  0.525, Loss:  1.277
    Epoch   0 Batch  100/1378 - Train Accuracy:  0.472, Validation Accuracy:  0.548, Loss:  1.230
    Epoch   0 Batch  101/1378 - Train Accuracy:  0.469, Validation Accuracy:  0.552, Loss:  1.309
    Epoch   0 Batch  102/1378 - Train Accuracy:  0.484, Validation Accuracy:  0.540, Loss:  1.223
    Epoch   0 Batch  103/1378 - Train Accuracy:  0.480, Validation Accuracy:  0.554, Loss:  1.229
    Epoch   0 Batch  104/1378 - Train Accuracy:  0.490, Validation Accuracy:  0.538, Loss:  1.172
    Epoch   0 Batch  105/1378 - Train Accuracy:  0.556, Validation Accuracy:  0.536, Loss:  1.085
    Epoch   0 Batch  106/1378 - Train Accuracy:  0.526, Validation Accuracy:  0.546, Loss:  1.204
    Epoch   0 Batch  107/1378 - Train Accuracy:  0.481, Validation Accuracy:  0.547, Loss:  1.208
    Epoch   0 Batch  108/1378 - Train Accuracy:  0.481, Validation Accuracy:  0.540, Loss:  1.185
    Epoch   0 Batch  109/1378 - Train Accuracy:  0.471, Validation Accuracy:  0.532, Loss:  1.212
    Epoch   0 Batch  110/1378 - Train Accuracy:  0.522, Validation Accuracy:  0.521, Loss:  1.170
    Epoch   0 Batch  111/1378 - Train Accuracy:  0.472, Validation Accuracy:  0.523, Loss:  1.224
    Epoch   0 Batch  112/1378 - Train Accuracy:  0.493, Validation Accuracy:  0.547, Loss:  1.203
    Epoch   0 Batch  113/1378 - Train Accuracy:  0.531, Validation Accuracy:  0.559, Loss:  1.203
    Epoch   0 Batch  114/1378 - Train Accuracy:  0.538, Validation Accuracy:  0.572, Loss:  1.235
    Epoch   0 Batch  115/1378 - Train Accuracy:  0.539, Validation Accuracy:  0.565, Loss:  1.207
    Epoch   0 Batch  116/1378 - Train Accuracy:  0.511, Validation Accuracy:  0.564, Loss:  1.185
    Epoch   0 Batch  117/1378 - Train Accuracy:  0.538, Validation Accuracy:  0.559, Loss:  1.163
    Epoch   0 Batch  118/1378 - Train Accuracy:  0.519, Validation Accuracy:  0.566, Loss:  1.157
    Epoch   0 Batch  119/1378 - Train Accuracy:  0.522, Validation Accuracy:  0.570, Loss:  1.222
    Epoch   0 Batch  120/1378 - Train Accuracy:  0.541, Validation Accuracy:  0.567, Loss:  1.115
    Epoch   0 Batch  121/1378 - Train Accuracy:  0.513, Validation Accuracy:  0.536, Loss:  1.085
    Epoch   0 Batch  122/1378 - Train Accuracy:  0.558, Validation Accuracy:  0.542, Loss:  1.196
    Epoch   0 Batch  123/1378 - Train Accuracy:  0.502, Validation Accuracy:  0.557, Loss:  1.197
    Epoch   0 Batch  124/1378 - Train Accuracy:  0.513, Validation Accuracy:  0.555, Loss:  1.163
    Epoch   0 Batch  125/1378 - Train Accuracy:  0.563, Validation Accuracy:  0.556, Loss:  1.099
    Epoch   0 Batch  126/1378 - Train Accuracy:  0.540, Validation Accuracy:  0.554, Loss:  1.130
    Epoch   0 Batch  127/1378 - Train Accuracy:  0.507, Validation Accuracy:  0.569, Loss:  1.175
    Epoch   0 Batch  128/1378 - Train Accuracy:  0.504, Validation Accuracy:  0.548, Loss:  1.164
    Epoch   0 Batch  129/1378 - Train Accuracy:  0.534, Validation Accuracy:  0.529, Loss:  1.142
    Epoch   0 Batch  130/1378 - Train Accuracy:  0.482, Validation Accuracy:  0.536, Loss:  1.167
    Epoch   0 Batch  131/1378 - Train Accuracy:  0.508, Validation Accuracy:  0.532, Loss:  1.201
    Epoch   0 Batch  132/1378 - Train Accuracy:  0.488, Validation Accuracy:  0.516, Loss:  1.208
    Epoch   0 Batch  133/1378 - Train Accuracy:  0.424, Validation Accuracy:  0.495, Loss:  1.138
    Epoch   0 Batch  134/1378 - Train Accuracy:  0.384, Validation Accuracy:  0.420, Loss:  1.168
    Epoch   0 Batch  135/1378 - Train Accuracy:  0.455, Validation Accuracy:  0.480, Loss:  1.160
    Epoch   0 Batch  136/1378 - Train Accuracy:  0.506, Validation Accuracy:  0.543, Loss:  1.231
    Epoch   0 Batch  137/1378 - Train Accuracy:  0.511, Validation Accuracy:  0.551, Loss:  1.151
    Epoch   0 Batch  138/1378 - Train Accuracy:  0.549, Validation Accuracy:  0.548, Loss:  1.126
    Epoch   0 Batch  139/1378 - Train Accuracy:  0.570, Validation Accuracy:  0.544, Loss:  1.056
    Epoch   0 Batch  140/1378 - Train Accuracy:  0.534, Validation Accuracy:  0.560, Loss:  1.147
    Epoch   0 Batch  141/1378 - Train Accuracy:  0.546, Validation Accuracy:  0.556, Loss:  1.030
    Epoch   0 Batch  142/1378 - Train Accuracy:  0.507, Validation Accuracy:  0.553, Loss:  1.160
    Epoch   0 Batch  143/1378 - Train Accuracy:  0.499, Validation Accuracy:  0.541, Loss:  1.106
    Epoch   0 Batch  144/1378 - Train Accuracy:  0.518, Validation Accuracy:  0.551, Loss:  1.121
    Epoch   0 Batch  145/1378 - Train Accuracy:  0.471, Validation Accuracy:  0.548, Loss:  1.108
    Epoch   0 Batch  146/1378 - Train Accuracy:  0.481, Validation Accuracy:  0.565, Loss:  1.115
    Epoch   0 Batch  147/1378 - Train Accuracy:  0.512, Validation Accuracy:  0.570, Loss:  1.073
    Epoch   0 Batch  148/1378 - Train Accuracy:  0.520, Validation Accuracy:  0.571, Loss:  1.066
    Epoch   0 Batch  149/1378 - Train Accuracy:  0.524, Validation Accuracy:  0.569, Loss:  1.113
    Epoch   0 Batch  150/1378 - Train Accuracy:  0.480, Validation Accuracy:  0.579, Loss:  1.162
    Epoch   0 Batch  151/1378 - Train Accuracy:  0.474, Validation Accuracy:  0.583, Loss:  1.194
    Epoch   0 Batch  152/1378 - Train Accuracy:  0.522, Validation Accuracy:  0.598, Loss:  1.093
    Epoch   0 Batch  153/1378 - Train Accuracy:  0.558, Validation Accuracy:  0.593, Loss:  1.066
    Epoch   0 Batch  154/1378 - Train Accuracy:  0.551, Validation Accuracy:  0.585, Loss:  1.148
    Epoch   0 Batch  155/1378 - Train Accuracy:  0.514, Validation Accuracy:  0.573, Loss:  1.090
    Epoch   0 Batch  156/1378 - Train Accuracy:  0.573, Validation Accuracy:  0.579, Loss:  1.089
    Epoch   0 Batch  157/1378 - Train Accuracy:  0.524, Validation Accuracy:  0.579, Loss:  1.112
    Epoch   0 Batch  158/1378 - Train Accuracy:  0.533, Validation Accuracy:  0.587, Loss:  1.076
    Epoch   0 Batch  159/1378 - Train Accuracy:  0.541, Validation Accuracy:  0.593, Loss:  1.064
    Epoch   0 Batch  160/1378 - Train Accuracy:  0.560, Validation Accuracy:  0.582, Loss:  1.048
    Epoch   0 Batch  161/1378 - Train Accuracy:  0.545, Validation Accuracy:  0.596, Loss:  1.050
    Epoch   0 Batch  162/1378 - Train Accuracy:  0.563, Validation Accuracy:  0.586, Loss:  0.997
    Epoch   0 Batch  163/1378 - Train Accuracy:  0.531, Validation Accuracy:  0.578, Loss:  1.072
    Epoch   0 Batch  164/1378 - Train Accuracy:  0.553, Validation Accuracy:  0.565, Loss:  1.091
    Epoch   0 Batch  165/1378 - Train Accuracy:  0.539, Validation Accuracy:  0.509, Loss:  1.033
    Epoch   0 Batch  166/1378 - Train Accuracy:  0.568, Validation Accuracy:  0.525, Loss:  1.034
    Epoch   0 Batch  167/1378 - Train Accuracy:  0.510, Validation Accuracy:  0.518, Loss:  0.990
    Epoch   0 Batch  168/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.568, Loss:  1.099
    Epoch   0 Batch  169/1378 - Train Accuracy:  0.545, Validation Accuracy:  0.584, Loss:  1.036
    Epoch   0 Batch  170/1378 - Train Accuracy:  0.516, Validation Accuracy:  0.605, Loss:  1.076
    Epoch   0 Batch  171/1378 - Train Accuracy:  0.564, Validation Accuracy:  0.582, Loss:  1.085
    Epoch   0 Batch  172/1378 - Train Accuracy:  0.554, Validation Accuracy:  0.560, Loss:  1.054
    Epoch   0 Batch  173/1378 - Train Accuracy:  0.501, Validation Accuracy:  0.542, Loss:  1.048
    Epoch   0 Batch  174/1378 - Train Accuracy:  0.528, Validation Accuracy:  0.565, Loss:  1.028
    Epoch   0 Batch  175/1378 - Train Accuracy:  0.542, Validation Accuracy:  0.591, Loss:  1.017
    Epoch   0 Batch  176/1378 - Train Accuracy:  0.596, Validation Accuracy:  0.609, Loss:  1.034
    Epoch   0 Batch  177/1378 - Train Accuracy:  0.591, Validation Accuracy:  0.606, Loss:  1.079
    Epoch   0 Batch  178/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.607, Loss:  1.017
    Epoch   0 Batch  179/1378 - Train Accuracy:  0.522, Validation Accuracy:  0.605, Loss:  1.112
    Epoch   0 Batch  180/1378 - Train Accuracy:  0.584, Validation Accuracy:  0.609, Loss:  1.058
    Epoch   0 Batch  181/1378 - Train Accuracy:  0.561, Validation Accuracy:  0.601, Loss:  1.088
    Epoch   0 Batch  182/1378 - Train Accuracy:  0.580, Validation Accuracy:  0.608, Loss:  1.009
    Epoch   0 Batch  183/1378 - Train Accuracy:  0.517, Validation Accuracy:  0.570, Loss:  1.053
    Epoch   0 Batch  184/1378 - Train Accuracy:  0.461, Validation Accuracy:  0.540, Loss:  1.095
    Epoch   0 Batch  185/1378 - Train Accuracy:  0.518, Validation Accuracy:  0.565, Loss:  1.093
    Epoch   0 Batch  186/1378 - Train Accuracy:  0.605, Validation Accuracy:  0.585, Loss:  1.057
    Epoch   0 Batch  187/1378 - Train Accuracy:  0.597, Validation Accuracy:  0.598, Loss:  1.017
    Epoch   0 Batch  188/1378 - Train Accuracy:  0.582, Validation Accuracy:  0.610, Loss:  1.047
    Epoch   0 Batch  189/1378 - Train Accuracy:  0.555, Validation Accuracy:  0.596, Loss:  1.092
    Epoch   0 Batch  190/1378 - Train Accuracy:  0.571, Validation Accuracy:  0.597, Loss:  1.056
    Epoch   0 Batch  191/1378 - Train Accuracy:  0.541, Validation Accuracy:  0.535, Loss:  1.059
    Epoch   0 Batch  192/1378 - Train Accuracy:  0.432, Validation Accuracy:  0.499, Loss:  1.074
    Epoch   0 Batch  193/1378 - Train Accuracy:  0.552, Validation Accuracy:  0.503, Loss:  0.962
    Epoch   0 Batch  194/1378 - Train Accuracy:  0.600, Validation Accuracy:  0.582, Loss:  0.948
    Epoch   0 Batch  195/1378 - Train Accuracy:  0.613, Validation Accuracy:  0.583, Loss:  1.041
    Epoch   0 Batch  196/1378 - Train Accuracy:  0.526, Validation Accuracy:  0.581, Loss:  1.105
    Epoch   0 Batch  197/1378 - Train Accuracy:  0.542, Validation Accuracy:  0.577, Loss:  1.003
    Epoch   0 Batch  198/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.602, Loss:  0.994
    Epoch   0 Batch  199/1378 - Train Accuracy:  0.532, Validation Accuracy:  0.610, Loss:  1.090
    Epoch   0 Batch  200/1378 - Train Accuracy:  0.585, Validation Accuracy:  0.602, Loss:  0.980
    Epoch   0 Batch  201/1378 - Train Accuracy:  0.590, Validation Accuracy:  0.620, Loss:  1.093
    Epoch   0 Batch  202/1378 - Train Accuracy:  0.577, Validation Accuracy:  0.586, Loss:  0.936
    Epoch   0 Batch  203/1378 - Train Accuracy:  0.592, Validation Accuracy:  0.591, Loss:  1.055
    Epoch   0 Batch  204/1378 - Train Accuracy:  0.583, Validation Accuracy:  0.602, Loss:  0.994
    Epoch   0 Batch  205/1378 - Train Accuracy:  0.617, Validation Accuracy:  0.610, Loss:  0.934
    Epoch   0 Batch  206/1378 - Train Accuracy:  0.567, Validation Accuracy:  0.614, Loss:  1.013
    Epoch   0 Batch  207/1378 - Train Accuracy:  0.547, Validation Accuracy:  0.605, Loss:  1.002
    Epoch   0 Batch  208/1378 - Train Accuracy:  0.604, Validation Accuracy:  0.605, Loss:  1.062
    Epoch   0 Batch  209/1378 - Train Accuracy:  0.561, Validation Accuracy:  0.586, Loss:  1.051
    Epoch   0 Batch  210/1378 - Train Accuracy:  0.574, Validation Accuracy:  0.587, Loss:  1.022
    Epoch   0 Batch  211/1378 - Train Accuracy:  0.547, Validation Accuracy:  0.606, Loss:  1.092
    Epoch   0 Batch  212/1378 - Train Accuracy:  0.589, Validation Accuracy:  0.617, Loss:  0.997
    Epoch   0 Batch  213/1378 - Train Accuracy:  0.565, Validation Accuracy:  0.591, Loss:  1.042
    Epoch   0 Batch  214/1378 - Train Accuracy:  0.582, Validation Accuracy:  0.599, Loss:  1.072
    Epoch   0 Batch  215/1378 - Train Accuracy:  0.571, Validation Accuracy:  0.592, Loss:  1.031
    Epoch   0 Batch  216/1378 - Train Accuracy:  0.585, Validation Accuracy:  0.625, Loss:  1.076
    Epoch   0 Batch  217/1378 - Train Accuracy:  0.593, Validation Accuracy:  0.624, Loss:  0.957
    Epoch   0 Batch  218/1378 - Train Accuracy:  0.567, Validation Accuracy:  0.623, Loss:  1.103
    Epoch   0 Batch  219/1378 - Train Accuracy:  0.599, Validation Accuracy:  0.603, Loss:  0.931
    Epoch   0 Batch  220/1378 - Train Accuracy:  0.540, Validation Accuracy:  0.589, Loss:  1.006
    Epoch   0 Batch  221/1378 - Train Accuracy:  0.590, Validation Accuracy:  0.585, Loss:  0.907
    Epoch   0 Batch  222/1378 - Train Accuracy:  0.514, Validation Accuracy:  0.567, Loss:  1.052
    Epoch   0 Batch  223/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.601, Loss:  1.009
    Epoch   0 Batch  224/1378 - Train Accuracy:  0.591, Validation Accuracy:  0.604, Loss:  1.001
    Epoch   0 Batch  225/1378 - Train Accuracy:  0.584, Validation Accuracy:  0.610, Loss:  1.026
    Epoch   0 Batch  226/1378 - Train Accuracy:  0.589, Validation Accuracy:  0.617, Loss:  0.989
    Epoch   0 Batch  227/1378 - Train Accuracy:  0.579, Validation Accuracy:  0.620, Loss:  1.005
    Epoch   0 Batch  228/1378 - Train Accuracy:  0.576, Validation Accuracy:  0.615, Loss:  0.999
    Epoch   0 Batch  229/1378 - Train Accuracy:  0.620, Validation Accuracy:  0.619, Loss:  0.952
    Epoch   0 Batch  230/1378 - Train Accuracy:  0.592, Validation Accuracy:  0.617, Loss:  1.006
    Epoch   0 Batch  231/1378 - Train Accuracy:  0.556, Validation Accuracy:  0.581, Loss:  0.935
    Epoch   0 Batch  232/1378 - Train Accuracy:  0.556, Validation Accuracy:  0.554, Loss:  1.030
    Epoch   0 Batch  233/1378 - Train Accuracy:  0.524, Validation Accuracy:  0.526, Loss:  0.995
    Epoch   0 Batch  234/1378 - Train Accuracy:  0.587, Validation Accuracy:  0.606, Loss:  0.964
    Epoch   0 Batch  235/1378 - Train Accuracy:  0.591, Validation Accuracy:  0.605, Loss:  0.951
    Epoch   0 Batch  236/1378 - Train Accuracy:  0.614, Validation Accuracy:  0.611, Loss:  0.985
    Epoch   0 Batch  237/1378 - Train Accuracy:  0.554, Validation Accuracy:  0.610, Loss:  1.020
    Epoch   0 Batch  238/1378 - Train Accuracy:  0.547, Validation Accuracy:  0.595, Loss:  1.038
    Epoch   0 Batch  239/1378 - Train Accuracy:  0.582, Validation Accuracy:  0.592, Loss:  1.026
    Epoch   0 Batch  240/1378 - Train Accuracy:  0.587, Validation Accuracy:  0.597, Loss:  1.017
    Epoch   0 Batch  241/1378 - Train Accuracy:  0.590, Validation Accuracy:  0.610, Loss:  0.958
    Epoch   0 Batch  242/1378 - Train Accuracy:  0.575, Validation Accuracy:  0.602, Loss:  1.009
    Epoch   0 Batch  243/1378 - Train Accuracy:  0.591, Validation Accuracy:  0.601, Loss:  1.048
    Epoch   0 Batch  244/1378 - Train Accuracy:  0.645, Validation Accuracy:  0.611, Loss:  0.943
    Epoch   0 Batch  245/1378 - Train Accuracy:  0.617, Validation Accuracy:  0.615, Loss:  0.934
    Epoch   0 Batch  246/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.618, Loss:  0.945
    Epoch   0 Batch  247/1378 - Train Accuracy:  0.610, Validation Accuracy:  0.626, Loss:  0.981
    Epoch   0 Batch  248/1378 - Train Accuracy:  0.615, Validation Accuracy:  0.634, Loss:  0.980
    Epoch   0 Batch  249/1378 - Train Accuracy:  0.623, Validation Accuracy:  0.635, Loss:  0.951
    Epoch   0 Batch  250/1378 - Train Accuracy:  0.593, Validation Accuracy:  0.634, Loss:  0.981
    Epoch   0 Batch  251/1378 - Train Accuracy:  0.617, Validation Accuracy:  0.634, Loss:  0.986
    Epoch   0 Batch  252/1378 - Train Accuracy:  0.626, Validation Accuracy:  0.601, Loss:  0.962
    Epoch   0 Batch  253/1378 - Train Accuracy:  0.581, Validation Accuracy:  0.604, Loss:  1.026
    Epoch   0 Batch  254/1378 - Train Accuracy:  0.621, Validation Accuracy:  0.602, Loss:  0.912
    Epoch   0 Batch  255/1378 - Train Accuracy:  0.595, Validation Accuracy:  0.600, Loss:  0.945
    Epoch   0 Batch  256/1378 - Train Accuracy:  0.592, Validation Accuracy:  0.588, Loss:  0.968
    Epoch   0 Batch  257/1378 - Train Accuracy:  0.572, Validation Accuracy:  0.607, Loss:  0.958
    Epoch   0 Batch  258/1378 - Train Accuracy:  0.566, Validation Accuracy:  0.604, Loss:  0.975
    Epoch   0 Batch  259/1378 - Train Accuracy:  0.601, Validation Accuracy:  0.617, Loss:  0.947
    Epoch   0 Batch  260/1378 - Train Accuracy:  0.600, Validation Accuracy:  0.634, Loss:  0.927
    Epoch   0 Batch  261/1378 - Train Accuracy:  0.611, Validation Accuracy:  0.637, Loss:  0.934
    Epoch   0 Batch  262/1378 - Train Accuracy:  0.607, Validation Accuracy:  0.614, Loss:  0.942
    Epoch   0 Batch  263/1378 - Train Accuracy:  0.587, Validation Accuracy:  0.616, Loss:  0.926
    Epoch   0 Batch  264/1378 - Train Accuracy:  0.611, Validation Accuracy:  0.587, Loss:  0.903
    Epoch   0 Batch  265/1378 - Train Accuracy:  0.533, Validation Accuracy:  0.610, Loss:  0.999
    Epoch   0 Batch  266/1378 - Train Accuracy:  0.593, Validation Accuracy:  0.601, Loss:  0.925
    Epoch   0 Batch  267/1378 - Train Accuracy:  0.595, Validation Accuracy:  0.615, Loss:  0.980
    Epoch   0 Batch  268/1378 - Train Accuracy:  0.601, Validation Accuracy:  0.615, Loss:  0.903
    Epoch   0 Batch  269/1378 - Train Accuracy:  0.595, Validation Accuracy:  0.610, Loss:  0.966
    Epoch   0 Batch  270/1378 - Train Accuracy:  0.610, Validation Accuracy:  0.613, Loss:  0.958
    Epoch   0 Batch  271/1378 - Train Accuracy:  0.573, Validation Accuracy:  0.599, Loss:  1.027
    Epoch   0 Batch  272/1378 - Train Accuracy:  0.605, Validation Accuracy:  0.590, Loss:  0.896
    Epoch   0 Batch  273/1378 - Train Accuracy:  0.577, Validation Accuracy:  0.605, Loss:  0.959
    Epoch   0 Batch  274/1378 - Train Accuracy:  0.627, Validation Accuracy:  0.593, Loss:  0.914
    Epoch   0 Batch  275/1378 - Train Accuracy:  0.562, Validation Accuracy:  0.598, Loss:  0.959
    Epoch   0 Batch  276/1378 - Train Accuracy:  0.574, Validation Accuracy:  0.608, Loss:  1.010
    Epoch   0 Batch  277/1378 - Train Accuracy:  0.603, Validation Accuracy:  0.625, Loss:  0.927
    Epoch   0 Batch  278/1378 - Train Accuracy:  0.604, Validation Accuracy:  0.637, Loss:  0.962
    Epoch   0 Batch  279/1378 - Train Accuracy:  0.611, Validation Accuracy:  0.638, Loss:  0.985
    Epoch   0 Batch  280/1378 - Train Accuracy:  0.638, Validation Accuracy:  0.638, Loss:  0.977
    Epoch   0 Batch  281/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.617, Loss:  0.895
    Epoch   0 Batch  282/1378 - Train Accuracy:  0.609, Validation Accuracy:  0.634, Loss:  0.976
    Epoch   0 Batch  283/1378 - Train Accuracy:  0.634, Validation Accuracy:  0.632, Loss:  0.936
    Epoch   0 Batch  284/1378 - Train Accuracy:  0.577, Validation Accuracy:  0.634, Loss:  0.983
    Epoch   0 Batch  285/1378 - Train Accuracy:  0.595, Validation Accuracy:  0.602, Loss:  0.967
    Epoch   0 Batch  286/1378 - Train Accuracy:  0.515, Validation Accuracy:  0.580, Loss:  0.966
    Epoch   0 Batch  287/1378 - Train Accuracy:  0.574, Validation Accuracy:  0.613, Loss:  1.001
    Epoch   0 Batch  288/1378 - Train Accuracy:  0.656, Validation Accuracy:  0.626, Loss:  0.920
    Epoch   0 Batch  289/1378 - Train Accuracy:  0.568, Validation Accuracy:  0.627, Loss:  1.003
    Epoch   0 Batch  290/1378 - Train Accuracy:  0.602, Validation Accuracy:  0.623, Loss:  0.975
    Epoch   0 Batch  291/1378 - Train Accuracy:  0.548, Validation Accuracy:  0.612, Loss:  0.990
    Epoch   0 Batch  292/1378 - Train Accuracy:  0.617, Validation Accuracy:  0.618, Loss:  0.928
    Epoch   0 Batch  293/1378 - Train Accuracy:  0.641, Validation Accuracy:  0.609, Loss:  0.939
    Epoch   0 Batch  294/1378 - Train Accuracy:  0.602, Validation Accuracy:  0.610, Loss:  0.947
    Epoch   0 Batch  295/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.606, Loss:  0.935
    Epoch   0 Batch  296/1378 - Train Accuracy:  0.566, Validation Accuracy:  0.618, Loss:  0.962
    Epoch   0 Batch  297/1378 - Train Accuracy:  0.559, Validation Accuracy:  0.604, Loss:  1.004
    Epoch   0 Batch  298/1378 - Train Accuracy:  0.583, Validation Accuracy:  0.613, Loss:  0.977
    Epoch   0 Batch  299/1378 - Train Accuracy:  0.607, Validation Accuracy:  0.639, Loss:  0.936
    Epoch   0 Batch  300/1378 - Train Accuracy:  0.622, Validation Accuracy:  0.664, Loss:  0.946
    Epoch   0 Batch  301/1378 - Train Accuracy:  0.615, Validation Accuracy:  0.630, Loss:  0.959
    Epoch   0 Batch  302/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.618, Loss:  0.922
    Epoch   0 Batch  303/1378 - Train Accuracy:  0.555, Validation Accuracy:  0.630, Loss:  0.969
    Epoch   0 Batch  304/1378 - Train Accuracy:  0.616, Validation Accuracy:  0.594, Loss:  0.897
    Epoch   0 Batch  305/1378 - Train Accuracy:  0.624, Validation Accuracy:  0.603, Loss:  0.988
    Epoch   0 Batch  306/1378 - Train Accuracy:  0.611, Validation Accuracy:  0.642, Loss:  0.905
    Epoch   0 Batch  307/1378 - Train Accuracy:  0.649, Validation Accuracy:  0.638, Loss:  0.886
    Epoch   0 Batch  308/1378 - Train Accuracy:  0.629, Validation Accuracy:  0.633, Loss:  0.948
    Epoch   0 Batch  309/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.636, Loss:  0.898
    Epoch   0 Batch  310/1378 - Train Accuracy:  0.606, Validation Accuracy:  0.630, Loss:  0.966
    Epoch   0 Batch  311/1378 - Train Accuracy:  0.592, Validation Accuracy:  0.636, Loss:  0.965
    Epoch   0 Batch  312/1378 - Train Accuracy:  0.577, Validation Accuracy:  0.583, Loss:  0.915
    Epoch   0 Batch  313/1378 - Train Accuracy:  0.582, Validation Accuracy:  0.570, Loss:  0.911
    Epoch   0 Batch  314/1378 - Train Accuracy:  0.573, Validation Accuracy:  0.558, Loss:  0.975
    Epoch   0 Batch  315/1378 - Train Accuracy:  0.559, Validation Accuracy:  0.550, Loss:  0.920
    Epoch   0 Batch  316/1378 - Train Accuracy:  0.567, Validation Accuracy:  0.547, Loss:  0.980
    Epoch   0 Batch  317/1378 - Train Accuracy:  0.612, Validation Accuracy:  0.582, Loss:  0.947
    Epoch   0 Batch  318/1378 - Train Accuracy:  0.616, Validation Accuracy:  0.613, Loss:  0.959
    Epoch   0 Batch  319/1378 - Train Accuracy:  0.601, Validation Accuracy:  0.620, Loss:  0.934
    Epoch   0 Batch  320/1378 - Train Accuracy:  0.552, Validation Accuracy:  0.626, Loss:  0.991
    Epoch   0 Batch  321/1378 - Train Accuracy:  0.645, Validation Accuracy:  0.632, Loss:  0.907
    Epoch   0 Batch  322/1378 - Train Accuracy:  0.595, Validation Accuracy:  0.630, Loss:  0.950
    Epoch   0 Batch  323/1378 - Train Accuracy:  0.586, Validation Accuracy:  0.590, Loss:  0.903
    Epoch   0 Batch  324/1378 - Train Accuracy:  0.576, Validation Accuracy:  0.557, Loss:  0.907
    Epoch   0 Batch  325/1378 - Train Accuracy:  0.524, Validation Accuracy:  0.556, Loss:  0.953
    Epoch   0 Batch  326/1378 - Train Accuracy:  0.586, Validation Accuracy:  0.571, Loss:  0.916
    Epoch   0 Batch  327/1378 - Train Accuracy:  0.562, Validation Accuracy:  0.610, Loss:  0.965
    Epoch   0 Batch  328/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.628, Loss:  0.970
    Epoch   0 Batch  329/1378 - Train Accuracy:  0.609, Validation Accuracy:  0.627, Loss:  0.961
    Epoch   0 Batch  330/1378 - Train Accuracy:  0.602, Validation Accuracy:  0.616, Loss:  0.908
    Epoch   0 Batch  331/1378 - Train Accuracy:  0.638, Validation Accuracy:  0.619, Loss:  0.897
    Epoch   0 Batch  332/1378 - Train Accuracy:  0.575, Validation Accuracy:  0.623, Loss:  0.965
    Epoch   0 Batch  333/1378 - Train Accuracy:  0.607, Validation Accuracy:  0.599, Loss:  0.904
    Epoch   0 Batch  334/1378 - Train Accuracy:  0.600, Validation Accuracy:  0.567, Loss:  0.917
    Epoch   0 Batch  335/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.570, Loss:  0.863
    Epoch   0 Batch  336/1378 - Train Accuracy:  0.622, Validation Accuracy:  0.595, Loss:  0.915
    Epoch   0 Batch  337/1378 - Train Accuracy:  0.642, Validation Accuracy:  0.647, Loss:  0.946
    Epoch   0 Batch  338/1378 - Train Accuracy:  0.658, Validation Accuracy:  0.640, Loss:  0.933
    Epoch   0 Batch  339/1378 - Train Accuracy:  0.544, Validation Accuracy:  0.623, Loss:  0.967
    Epoch   0 Batch  340/1378 - Train Accuracy:  0.614, Validation Accuracy:  0.620, Loss:  0.911
    Epoch   0 Batch  341/1378 - Train Accuracy:  0.631, Validation Accuracy:  0.626, Loss:  0.923
    Epoch   0 Batch  342/1378 - Train Accuracy:  0.607, Validation Accuracy:  0.618, Loss:  0.944
    Epoch   0 Batch  343/1378 - Train Accuracy:  0.615, Validation Accuracy:  0.637, Loss:  0.894
    Epoch   0 Batch  344/1378 - Train Accuracy:  0.567, Validation Accuracy:  0.605, Loss:  0.966
    Epoch   0 Batch  345/1378 - Train Accuracy:  0.556, Validation Accuracy:  0.629, Loss:  1.000
    Epoch   0 Batch  346/1378 - Train Accuracy:  0.631, Validation Accuracy:  0.634, Loss:  0.883
    Epoch   0 Batch  347/1378 - Train Accuracy:  0.588, Validation Accuracy:  0.639, Loss:  0.886
    Epoch   0 Batch  348/1378 - Train Accuracy:  0.621, Validation Accuracy:  0.614, Loss:  0.929
    Epoch   0 Batch  349/1378 - Train Accuracy:  0.590, Validation Accuracy:  0.620, Loss:  0.901
    Epoch   0 Batch  350/1378 - Train Accuracy:  0.609, Validation Accuracy:  0.649, Loss:  0.895
    Epoch   0 Batch  351/1378 - Train Accuracy:  0.627, Validation Accuracy:  0.657, Loss:  0.939
    Epoch   0 Batch  352/1378 - Train Accuracy:  0.613, Validation Accuracy:  0.652, Loss:  0.934
    Epoch   0 Batch  353/1378 - Train Accuracy:  0.590, Validation Accuracy:  0.612, Loss:  0.875
    Epoch   0 Batch  354/1378 - Train Accuracy:  0.573, Validation Accuracy:  0.607, Loss:  0.903
    Epoch   0 Batch  355/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.605, Loss:  0.881
    Epoch   0 Batch  356/1378 - Train Accuracy:  0.570, Validation Accuracy:  0.596, Loss:  0.942
    Epoch   0 Batch  357/1378 - Train Accuracy:  0.537, Validation Accuracy:  0.595, Loss:  0.987
    Epoch   0 Batch  358/1378 - Train Accuracy:  0.616, Validation Accuracy:  0.615, Loss:  0.954
    Epoch   0 Batch  359/1378 - Train Accuracy:  0.649, Validation Accuracy:  0.642, Loss:  0.915
    Epoch   0 Batch  360/1378 - Train Accuracy:  0.600, Validation Accuracy:  0.643, Loss:  0.933
    Epoch   0 Batch  361/1378 - Train Accuracy:  0.608, Validation Accuracy:  0.636, Loss:  0.915
    Epoch   0 Batch  362/1378 - Train Accuracy:  0.637, Validation Accuracy:  0.626, Loss:  0.916
    Epoch   0 Batch  363/1378 - Train Accuracy:  0.613, Validation Accuracy:  0.628, Loss:  0.904
    Epoch   0 Batch  364/1378 - Train Accuracy:  0.584, Validation Accuracy:  0.623, Loss:  0.919
    Epoch   0 Batch  365/1378 - Train Accuracy:  0.611, Validation Accuracy:  0.621, Loss:  0.919
    Epoch   0 Batch  366/1378 - Train Accuracy:  0.630, Validation Accuracy:  0.630, Loss:  0.934
    Epoch   0 Batch  367/1378 - Train Accuracy:  0.637, Validation Accuracy:  0.652, Loss:  0.934
    Epoch   0 Batch  368/1378 - Train Accuracy:  0.627, Validation Accuracy:  0.651, Loss:  0.930
    Epoch   0 Batch  369/1378 - Train Accuracy:  0.590, Validation Accuracy:  0.655, Loss:  0.981
    Epoch   0 Batch  370/1378 - Train Accuracy:  0.653, Validation Accuracy:  0.652, Loss:  0.881
    Epoch   0 Batch  371/1378 - Train Accuracy:  0.645, Validation Accuracy:  0.655, Loss:  0.952
    Epoch   0 Batch  372/1378 - Train Accuracy:  0.592, Validation Accuracy:  0.647, Loss:  0.938
    Epoch   0 Batch  373/1378 - Train Accuracy:  0.578, Validation Accuracy:  0.622, Loss:  0.905
    Epoch   0 Batch  374/1378 - Train Accuracy:  0.546, Validation Accuracy:  0.566, Loss:  0.911
    Epoch   0 Batch  375/1378 - Train Accuracy:  0.540, Validation Accuracy:  0.561, Loss:  0.884
    Epoch   0 Batch  376/1378 - Train Accuracy:  0.543, Validation Accuracy:  0.567, Loss:  0.910
    Epoch   0 Batch  377/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.626, Loss:  0.874
    Epoch   0 Batch  378/1378 - Train Accuracy:  0.665, Validation Accuracy:  0.647, Loss:  0.973
    Epoch   0 Batch  379/1378 - Train Accuracy:  0.647, Validation Accuracy:  0.651, Loss:  0.869
    Epoch   0 Batch  380/1378 - Train Accuracy:  0.605, Validation Accuracy:  0.641, Loss:  0.916
    Epoch   0 Batch  381/1378 - Train Accuracy:  0.601, Validation Accuracy:  0.651, Loss:  0.941
    Epoch   0 Batch  382/1378 - Train Accuracy:  0.604, Validation Accuracy:  0.646, Loss:  0.865
    Epoch   0 Batch  383/1378 - Train Accuracy:  0.621, Validation Accuracy:  0.625, Loss:  0.886
    Epoch   0 Batch  384/1378 - Train Accuracy:  0.609, Validation Accuracy:  0.613, Loss:  0.903
    Epoch   0 Batch  385/1378 - Train Accuracy:  0.528, Validation Accuracy:  0.530, Loss:  0.910
    Epoch   0 Batch  386/1378 - Train Accuracy:  0.561, Validation Accuracy:  0.528, Loss:  0.919
    Epoch   0 Batch  387/1378 - Train Accuracy:  0.588, Validation Accuracy:  0.565, Loss:  0.897
    Epoch   0 Batch  388/1378 - Train Accuracy:  0.609, Validation Accuracy:  0.623, Loss:  0.917
    Epoch   0 Batch  389/1378 - Train Accuracy:  0.581, Validation Accuracy:  0.635, Loss:  0.976
    Epoch   0 Batch  390/1378 - Train Accuracy:  0.641, Validation Accuracy:  0.630, Loss:  0.849
    Epoch   0 Batch  391/1378 - Train Accuracy:  0.657, Validation Accuracy:  0.640, Loss:  0.777
    Epoch   0 Batch  392/1378 - Train Accuracy:  0.610, Validation Accuracy:  0.655, Loss:  0.949
    Epoch   0 Batch  393/1378 - Train Accuracy:  0.648, Validation Accuracy:  0.659, Loss:  0.835
    Epoch   0 Batch  394/1378 - Train Accuracy:  0.594, Validation Accuracy:  0.660, Loss:  0.908
    Epoch   0 Batch  395/1378 - Train Accuracy:  0.629, Validation Accuracy:  0.659, Loss:  0.911
    Epoch   0 Batch  396/1378 - Train Accuracy:  0.638, Validation Accuracy:  0.649, Loss:  0.877
    Epoch   0 Batch  397/1378 - Train Accuracy:  0.607, Validation Accuracy:  0.640, Loss:  0.921
    Epoch   0 Batch  398/1378 - Train Accuracy:  0.572, Validation Accuracy:  0.622, Loss:  0.896
    Epoch   0 Batch  399/1378 - Train Accuracy:  0.662, Validation Accuracy:  0.640, Loss:  0.854
    Epoch   0 Batch  400/1378 - Train Accuracy:  0.626, Validation Accuracy:  0.654, Loss:  0.957
    Epoch   0 Batch  401/1378 - Train Accuracy:  0.619, Validation Accuracy:  0.664, Loss:  0.848
    Epoch   0 Batch  402/1378 - Train Accuracy:  0.651, Validation Accuracy:  0.658, Loss:  0.869
    Epoch   0 Batch  403/1378 - Train Accuracy:  0.677, Validation Accuracy:  0.655, Loss:  0.831
    Epoch   0 Batch  404/1378 - Train Accuracy:  0.625, Validation Accuracy:  0.655, Loss:  0.843
    Epoch   0 Batch  405/1378 - Train Accuracy:  0.650, Validation Accuracy:  0.645, Loss:  0.812
    Epoch   0 Batch  406/1378 - Train Accuracy:  0.648, Validation Accuracy:  0.648, Loss:  0.913
    Epoch   0 Batch  407/1378 - Train Accuracy:  0.635, Validation Accuracy:  0.647, Loss:  0.867
    Epoch   0 Batch  408/1378 - Train Accuracy:  0.656, Validation Accuracy:  0.646, Loss:  0.840
    Epoch   0 Batch  409/1378 - Train Accuracy:  0.616, Validation Accuracy:  0.636, Loss:  0.890
    Epoch   0 Batch  410/1378 - Train Accuracy:  0.688, Validation Accuracy:  0.646, Loss:  0.878
    Epoch   0 Batch  411/1378 - Train Accuracy:  0.641, Validation Accuracy:  0.648, Loss:  0.872
    Epoch   0 Batch  412/1378 - Train Accuracy:  0.610, Validation Accuracy:  0.647, Loss:  0.875
    Epoch   0 Batch  413/1378 - Train Accuracy:  0.622, Validation Accuracy:  0.637, Loss:  0.849
    Epoch   0 Batch  414/1378 - Train Accuracy:  0.664, Validation Accuracy:  0.633, Loss:  0.881
    Epoch   0 Batch  415/1378 - Train Accuracy:  0.649, Validation Accuracy:  0.626, Loss:  0.869
    Epoch   0 Batch  416/1378 - Train Accuracy:  0.620, Validation Accuracy:  0.619, Loss:  0.896
    Epoch   0 Batch  417/1378 - Train Accuracy:  0.608, Validation Accuracy:  0.621, Loss:  0.922
    Epoch   0 Batch  418/1378 - Train Accuracy:  0.636, Validation Accuracy:  0.629, Loss:  0.846
    Epoch   0 Batch  419/1378 - Train Accuracy:  0.626, Validation Accuracy:  0.610, Loss:  0.905
    Epoch   0 Batch  420/1378 - Train Accuracy:  0.665, Validation Accuracy:  0.642, Loss:  0.908
    Epoch   0 Batch  421/1378 - Train Accuracy:  0.659, Validation Accuracy:  0.659, Loss:  0.838
    Epoch   0 Batch  422/1378 - Train Accuracy:  0.660, Validation Accuracy:  0.657, Loss:  0.817
    Epoch   0 Batch  423/1378 - Train Accuracy:  0.634, Validation Accuracy:  0.658, Loss:  0.841
    Epoch   0 Batch  424/1378 - Train Accuracy:  0.617, Validation Accuracy:  0.655, Loss:  0.888
    Epoch   0 Batch  425/1378 - Train Accuracy:  0.602, Validation Accuracy:  0.654, Loss:  0.804
    Epoch   0 Batch  426/1378 - Train Accuracy:  0.620, Validation Accuracy:  0.635, Loss:  0.818
    Epoch   0 Batch  427/1378 - Train Accuracy:  0.651, Validation Accuracy:  0.657, Loss:  0.898
    Epoch   0 Batch  428/1378 - Train Accuracy:  0.624, Validation Accuracy:  0.668, Loss:  0.842
    Epoch   0 Batch  429/1378 - Train Accuracy:  0.672, Validation Accuracy:  0.670, Loss:  0.871
    Epoch   0 Batch  430/1378 - Train Accuracy:  0.665, Validation Accuracy:  0.657, Loss:  0.862
    Epoch   0 Batch  431/1378 - Train Accuracy:  0.655, Validation Accuracy:  0.668, Loss:  0.838
    Epoch   0 Batch  432/1378 - Train Accuracy:  0.614, Validation Accuracy:  0.663, Loss:  0.924
    Epoch   0 Batch  433/1378 - Train Accuracy:  0.675, Validation Accuracy:  0.655, Loss:  0.915
    Epoch   0 Batch  434/1378 - Train Accuracy:  0.661, Validation Accuracy:  0.655, Loss:  0.859
    Epoch   0 Batch  435/1378 - Train Accuracy:  0.624, Validation Accuracy:  0.642, Loss:  0.835
    Epoch   0 Batch  436/1378 - Train Accuracy:  0.653, Validation Accuracy:  0.654, Loss:  0.828
    Epoch   0 Batch  437/1378 - Train Accuracy:  0.646, Validation Accuracy:  0.650, Loss:  0.917
    Epoch   0 Batch  438/1378 - Train Accuracy:  0.640, Validation Accuracy:  0.657, Loss:  0.832
    Epoch   0 Batch  439/1378 - Train Accuracy:  0.669, Validation Accuracy:  0.668, Loss:  0.804
    Epoch   0 Batch  440/1378 - Train Accuracy:  0.659, Validation Accuracy:  0.644, Loss:  0.871
    Epoch   0 Batch  441/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.635, Loss:  0.852
    Epoch   0 Batch  442/1378 - Train Accuracy:  0.661, Validation Accuracy:  0.647, Loss:  0.833
    Epoch   0 Batch  443/1378 - Train Accuracy:  0.660, Validation Accuracy:  0.660, Loss:  0.819
    Epoch   0 Batch  444/1378 - Train Accuracy:  0.626, Validation Accuracy:  0.660, Loss:  0.852
    Epoch   0 Batch  445/1378 - Train Accuracy:  0.736, Validation Accuracy:  0.651, Loss:  0.818
    Epoch   0 Batch  446/1378 - Train Accuracy:  0.581, Validation Accuracy:  0.659, Loss:  0.902
    Epoch   0 Batch  447/1378 - Train Accuracy:  0.602, Validation Accuracy:  0.655, Loss:  0.854
    Epoch   0 Batch  448/1378 - Train Accuracy:  0.678, Validation Accuracy:  0.653, Loss:  0.865
    Epoch   0 Batch  449/1378 - Train Accuracy:  0.658, Validation Accuracy:  0.677, Loss:  0.848
    Epoch   0 Batch  450/1378 - Train Accuracy:  0.639, Validation Accuracy:  0.674, Loss:  0.862
    Epoch   0 Batch  451/1378 - Train Accuracy:  0.616, Validation Accuracy:  0.682, Loss:  0.835
    Epoch   0 Batch  452/1378 - Train Accuracy:  0.587, Validation Accuracy:  0.703, Loss:  0.840
    Epoch   0 Batch  453/1378 - Train Accuracy:  0.638, Validation Accuracy:  0.674, Loss:  0.849
    Epoch   0 Batch  454/1378 - Train Accuracy:  0.654, Validation Accuracy:  0.669, Loss:  0.815
    Epoch   0 Batch  455/1378 - Train Accuracy:  0.670, Validation Accuracy:  0.658, Loss:  0.852
    Epoch   0 Batch  456/1378 - Train Accuracy:  0.670, Validation Accuracy:  0.655, Loss:  0.812
    Epoch   0 Batch  457/1378 - Train Accuracy:  0.643, Validation Accuracy:  0.651, Loss:  0.820
    Epoch   0 Batch  458/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.658, Loss:  0.850
    Epoch   0 Batch  459/1378 - Train Accuracy:  0.664, Validation Accuracy:  0.649, Loss:  0.725
    Epoch   0 Batch  460/1378 - Train Accuracy:  0.630, Validation Accuracy:  0.650, Loss:  0.847
    Epoch   0 Batch  461/1378 - Train Accuracy:  0.616, Validation Accuracy:  0.655, Loss:  0.783
    Epoch   0 Batch  462/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.664, Loss:  0.848
    Epoch   0 Batch  463/1378 - Train Accuracy:  0.684, Validation Accuracy:  0.666, Loss:  0.823
    Epoch   0 Batch  464/1378 - Train Accuracy:  0.645, Validation Accuracy:  0.655, Loss:  0.811
    Epoch   0 Batch  465/1378 - Train Accuracy:  0.607, Validation Accuracy:  0.653, Loss:  0.821
    Epoch   0 Batch  466/1378 - Train Accuracy:  0.666, Validation Accuracy:  0.664, Loss:  0.866
    Epoch   0 Batch  467/1378 - Train Accuracy:  0.625, Validation Accuracy:  0.654, Loss:  0.875
    Epoch   0 Batch  468/1378 - Train Accuracy:  0.604, Validation Accuracy:  0.642, Loss:  0.827
    Epoch   0 Batch  469/1378 - Train Accuracy:  0.645, Validation Accuracy:  0.644, Loss:  0.792
    Epoch   0 Batch  470/1378 - Train Accuracy:  0.714, Validation Accuracy:  0.635, Loss:  0.790
    Epoch   0 Batch  471/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.656, Loss:  0.842
    Epoch   0 Batch  472/1378 - Train Accuracy:  0.638, Validation Accuracy:  0.674, Loss:  0.792
    Epoch   0 Batch  473/1378 - Train Accuracy:  0.658, Validation Accuracy:  0.682, Loss:  0.825
    Epoch   0 Batch  474/1378 - Train Accuracy:  0.642, Validation Accuracy:  0.699, Loss:  0.839
    Epoch   0 Batch  475/1378 - Train Accuracy:  0.670, Validation Accuracy:  0.687, Loss:  0.756
    Epoch   0 Batch  476/1378 - Train Accuracy:  0.693, Validation Accuracy:  0.685, Loss:  0.839
    Epoch   0 Batch  477/1378 - Train Accuracy:  0.612, Validation Accuracy:  0.652, Loss:  0.782
    Epoch   0 Batch  478/1378 - Train Accuracy:  0.680, Validation Accuracy:  0.637, Loss:  0.780
    Epoch   0 Batch  479/1378 - Train Accuracy:  0.615, Validation Accuracy:  0.647, Loss:  0.817
    Epoch   0 Batch  480/1378 - Train Accuracy:  0.692, Validation Accuracy:  0.665, Loss:  0.809
    Epoch   0 Batch  481/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.677, Loss:  0.804
    Epoch   0 Batch  482/1378 - Train Accuracy:  0.668, Validation Accuracy:  0.679, Loss:  0.805
    Epoch   0 Batch  483/1378 - Train Accuracy:  0.648, Validation Accuracy:  0.666, Loss:  0.772
    Epoch   0 Batch  484/1378 - Train Accuracy:  0.678, Validation Accuracy:  0.668, Loss:  0.779
    Epoch   0 Batch  485/1378 - Train Accuracy:  0.599, Validation Accuracy:  0.664, Loss:  0.821
    Epoch   0 Batch  486/1378 - Train Accuracy:  0.686, Validation Accuracy:  0.664, Loss:  0.772
    Epoch   0 Batch  487/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.664, Loss:  0.809
    Epoch   0 Batch  488/1378 - Train Accuracy:  0.658, Validation Accuracy:  0.648, Loss:  0.820
    Epoch   0 Batch  489/1378 - Train Accuracy:  0.644, Validation Accuracy:  0.644, Loss:  0.835
    Epoch   0 Batch  490/1378 - Train Accuracy:  0.620, Validation Accuracy:  0.642, Loss:  0.807
    Epoch   0 Batch  491/1378 - Train Accuracy:  0.637, Validation Accuracy:  0.629, Loss:  0.774
    Epoch   0 Batch  492/1378 - Train Accuracy:  0.671, Validation Accuracy:  0.644, Loss:  0.841
    Epoch   0 Batch  493/1378 - Train Accuracy:  0.672, Validation Accuracy:  0.656, Loss:  0.778
    Epoch   0 Batch  494/1378 - Train Accuracy:  0.650, Validation Accuracy:  0.688, Loss:  0.782
    Epoch   0 Batch  495/1378 - Train Accuracy:  0.693, Validation Accuracy:  0.682, Loss:  0.789
    Epoch   0 Batch  496/1378 - Train Accuracy:  0.707, Validation Accuracy:  0.667, Loss:  0.826
    Epoch   0 Batch  497/1378 - Train Accuracy:  0.661, Validation Accuracy:  0.660, Loss:  0.780
    Epoch   0 Batch  498/1378 - Train Accuracy:  0.676, Validation Accuracy:  0.669, Loss:  0.820
    Epoch   0 Batch  499/1378 - Train Accuracy:  0.653, Validation Accuracy:  0.672, Loss:  0.817
    Epoch   0 Batch  500/1378 - Train Accuracy:  0.643, Validation Accuracy:  0.665, Loss:  0.786
    Epoch   0 Batch  501/1378 - Train Accuracy:  0.690, Validation Accuracy:  0.675, Loss:  0.777
    Epoch   0 Batch  502/1378 - Train Accuracy:  0.668, Validation Accuracy:  0.672, Loss:  0.807
    Epoch   0 Batch  503/1378 - Train Accuracy:  0.653, Validation Accuracy:  0.682, Loss:  0.805
    Epoch   0 Batch  504/1378 - Train Accuracy:  0.690, Validation Accuracy:  0.677, Loss:  0.773
    Epoch   0 Batch  505/1378 - Train Accuracy:  0.667, Validation Accuracy:  0.682, Loss:  0.822
    Epoch   0 Batch  506/1378 - Train Accuracy:  0.649, Validation Accuracy:  0.685, Loss:  0.763
    Epoch   0 Batch  507/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.672, Loss:  0.809
    Epoch   0 Batch  508/1378 - Train Accuracy:  0.673, Validation Accuracy:  0.690, Loss:  0.798
    Epoch   0 Batch  509/1378 - Train Accuracy:  0.675, Validation Accuracy:  0.709, Loss:  0.750
    Epoch   0 Batch  510/1378 - Train Accuracy:  0.659, Validation Accuracy:  0.701, Loss:  0.775
    Epoch   0 Batch  511/1378 - Train Accuracy:  0.624, Validation Accuracy:  0.703, Loss:  0.811
    Epoch   0 Batch  512/1378 - Train Accuracy:  0.676, Validation Accuracy:  0.705, Loss:  0.767
    Epoch   0 Batch  513/1378 - Train Accuracy:  0.705, Validation Accuracy:  0.660, Loss:  0.792
    Epoch   0 Batch  514/1378 - Train Accuracy:  0.624, Validation Accuracy:  0.640, Loss:  0.737
    Epoch   0 Batch  515/1378 - Train Accuracy:  0.696, Validation Accuracy:  0.659, Loss:  0.748
    Epoch   0 Batch  516/1378 - Train Accuracy:  0.641, Validation Accuracy:  0.677, Loss:  0.762
    Epoch   0 Batch  517/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.700, Loss:  0.757
    Epoch   0 Batch  518/1378 - Train Accuracy:  0.703, Validation Accuracy:  0.697, Loss:  0.816
    Epoch   0 Batch  519/1378 - Train Accuracy:  0.681, Validation Accuracy:  0.691, Loss:  0.785
    Epoch   0 Batch  520/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.687, Loss:  0.784
    Epoch   0 Batch  521/1378 - Train Accuracy:  0.679, Validation Accuracy:  0.686, Loss:  0.757
    Epoch   0 Batch  522/1378 - Train Accuracy:  0.690, Validation Accuracy:  0.694, Loss:  0.756
    Epoch   0 Batch  523/1378 - Train Accuracy:  0.646, Validation Accuracy:  0.713, Loss:  0.727
    Epoch   0 Batch  524/1378 - Train Accuracy:  0.625, Validation Accuracy:  0.687, Loss:  0.786
    Epoch   0 Batch  525/1378 - Train Accuracy:  0.583, Validation Accuracy:  0.607, Loss:  0.791
    Epoch   0 Batch  526/1378 - Train Accuracy:  0.622, Validation Accuracy:  0.602, Loss:  0.773
    Epoch   0 Batch  527/1378 - Train Accuracy:  0.622, Validation Accuracy:  0.615, Loss:  0.817
    Epoch   0 Batch  528/1378 - Train Accuracy:  0.628, Validation Accuracy:  0.701, Loss:  0.802
    Epoch   0 Batch  529/1378 - Train Accuracy:  0.691, Validation Accuracy:  0.711, Loss:  0.764
    Epoch   0 Batch  530/1378 - Train Accuracy:  0.669, Validation Accuracy:  0.702, Loss:  0.755
    Epoch   0 Batch  531/1378 - Train Accuracy:  0.678, Validation Accuracy:  0.707, Loss:  0.778
    Epoch   0 Batch  532/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.713, Loss:  0.727
    Epoch   0 Batch  533/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.734, Loss:  0.745
    Epoch   0 Batch  534/1378 - Train Accuracy:  0.638, Validation Accuracy:  0.741, Loss:  0.760
    Epoch   0 Batch  535/1378 - Train Accuracy:  0.670, Validation Accuracy:  0.715, Loss:  0.813
    Epoch   0 Batch  536/1378 - Train Accuracy:  0.641, Validation Accuracy:  0.678, Loss:  0.740
    Epoch   0 Batch  537/1378 - Train Accuracy:  0.637, Validation Accuracy:  0.645, Loss:  0.761
    Epoch   0 Batch  538/1378 - Train Accuracy:  0.688, Validation Accuracy:  0.623, Loss:  0.699
    Epoch   0 Batch  539/1378 - Train Accuracy:  0.596, Validation Accuracy:  0.669, Loss:  0.810
    Epoch   0 Batch  540/1378 - Train Accuracy:  0.679, Validation Accuracy:  0.702, Loss:  0.758
    Epoch   0 Batch  541/1378 - Train Accuracy:  0.702, Validation Accuracy:  0.709, Loss:  0.726
    Epoch   0 Batch  542/1378 - Train Accuracy:  0.689, Validation Accuracy:  0.710, Loss:  0.781
    Epoch   0 Batch  543/1378 - Train Accuracy:  0.660, Validation Accuracy:  0.684, Loss:  0.805
    Epoch   0 Batch  544/1378 - Train Accuracy:  0.690, Validation Accuracy:  0.672, Loss:  0.752
    Epoch   0 Batch  545/1378 - Train Accuracy:  0.694, Validation Accuracy:  0.676, Loss:  0.776
    Epoch   0 Batch  546/1378 - Train Accuracy:  0.625, Validation Accuracy:  0.631, Loss:  0.797
    Epoch   0 Batch  547/1378 - Train Accuracy:  0.660, Validation Accuracy:  0.613, Loss:  0.773
    Epoch   0 Batch  548/1378 - Train Accuracy:  0.656, Validation Accuracy:  0.610, Loss:  0.764
    Epoch   0 Batch  549/1378 - Train Accuracy:  0.691, Validation Accuracy:  0.648, Loss:  0.717
    Epoch   0 Batch  550/1378 - Train Accuracy:  0.744, Validation Accuracy:  0.689, Loss:  0.727
    Epoch   0 Batch  551/1378 - Train Accuracy:  0.677, Validation Accuracy:  0.726, Loss:  0.718
    Epoch   0 Batch  552/1378 - Train Accuracy:  0.712, Validation Accuracy:  0.720, Loss:  0.735
    Epoch   0 Batch  553/1378 - Train Accuracy:  0.678, Validation Accuracy:  0.723, Loss:  0.780
    Epoch   0 Batch  554/1378 - Train Accuracy:  0.740, Validation Accuracy:  0.715, Loss:  0.726
    Epoch   0 Batch  555/1378 - Train Accuracy:  0.681, Validation Accuracy:  0.709, Loss:  0.742
    Epoch   0 Batch  556/1378 - Train Accuracy:  0.699, Validation Accuracy:  0.699, Loss:  0.765
    Epoch   0 Batch  557/1378 - Train Accuracy:  0.685, Validation Accuracy:  0.683, Loss:  0.752
    Epoch   0 Batch  558/1378 - Train Accuracy:  0.675, Validation Accuracy:  0.697, Loss:  0.734
    Epoch   0 Batch  559/1378 - Train Accuracy:  0.690, Validation Accuracy:  0.704, Loss:  0.668
    Epoch   0 Batch  560/1378 - Train Accuracy:  0.685, Validation Accuracy:  0.710, Loss:  0.787
    Epoch   0 Batch  561/1378 - Train Accuracy:  0.713, Validation Accuracy:  0.705, Loss:  0.738
    Epoch   0 Batch  562/1378 - Train Accuracy:  0.694, Validation Accuracy:  0.704, Loss:  0.745
    Epoch   0 Batch  563/1378 - Train Accuracy:  0.667, Validation Accuracy:  0.693, Loss:  0.749
    Epoch   0 Batch  564/1378 - Train Accuracy:  0.652, Validation Accuracy:  0.696, Loss:  0.763
    Epoch   0 Batch  565/1378 - Train Accuracy:  0.687, Validation Accuracy:  0.697, Loss:  0.704
    Epoch   0 Batch  566/1378 - Train Accuracy:  0.679, Validation Accuracy:  0.705, Loss:  0.751
    Epoch   0 Batch  567/1378 - Train Accuracy:  0.699, Validation Accuracy:  0.640, Loss:  0.682
    Epoch   0 Batch  568/1378 - Train Accuracy:  0.632, Validation Accuracy:  0.627, Loss:  0.763
    Epoch   0 Batch  569/1378 - Train Accuracy:  0.651, Validation Accuracy:  0.640, Loss:  0.754
    Epoch   0 Batch  570/1378 - Train Accuracy:  0.674, Validation Accuracy:  0.678, Loss:  0.750
    Epoch   0 Batch  571/1378 - Train Accuracy:  0.713, Validation Accuracy:  0.687, Loss:  0.727
    Epoch   0 Batch  572/1378 - Train Accuracy:  0.706, Validation Accuracy:  0.675, Loss:  0.746
    Epoch   0 Batch  573/1378 - Train Accuracy:  0.691, Validation Accuracy:  0.675, Loss:  0.744
    Epoch   0 Batch  574/1378 - Train Accuracy:  0.709, Validation Accuracy:  0.694, Loss:  0.739
    Epoch   0 Batch  575/1378 - Train Accuracy:  0.655, Validation Accuracy:  0.691, Loss:  0.741
    Epoch   0 Batch  576/1378 - Train Accuracy:  0.707, Validation Accuracy:  0.684, Loss:  0.691
    Epoch   0 Batch  577/1378 - Train Accuracy:  0.678, Validation Accuracy:  0.649, Loss:  0.704
    Epoch   0 Batch  578/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.650, Loss:  0.697
    Epoch   0 Batch  579/1378 - Train Accuracy:  0.651, Validation Accuracy:  0.667, Loss:  0.709
    Epoch   0 Batch  580/1378 - Train Accuracy:  0.739, Validation Accuracy:  0.685, Loss:  0.711
    Epoch   0 Batch  581/1378 - Train Accuracy:  0.727, Validation Accuracy:  0.718, Loss:  0.699
    Epoch   0 Batch  582/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.735, Loss:  0.797
    Epoch   0 Batch  583/1378 - Train Accuracy:  0.725, Validation Accuracy:  0.734, Loss:  0.695
    Epoch   0 Batch  584/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.740, Loss:  0.750
    Epoch   0 Batch  585/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.734, Loss:  0.667
    Epoch   0 Batch  586/1378 - Train Accuracy:  0.659, Validation Accuracy:  0.728, Loss:  0.704
    Epoch   0 Batch  587/1378 - Train Accuracy:  0.649, Validation Accuracy:  0.696, Loss:  0.768
    Epoch   0 Batch  588/1378 - Train Accuracy:  0.691, Validation Accuracy:  0.666, Loss:  0.703
    Epoch   0 Batch  589/1378 - Train Accuracy:  0.654, Validation Accuracy:  0.655, Loss:  0.728
    Epoch   0 Batch  590/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.685, Loss:  0.764
    Epoch   0 Batch  591/1378 - Train Accuracy:  0.686, Validation Accuracy:  0.702, Loss:  0.694
    Epoch   0 Batch  592/1378 - Train Accuracy:  0.665, Validation Accuracy:  0.710, Loss:  0.697
    Epoch   0 Batch  593/1378 - Train Accuracy:  0.686, Validation Accuracy:  0.719, Loss:  0.661
    Epoch   0 Batch  594/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.713, Loss:  0.711
    Epoch   0 Batch  595/1378 - Train Accuracy:  0.717, Validation Accuracy:  0.701, Loss:  0.701
    Epoch   0 Batch  596/1378 - Train Accuracy:  0.725, Validation Accuracy:  0.691, Loss:  0.774
    Epoch   0 Batch  597/1378 - Train Accuracy:  0.666, Validation Accuracy:  0.688, Loss:  0.711
    Epoch   0 Batch  598/1378 - Train Accuracy:  0.688, Validation Accuracy:  0.661, Loss:  0.672
    Epoch   0 Batch  599/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.669, Loss:  0.708
    Epoch   0 Batch  600/1378 - Train Accuracy:  0.703, Validation Accuracy:  0.686, Loss:  0.729
    Epoch   0 Batch  601/1378 - Train Accuracy:  0.719, Validation Accuracy:  0.739, Loss:  0.712
    Epoch   0 Batch  602/1378 - Train Accuracy:  0.683, Validation Accuracy:  0.731, Loss:  0.764
    Epoch   0 Batch  603/1378 - Train Accuracy:  0.732, Validation Accuracy:  0.739, Loss:  0.715
    Epoch   0 Batch  604/1378 - Train Accuracy:  0.743, Validation Accuracy:  0.714, Loss:  0.732
    Epoch   0 Batch  605/1378 - Train Accuracy:  0.670, Validation Accuracy:  0.716, Loss:  0.702
    Epoch   0 Batch  606/1378 - Train Accuracy:  0.734, Validation Accuracy:  0.719, Loss:  0.717
    Epoch   0 Batch  607/1378 - Train Accuracy:  0.703, Validation Accuracy:  0.704, Loss:  0.720
    Epoch   0 Batch  608/1378 - Train Accuracy:  0.750, Validation Accuracy:  0.690, Loss:  0.651
    Epoch   0 Batch  609/1378 - Train Accuracy:  0.679, Validation Accuracy:  0.672, Loss:  0.677
    Epoch   0 Batch  610/1378 - Train Accuracy:  0.716, Validation Accuracy:  0.665, Loss:  0.719
    Epoch   0 Batch  611/1378 - Train Accuracy:  0.753, Validation Accuracy:  0.679, Loss:  0.676
    Epoch   0 Batch  612/1378 - Train Accuracy:  0.733, Validation Accuracy:  0.686, Loss:  0.729
    Epoch   0 Batch  613/1378 - Train Accuracy:  0.724, Validation Accuracy:  0.692, Loss:  0.711
    Epoch   0 Batch  614/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.713, Loss:  0.697
    Epoch   0 Batch  615/1378 - Train Accuracy:  0.746, Validation Accuracy:  0.715, Loss:  0.711
    Epoch   0 Batch  616/1378 - Train Accuracy:  0.697, Validation Accuracy:  0.714, Loss:  0.718
    Epoch   0 Batch  617/1378 - Train Accuracy:  0.698, Validation Accuracy:  0.711, Loss:  0.738
    Epoch   0 Batch  618/1378 - Train Accuracy:  0.725, Validation Accuracy:  0.712, Loss:  0.682
    Epoch   0 Batch  619/1378 - Train Accuracy:  0.658, Validation Accuracy:  0.708, Loss:  0.744
    Epoch   0 Batch  620/1378 - Train Accuracy:  0.734, Validation Accuracy:  0.713, Loss:  0.664
    Epoch   0 Batch  621/1378 - Train Accuracy:  0.745, Validation Accuracy:  0.715, Loss:  0.678
    Epoch   0 Batch  622/1378 - Train Accuracy:  0.755, Validation Accuracy:  0.719, Loss:  0.658
    Epoch   0 Batch  623/1378 - Train Accuracy:  0.707, Validation Accuracy:  0.701, Loss:  0.715
    Epoch   0 Batch  624/1378 - Train Accuracy:  0.688, Validation Accuracy:  0.704, Loss:  0.691
    Epoch   0 Batch  625/1378 - Train Accuracy:  0.726, Validation Accuracy:  0.702, Loss:  0.687
    Epoch   0 Batch  626/1378 - Train Accuracy:  0.688, Validation Accuracy:  0.686, Loss:  0.716
    Epoch   0 Batch  627/1378 - Train Accuracy:  0.699, Validation Accuracy:  0.686, Loss:  0.650
    Epoch   0 Batch  628/1378 - Train Accuracy:  0.682, Validation Accuracy:  0.662, Loss:  0.671
    Epoch   0 Batch  629/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.675, Loss:  0.705
    Epoch   0 Batch  630/1378 - Train Accuracy:  0.680, Validation Accuracy:  0.666, Loss:  0.687
    Epoch   0 Batch  631/1378 - Train Accuracy:  0.712, Validation Accuracy:  0.673, Loss:  0.662
    Epoch   0 Batch  632/1378 - Train Accuracy:  0.778, Validation Accuracy:  0.694, Loss:  0.647
    Epoch   0 Batch  633/1378 - Train Accuracy:  0.731, Validation Accuracy:  0.704, Loss:  0.666
    Epoch   0 Batch  634/1378 - Train Accuracy:  0.657, Validation Accuracy:  0.712, Loss:  0.779
    Epoch   0 Batch  635/1378 - Train Accuracy:  0.747, Validation Accuracy:  0.719, Loss:  0.686
    Epoch   0 Batch  636/1378 - Train Accuracy:  0.663, Validation Accuracy:  0.711, Loss:  0.717
    Epoch   0 Batch  637/1378 - Train Accuracy:  0.687, Validation Accuracy:  0.709, Loss:  0.684
    Epoch   0 Batch  638/1378 - Train Accuracy:  0.757, Validation Accuracy:  0.721, Loss:  0.637
    Epoch   0 Batch  639/1378 - Train Accuracy:  0.736, Validation Accuracy:  0.697, Loss:  0.668
    Epoch   0 Batch  640/1378 - Train Accuracy:  0.717, Validation Accuracy:  0.690, Loss:  0.655
    Epoch   0 Batch  641/1378 - Train Accuracy:  0.738, Validation Accuracy:  0.681, Loss:  0.687
    Epoch   0 Batch  642/1378 - Train Accuracy:  0.686, Validation Accuracy:  0.676, Loss:  0.673
    Epoch   0 Batch  643/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.692, Loss:  0.705
    Epoch   0 Batch  644/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.703, Loss:  0.681
    Epoch   0 Batch  645/1378 - Train Accuracy:  0.725, Validation Accuracy:  0.713, Loss:  0.691
    Epoch   0 Batch  646/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.736, Loss:  0.662
    Epoch   0 Batch  647/1378 - Train Accuracy:  0.729, Validation Accuracy:  0.738, Loss:  0.638
    Epoch   0 Batch  648/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.747, Loss:  0.684
    Epoch   0 Batch  649/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.749, Loss:  0.731
    Epoch   0 Batch  650/1378 - Train Accuracy:  0.680, Validation Accuracy:  0.692, Loss:  0.696
    Epoch   0 Batch  651/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.708, Loss:  0.681
    Epoch   0 Batch  652/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.703, Loss:  0.677
    Epoch   0 Batch  653/1378 - Train Accuracy:  0.714, Validation Accuracy:  0.706, Loss:  0.725
    Epoch   0 Batch  654/1378 - Train Accuracy:  0.729, Validation Accuracy:  0.692, Loss:  0.649
    Epoch   0 Batch  655/1378 - Train Accuracy:  0.717, Validation Accuracy:  0.700, Loss:  0.673
    Epoch   0 Batch  656/1378 - Train Accuracy:  0.725, Validation Accuracy:  0.689, Loss:  0.669
    Epoch   0 Batch  657/1378 - Train Accuracy:  0.706, Validation Accuracy:  0.711, Loss:  0.678
    Epoch   0 Batch  658/1378 - Train Accuracy:  0.699, Validation Accuracy:  0.712, Loss:  0.666
    Epoch   0 Batch  659/1378 - Train Accuracy:  0.726, Validation Accuracy:  0.702, Loss:  0.688
    Epoch   0 Batch  660/1378 - Train Accuracy:  0.719, Validation Accuracy:  0.706, Loss:  0.704
    Epoch   0 Batch  661/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.695, Loss:  0.631
    Epoch   0 Batch  662/1378 - Train Accuracy:  0.700, Validation Accuracy:  0.695, Loss:  0.692
    Epoch   0 Batch  663/1378 - Train Accuracy:  0.750, Validation Accuracy:  0.719, Loss:  0.669
    Epoch   0 Batch  664/1378 - Train Accuracy:  0.732, Validation Accuracy:  0.735, Loss:  0.672
    Epoch   0 Batch  665/1378 - Train Accuracy:  0.760, Validation Accuracy:  0.725, Loss:  0.653
    Epoch   0 Batch  666/1378 - Train Accuracy:  0.760, Validation Accuracy:  0.732, Loss:  0.685
    Epoch   0 Batch  667/1378 - Train Accuracy:  0.746, Validation Accuracy:  0.709, Loss:  0.614
    Epoch   0 Batch  668/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.691, Loss:  0.753
    Epoch   0 Batch  669/1378 - Train Accuracy:  0.735, Validation Accuracy:  0.653, Loss:  0.675
    Epoch   0 Batch  670/1378 - Train Accuracy:  0.715, Validation Accuracy:  0.663, Loss:  0.676
    Epoch   0 Batch  671/1378 - Train Accuracy:  0.752, Validation Accuracy:  0.667, Loss:  0.679
    Epoch   0 Batch  672/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.661, Loss:  0.672
    Epoch   0 Batch  673/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.665, Loss:  0.680
    Epoch   0 Batch  674/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.661, Loss:  0.644
    Epoch   0 Batch  675/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.694, Loss:  0.704
    Epoch   0 Batch  676/1378 - Train Accuracy:  0.758, Validation Accuracy:  0.708, Loss:  0.678
    Epoch   0 Batch  677/1378 - Train Accuracy:  0.727, Validation Accuracy:  0.710, Loss:  0.677
    Epoch   0 Batch  678/1378 - Train Accuracy:  0.694, Validation Accuracy:  0.700, Loss:  0.686
    Epoch   0 Batch  679/1378 - Train Accuracy:  0.722, Validation Accuracy:  0.705, Loss:  0.698
    Epoch   0 Batch  680/1378 - Train Accuracy:  0.740, Validation Accuracy:  0.698, Loss:  0.580
    Epoch   0 Batch  681/1378 - Train Accuracy:  0.705, Validation Accuracy:  0.685, Loss:  0.703
    Epoch   0 Batch  682/1378 - Train Accuracy:  0.694, Validation Accuracy:  0.693, Loss:  0.684
    Epoch   0 Batch  683/1378 - Train Accuracy:  0.701, Validation Accuracy:  0.702, Loss:  0.711
    Epoch   0 Batch  684/1378 - Train Accuracy:  0.715, Validation Accuracy:  0.698, Loss:  0.637
    Epoch   0 Batch  685/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.720, Loss:  0.702
    Epoch   0 Batch  686/1378 - Train Accuracy:  0.729, Validation Accuracy:  0.728, Loss:  0.647
    Epoch   0 Batch  687/1378 - Train Accuracy:  0.787, Validation Accuracy:  0.723, Loss:  0.626
    Epoch   0 Batch  688/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.718, Loss:  0.668
    Epoch   0 Batch  689/1378 - Train Accuracy:  0.784, Validation Accuracy:  0.722, Loss:  0.643
    Epoch   0 Batch  690/1378 - Train Accuracy:  0.761, Validation Accuracy:  0.703, Loss:  0.688
    Epoch   0 Batch  691/1378 - Train Accuracy:  0.745, Validation Accuracy:  0.712, Loss:  0.675
    Epoch   0 Batch  692/1378 - Train Accuracy:  0.755, Validation Accuracy:  0.726, Loss:  0.619
    Epoch   0 Batch  693/1378 - Train Accuracy:  0.706, Validation Accuracy:  0.735, Loss:  0.669
    Epoch   0 Batch  694/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.730, Loss:  0.674
    Epoch   0 Batch  695/1378 - Train Accuracy:  0.770, Validation Accuracy:  0.728, Loss:  0.655
    Epoch   0 Batch  696/1378 - Train Accuracy:  0.705, Validation Accuracy:  0.735, Loss:  0.663
    Epoch   0 Batch  697/1378 - Train Accuracy:  0.731, Validation Accuracy:  0.742, Loss:  0.674
    Epoch   0 Batch  698/1378 - Train Accuracy:  0.720, Validation Accuracy:  0.735, Loss:  0.707
    Epoch   0 Batch  699/1378 - Train Accuracy:  0.726, Validation Accuracy:  0.745, Loss:  0.686
    Epoch   0 Batch  700/1378 - Train Accuracy:  0.724, Validation Accuracy:  0.744, Loss:  0.654
    Epoch   0 Batch  701/1378 - Train Accuracy:  0.740, Validation Accuracy:  0.746, Loss:  0.721
    Epoch   0 Batch  702/1378 - Train Accuracy:  0.729, Validation Accuracy:  0.746, Loss:  0.654
    Epoch   0 Batch  703/1378 - Train Accuracy:  0.696, Validation Accuracy:  0.727, Loss:  0.737
    Epoch   0 Batch  704/1378 - Train Accuracy:  0.689, Validation Accuracy:  0.726, Loss:  0.610
    Epoch   0 Batch  705/1378 - Train Accuracy:  0.700, Validation Accuracy:  0.715, Loss:  0.687
    Epoch   0 Batch  706/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.699, Loss:  0.726
    Epoch   0 Batch  707/1378 - Train Accuracy:  0.732, Validation Accuracy:  0.692, Loss:  0.682
    Epoch   0 Batch  708/1378 - Train Accuracy:  0.719, Validation Accuracy:  0.691, Loss:  0.649
    Epoch   0 Batch  709/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.716, Loss:  0.674
    Epoch   0 Batch  710/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.720, Loss:  0.664
    Epoch   0 Batch  711/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.728, Loss:  0.665
    Epoch   0 Batch  712/1378 - Train Accuracy:  0.743, Validation Accuracy:  0.730, Loss:  0.720
    Epoch   0 Batch  713/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.740, Loss:  0.616
    Epoch   0 Batch  714/1378 - Train Accuracy:  0.718, Validation Accuracy:  0.738, Loss:  0.675
    Epoch   0 Batch  715/1378 - Train Accuracy:  0.755, Validation Accuracy:  0.735, Loss:  0.663
    Epoch   0 Batch  716/1378 - Train Accuracy:  0.741, Validation Accuracy:  0.742, Loss:  0.610
    Epoch   0 Batch  717/1378 - Train Accuracy:  0.732, Validation Accuracy:  0.740, Loss:  0.580
    Epoch   0 Batch  718/1378 - Train Accuracy:  0.781, Validation Accuracy:  0.715, Loss:  0.614
    Epoch   0 Batch  719/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.687, Loss:  0.677
    Epoch   0 Batch  720/1378 - Train Accuracy:  0.746, Validation Accuracy:  0.680, Loss:  0.648
    Epoch   0 Batch  721/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.690, Loss:  0.677
    Epoch   0 Batch  722/1378 - Train Accuracy:  0.789, Validation Accuracy:  0.717, Loss:  0.629
    Epoch   0 Batch  723/1378 - Train Accuracy:  0.733, Validation Accuracy:  0.735, Loss:  0.670
    Epoch   0 Batch  724/1378 - Train Accuracy:  0.748, Validation Accuracy:  0.749, Loss:  0.678
    Epoch   0 Batch  725/1378 - Train Accuracy:  0.735, Validation Accuracy:  0.762, Loss:  0.661
    Epoch   0 Batch  726/1378 - Train Accuracy:  0.722, Validation Accuracy:  0.750, Loss:  0.604
    Epoch   0 Batch  727/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.755, Loss:  0.654
    Epoch   0 Batch  728/1378 - Train Accuracy:  0.769, Validation Accuracy:  0.748, Loss:  0.639
    Epoch   0 Batch  729/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.750, Loss:  0.687
    Epoch   0 Batch  730/1378 - Train Accuracy:  0.741, Validation Accuracy:  0.756, Loss:  0.645
    Epoch   0 Batch  731/1378 - Train Accuracy:  0.776, Validation Accuracy:  0.755, Loss:  0.641
    Epoch   0 Batch  732/1378 - Train Accuracy:  0.712, Validation Accuracy:  0.739, Loss:  0.628
    Epoch   0 Batch  733/1378 - Train Accuracy:  0.761, Validation Accuracy:  0.712, Loss:  0.680
    Epoch   0 Batch  734/1378 - Train Accuracy:  0.782, Validation Accuracy:  0.716, Loss:  0.690
    Epoch   0 Batch  735/1378 - Train Accuracy:  0.719, Validation Accuracy:  0.723, Loss:  0.658
    Epoch   0 Batch  736/1378 - Train Accuracy:  0.736, Validation Accuracy:  0.724, Loss:  0.642
    Epoch   0 Batch  737/1378 - Train Accuracy:  0.768, Validation Accuracy:  0.742, Loss:  0.663
    Epoch   0 Batch  738/1378 - Train Accuracy:  0.765, Validation Accuracy:  0.750, Loss:  0.648
    Epoch   0 Batch  739/1378 - Train Accuracy:  0.745, Validation Accuracy:  0.754, Loss:  0.670
    Epoch   0 Batch  740/1378 - Train Accuracy:  0.722, Validation Accuracy:  0.765, Loss:  0.624
    Epoch   0 Batch  741/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.766, Loss:  0.643
    Epoch   0 Batch  742/1378 - Train Accuracy:  0.738, Validation Accuracy:  0.763, Loss:  0.638
    Epoch   0 Batch  743/1378 - Train Accuracy:  0.757, Validation Accuracy:  0.767, Loss:  0.679
    Epoch   0 Batch  744/1378 - Train Accuracy:  0.778, Validation Accuracy:  0.782, Loss:  0.628
    Epoch   0 Batch  745/1378 - Train Accuracy:  0.732, Validation Accuracy:  0.765, Loss:  0.571
    Epoch   0 Batch  746/1378 - Train Accuracy:  0.753, Validation Accuracy:  0.736, Loss:  0.658
    Epoch   0 Batch  747/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.729, Loss:  0.690
    Epoch   0 Batch  748/1378 - Train Accuracy:  0.724, Validation Accuracy:  0.731, Loss:  0.618
    Epoch   0 Batch  749/1378 - Train Accuracy:  0.771, Validation Accuracy:  0.733, Loss:  0.650
    Epoch   0 Batch  750/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.731, Loss:  0.592
    Epoch   0 Batch  751/1378 - Train Accuracy:  0.734, Validation Accuracy:  0.728, Loss:  0.622
    Epoch   0 Batch  752/1378 - Train Accuracy:  0.766, Validation Accuracy:  0.727, Loss:  0.679
    Epoch   0 Batch  753/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.710, Loss:  0.626
    Epoch   0 Batch  754/1378 - Train Accuracy:  0.776, Validation Accuracy:  0.718, Loss:  0.665
    Epoch   0 Batch  755/1378 - Train Accuracy:  0.710, Validation Accuracy:  0.719, Loss:  0.662
    Epoch   0 Batch  756/1378 - Train Accuracy:  0.710, Validation Accuracy:  0.728, Loss:  0.638
    Epoch   0 Batch  757/1378 - Train Accuracy:  0.785, Validation Accuracy:  0.732, Loss:  0.642
    Epoch   0 Batch  758/1378 - Train Accuracy:  0.774, Validation Accuracy:  0.731, Loss:  0.612
    Epoch   0 Batch  759/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.731, Loss:  0.633
    Epoch   0 Batch  760/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.729, Loss:  0.652
    Epoch   0 Batch  761/1378 - Train Accuracy:  0.713, Validation Accuracy:  0.732, Loss:  0.655
    Epoch   0 Batch  762/1378 - Train Accuracy:  0.762, Validation Accuracy:  0.754, Loss:  0.653
    Epoch   0 Batch  763/1378 - Train Accuracy:  0.788, Validation Accuracy:  0.750, Loss:  0.660
    Epoch   0 Batch  764/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.745, Loss:  0.642
    Epoch   0 Batch  765/1378 - Train Accuracy:  0.719, Validation Accuracy:  0.750, Loss:  0.642
    Epoch   0 Batch  766/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.766, Loss:  0.649
    Epoch   0 Batch  767/1378 - Train Accuracy:  0.696, Validation Accuracy:  0.775, Loss:  0.703
    Epoch   0 Batch  768/1378 - Train Accuracy:  0.784, Validation Accuracy:  0.778, Loss:  0.650
    Epoch   0 Batch  769/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.769, Loss:  0.660
    Epoch   0 Batch  770/1378 - Train Accuracy:  0.700, Validation Accuracy:  0.720, Loss:  0.668
    Epoch   0 Batch  771/1378 - Train Accuracy:  0.743, Validation Accuracy:  0.709, Loss:  0.685
    Epoch   0 Batch  772/1378 - Train Accuracy:  0.738, Validation Accuracy:  0.692, Loss:  0.608
    Epoch   0 Batch  773/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.705, Loss:  0.699
    Epoch   0 Batch  774/1378 - Train Accuracy:  0.747, Validation Accuracy:  0.731, Loss:  0.637
    Epoch   0 Batch  775/1378 - Train Accuracy:  0.725, Validation Accuracy:  0.750, Loss:  0.699
    Epoch   0 Batch  776/1378 - Train Accuracy:  0.747, Validation Accuracy:  0.765, Loss:  0.614
    Epoch   0 Batch  777/1378 - Train Accuracy:  0.789, Validation Accuracy:  0.767, Loss:  0.648
    Epoch   0 Batch  778/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.762, Loss:  0.601
    Epoch   0 Batch  779/1378 - Train Accuracy:  0.741, Validation Accuracy:  0.756, Loss:  0.663
    Epoch   0 Batch  780/1378 - Train Accuracy:  0.748, Validation Accuracy:  0.755, Loss:  0.646
    Epoch   0 Batch  781/1378 - Train Accuracy:  0.705, Validation Accuracy:  0.742, Loss:  0.679
    Epoch   0 Batch  782/1378 - Train Accuracy:  0.747, Validation Accuracy:  0.717, Loss:  0.623
    Epoch   0 Batch  783/1378 - Train Accuracy:  0.742, Validation Accuracy:  0.715, Loss:  0.669
    Epoch   0 Batch  784/1378 - Train Accuracy:  0.761, Validation Accuracy:  0.706, Loss:  0.632
    Epoch   0 Batch  785/1378 - Train Accuracy:  0.681, Validation Accuracy:  0.716, Loss:  0.647
    Epoch   0 Batch  786/1378 - Train Accuracy:  0.758, Validation Accuracy:  0.733, Loss:  0.612
    Epoch   0 Batch  787/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.760, Loss:  0.610
    Epoch   0 Batch  788/1378 - Train Accuracy:  0.737, Validation Accuracy:  0.780, Loss:  0.691
    Epoch   0 Batch  789/1378 - Train Accuracy:  0.770, Validation Accuracy:  0.765, Loss:  0.621
    Epoch   0 Batch  790/1378 - Train Accuracy:  0.790, Validation Accuracy:  0.766, Loss:  0.606
    Epoch   0 Batch  791/1378 - Train Accuracy:  0.709, Validation Accuracy:  0.765, Loss:  0.701
    Epoch   0 Batch  792/1378 - Train Accuracy:  0.779, Validation Accuracy:  0.775, Loss:  0.623
    Epoch   0 Batch  793/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.755, Loss:  0.680
    Epoch   0 Batch  794/1378 - Train Accuracy:  0.772, Validation Accuracy:  0.724, Loss:  0.617
    Epoch   0 Batch  795/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.707, Loss:  0.633
    Epoch   0 Batch  796/1378 - Train Accuracy:  0.744, Validation Accuracy:  0.719, Loss:  0.614
    Epoch   0 Batch  797/1378 - Train Accuracy:  0.759, Validation Accuracy:  0.742, Loss:  0.634
    Epoch   0 Batch  798/1378 - Train Accuracy:  0.753, Validation Accuracy:  0.768, Loss:  0.714
    Epoch   0 Batch  799/1378 - Train Accuracy:  0.784, Validation Accuracy:  0.770, Loss:  0.642
    Epoch   0 Batch  800/1378 - Train Accuracy:  0.758, Validation Accuracy:  0.768, Loss:  0.598
    Epoch   0 Batch  801/1378 - Train Accuracy:  0.761, Validation Accuracy:  0.774, Loss:  0.612
    Epoch   0 Batch  802/1378 - Train Accuracy:  0.778, Validation Accuracy:  0.770, Loss:  0.645
    Epoch   0 Batch  803/1378 - Train Accuracy:  0.748, Validation Accuracy:  0.757, Loss:  0.630
    Epoch   0 Batch  804/1378 - Train Accuracy:  0.773, Validation Accuracy:  0.761, Loss:  0.619
    Epoch   0 Batch  805/1378 - Train Accuracy:  0.716, Validation Accuracy:  0.754, Loss:  0.706
    Epoch   0 Batch  806/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.737, Loss:  0.615
    Epoch   0 Batch  807/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.743, Loss:  0.639
    Epoch   0 Batch  808/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.752, Loss:  0.596
    Epoch   0 Batch  809/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.760, Loss:  0.612
    Epoch   0 Batch  810/1378 - Train Accuracy:  0.752, Validation Accuracy:  0.770, Loss:  0.658
    Epoch   0 Batch  811/1378 - Train Accuracy:  0.785, Validation Accuracy:  0.778, Loss:  0.705
    Epoch   0 Batch  812/1378 - Train Accuracy:  0.727, Validation Accuracy:  0.770, Loss:  0.636
    Epoch   0 Batch  813/1378 - Train Accuracy:  0.729, Validation Accuracy:  0.779, Loss:  0.631
    Epoch   0 Batch  814/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.789, Loss:  0.636
    Epoch   0 Batch  815/1378 - Train Accuracy:  0.788, Validation Accuracy:  0.784, Loss:  0.610
    Epoch   0 Batch  816/1378 - Train Accuracy:  0.733, Validation Accuracy:  0.755, Loss:  0.595
    Epoch   0 Batch  817/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.750, Loss:  0.589
    Epoch   0 Batch  818/1378 - Train Accuracy:  0.743, Validation Accuracy:  0.735, Loss:  0.608
    Epoch   0 Batch  819/1378 - Train Accuracy:  0.765, Validation Accuracy:  0.729, Loss:  0.648
    Epoch   0 Batch  820/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.728, Loss:  0.624
    Epoch   0 Batch  821/1378 - Train Accuracy:  0.760, Validation Accuracy:  0.729, Loss:  0.641
    Epoch   0 Batch  822/1378 - Train Accuracy:  0.707, Validation Accuracy:  0.746, Loss:  0.620
    Epoch   0 Batch  823/1378 - Train Accuracy:  0.759, Validation Accuracy:  0.752, Loss:  0.577
    Epoch   0 Batch  824/1378 - Train Accuracy:  0.768, Validation Accuracy:  0.760, Loss:  0.611
    Epoch   0 Batch  825/1378 - Train Accuracy:  0.715, Validation Accuracy:  0.781, Loss:  0.587
    Epoch   0 Batch  826/1378 - Train Accuracy:  0.800, Validation Accuracy:  0.787, Loss:  0.598
    Epoch   0 Batch  827/1378 - Train Accuracy:  0.784, Validation Accuracy:  0.786, Loss:  0.639
    Epoch   0 Batch  828/1378 - Train Accuracy:  0.801, Validation Accuracy:  0.778, Loss:  0.590
    Epoch   0 Batch  829/1378 - Train Accuracy:  0.744, Validation Accuracy:  0.777, Loss:  0.642
    Epoch   0 Batch  830/1378 - Train Accuracy:  0.777, Validation Accuracy:  0.764, Loss:  0.618
    Epoch   0 Batch  831/1378 - Train Accuracy:  0.793, Validation Accuracy:  0.762, Loss:  0.621
    Epoch   0 Batch  832/1378 - Train Accuracy:  0.720, Validation Accuracy:  0.751, Loss:  0.598
    Epoch   0 Batch  833/1378 - Train Accuracy:  0.704, Validation Accuracy:  0.762, Loss:  0.646
    Epoch   0 Batch  834/1378 - Train Accuracy:  0.753, Validation Accuracy:  0.753, Loss:  0.606
    Epoch   0 Batch  835/1378 - Train Accuracy:  0.757, Validation Accuracy:  0.755, Loss:  0.699
    Epoch   0 Batch  836/1378 - Train Accuracy:  0.798, Validation Accuracy:  0.745, Loss:  0.657
    Epoch   0 Batch  837/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.759, Loss:  0.681
    Epoch   0 Batch  838/1378 - Train Accuracy:  0.759, Validation Accuracy:  0.768, Loss:  0.609
    Epoch   0 Batch  839/1378 - Train Accuracy:  0.744, Validation Accuracy:  0.769, Loss:  0.578
    Epoch   0 Batch  840/1378 - Train Accuracy:  0.731, Validation Accuracy:  0.773, Loss:  0.621
    Epoch   0 Batch  841/1378 - Train Accuracy:  0.794, Validation Accuracy:  0.758, Loss:  0.587
    Epoch   0 Batch  842/1378 - Train Accuracy:  0.759, Validation Accuracy:  0.770, Loss:  0.601
    Epoch   0 Batch  843/1378 - Train Accuracy:  0.796, Validation Accuracy:  0.775, Loss:  0.585
    Epoch   0 Batch  844/1378 - Train Accuracy:  0.758, Validation Accuracy:  0.766, Loss:  0.658
    Epoch   0 Batch  845/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.786, Loss:  0.630
    Epoch   0 Batch  846/1378 - Train Accuracy:  0.833, Validation Accuracy:  0.779, Loss:  0.571
    Epoch   0 Batch  847/1378 - Train Accuracy:  0.739, Validation Accuracy:  0.769, Loss:  0.625
    Epoch   0 Batch  848/1378 - Train Accuracy:  0.762, Validation Accuracy:  0.766, Loss:  0.658
    Epoch   0 Batch  849/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.734, Loss:  0.564
    Epoch   0 Batch  850/1378 - Train Accuracy:  0.736, Validation Accuracy:  0.719, Loss:  0.604
    Epoch   0 Batch  851/1378 - Train Accuracy:  0.772, Validation Accuracy:  0.736, Loss:  0.613
    Epoch   0 Batch  852/1378 - Train Accuracy:  0.786, Validation Accuracy:  0.748, Loss:  0.618
    Epoch   0 Batch  853/1378 - Train Accuracy:  0.709, Validation Accuracy:  0.759, Loss:  0.663
    Epoch   0 Batch  854/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.765, Loss:  0.665
    Epoch   0 Batch  855/1378 - Train Accuracy:  0.782, Validation Accuracy:  0.774, Loss:  0.638
    Epoch   0 Batch  856/1378 - Train Accuracy:  0.732, Validation Accuracy:  0.755, Loss:  0.604
    Epoch   0 Batch  857/1378 - Train Accuracy:  0.785, Validation Accuracy:  0.761, Loss:  0.602
    Epoch   0 Batch  858/1378 - Train Accuracy:  0.711, Validation Accuracy:  0.757, Loss:  0.698
    Epoch   0 Batch  859/1378 - Train Accuracy:  0.737, Validation Accuracy:  0.765, Loss:  0.604
    Epoch   0 Batch  860/1378 - Train Accuracy:  0.808, Validation Accuracy:  0.758, Loss:  0.638
    Epoch   0 Batch  861/1378 - Train Accuracy:  0.698, Validation Accuracy:  0.770, Loss:  0.619
    Epoch   0 Batch  862/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.752, Loss:  0.625
    Epoch   0 Batch  863/1378 - Train Accuracy:  0.740, Validation Accuracy:  0.765, Loss:  0.608
    Epoch   0 Batch  864/1378 - Train Accuracy:  0.768, Validation Accuracy:  0.770, Loss:  0.620
    Epoch   0 Batch  865/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.761, Loss:  0.636
    Epoch   0 Batch  866/1378 - Train Accuracy:  0.761, Validation Accuracy:  0.772, Loss:  0.538
    Epoch   0 Batch  867/1378 - Train Accuracy:  0.709, Validation Accuracy:  0.776, Loss:  0.739
    Epoch   0 Batch  868/1378 - Train Accuracy:  0.720, Validation Accuracy:  0.783, Loss:  0.629
    Epoch   0 Batch  869/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.791, Loss:  0.603
    Epoch   0 Batch  870/1378 - Train Accuracy:  0.721, Validation Accuracy:  0.790, Loss:  0.623
    Epoch   0 Batch  871/1378 - Train Accuracy:  0.772, Validation Accuracy:  0.789, Loss:  0.619
    Epoch   0 Batch  872/1378 - Train Accuracy:  0.763, Validation Accuracy:  0.798, Loss:  0.692
    Epoch   0 Batch  873/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.778, Loss:  0.609
    Epoch   0 Batch  874/1378 - Train Accuracy:  0.755, Validation Accuracy:  0.764, Loss:  0.661
    Epoch   0 Batch  875/1378 - Train Accuracy:  0.745, Validation Accuracy:  0.733, Loss:  0.643
    Epoch   0 Batch  876/1378 - Train Accuracy:  0.768, Validation Accuracy:  0.739, Loss:  0.575
    Epoch   0 Batch  877/1378 - Train Accuracy:  0.743, Validation Accuracy:  0.762, Loss:  0.610
    Epoch   0 Batch  878/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.794, Loss:  0.605
    Epoch   0 Batch  879/1378 - Train Accuracy:  0.755, Validation Accuracy:  0.806, Loss:  0.578
    Epoch   0 Batch  880/1378 - Train Accuracy:  0.791, Validation Accuracy:  0.794, Loss:  0.606
    Epoch   0 Batch  881/1378 - Train Accuracy:  0.773, Validation Accuracy:  0.790, Loss:  0.629
    Epoch   0 Batch  882/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.782, Loss:  0.571
    Epoch   0 Batch  883/1378 - Train Accuracy:  0.741, Validation Accuracy:  0.779, Loss:  0.616
    Epoch   0 Batch  884/1378 - Train Accuracy:  0.763, Validation Accuracy:  0.770, Loss:  0.604
    Epoch   0 Batch  885/1378 - Train Accuracy:  0.683, Validation Accuracy:  0.761, Loss:  0.639
    Epoch   0 Batch  886/1378 - Train Accuracy:  0.771, Validation Accuracy:  0.743, Loss:  0.617
    Epoch   0 Batch  887/1378 - Train Accuracy:  0.696, Validation Accuracy:  0.742, Loss:  0.673
    Epoch   0 Batch  888/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.734, Loss:  0.623
    Epoch   0 Batch  889/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.737, Loss:  0.585
    Epoch   0 Batch  890/1378 - Train Accuracy:  0.783, Validation Accuracy:  0.750, Loss:  0.584
    Epoch   0 Batch  891/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.758, Loss:  0.626
    Epoch   0 Batch  892/1378 - Train Accuracy:  0.734, Validation Accuracy:  0.756, Loss:  0.647
    Epoch   0 Batch  893/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.757, Loss:  0.618
    Epoch   0 Batch  894/1378 - Train Accuracy:  0.775, Validation Accuracy:  0.763, Loss:  0.598
    Epoch   0 Batch  895/1378 - Train Accuracy:  0.738, Validation Accuracy:  0.765, Loss:  0.608
    Epoch   0 Batch  896/1378 - Train Accuracy:  0.714, Validation Accuracy:  0.756, Loss:  0.624
    Epoch   0 Batch  897/1378 - Train Accuracy:  0.760, Validation Accuracy:  0.738, Loss:  0.614
    Epoch   0 Batch  898/1378 - Train Accuracy:  0.774, Validation Accuracy:  0.746, Loss:  0.651
    Epoch   0 Batch  899/1378 - Train Accuracy:  0.812, Validation Accuracy:  0.763, Loss:  0.625
    Epoch   0 Batch  900/1378 - Train Accuracy:  0.763, Validation Accuracy:  0.781, Loss:  0.622
    Epoch   0 Batch  901/1378 - Train Accuracy:  0.787, Validation Accuracy:  0.805, Loss:  0.608
    Epoch   0 Batch  902/1378 - Train Accuracy:  0.771, Validation Accuracy:  0.802, Loss:  0.580
    Epoch   0 Batch  903/1378 - Train Accuracy:  0.768, Validation Accuracy:  0.800, Loss:  0.640
    Epoch   0 Batch  904/1378 - Train Accuracy:  0.764, Validation Accuracy:  0.796, Loss:  0.617
    Epoch   0 Batch  905/1378 - Train Accuracy:  0.757, Validation Accuracy:  0.790, Loss:  0.630
    Epoch   0 Batch  906/1378 - Train Accuracy:  0.739, Validation Accuracy:  0.790, Loss:  0.621
    Epoch   0 Batch  907/1378 - Train Accuracy:  0.755, Validation Accuracy:  0.788, Loss:  0.636
    Epoch   0 Batch  908/1378 - Train Accuracy:  0.753, Validation Accuracy:  0.750, Loss:  0.649
    Epoch   0 Batch  909/1378 - Train Accuracy:  0.723, Validation Accuracy:  0.748, Loss:  0.609
    Epoch   0 Batch  910/1378 - Train Accuracy:  0.746, Validation Accuracy:  0.753, Loss:  0.619
    Epoch   0 Batch  911/1378 - Train Accuracy:  0.734, Validation Accuracy:  0.762, Loss:  0.628
    Epoch   0 Batch  912/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.778, Loss:  0.563
    Epoch   0 Batch  913/1378 - Train Accuracy:  0.783, Validation Accuracy:  0.788, Loss:  0.618
    Epoch   0 Batch  914/1378 - Train Accuracy:  0.781, Validation Accuracy:  0.772, Loss:  0.568
    Epoch   0 Batch  915/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.783, Loss:  0.635
    Epoch   0 Batch  916/1378 - Train Accuracy:  0.767, Validation Accuracy:  0.778, Loss:  0.544
    Epoch   0 Batch  917/1378 - Train Accuracy:  0.771, Validation Accuracy:  0.771, Loss:  0.588
    Epoch   0 Batch  918/1378 - Train Accuracy:  0.817, Validation Accuracy:  0.764, Loss:  0.557
    Epoch   0 Batch  919/1378 - Train Accuracy:  0.772, Validation Accuracy:  0.775, Loss:  0.618
    Epoch   0 Batch  920/1378 - Train Accuracy:  0.784, Validation Accuracy:  0.769, Loss:  0.602
    Epoch   0 Batch  921/1378 - Train Accuracy:  0.730, Validation Accuracy:  0.795, Loss:  0.684
    Epoch   0 Batch  922/1378 - Train Accuracy:  0.792, Validation Accuracy:  0.798, Loss:  0.574
    Epoch   0 Batch  923/1378 - Train Accuracy:  0.757, Validation Accuracy:  0.799, Loss:  0.590
    Epoch   0 Batch  924/1378 - Train Accuracy:  0.771, Validation Accuracy:  0.805, Loss:  0.643
    Epoch   0 Batch  925/1378 - Train Accuracy:  0.699, Validation Accuracy:  0.801, Loss:  0.622
    Epoch   0 Batch  926/1378 - Train Accuracy:  0.817, Validation Accuracy:  0.798, Loss:  0.621
    Epoch   0 Batch  927/1378 - Train Accuracy:  0.783, Validation Accuracy:  0.794, Loss:  0.572
    Epoch   0 Batch  928/1378 - Train Accuracy:  0.745, Validation Accuracy:  0.791, Loss:  0.624
    Epoch   0 Batch  929/1378 - Train Accuracy:  0.771, Validation Accuracy:  0.781, Loss:  0.583
    Epoch   0 Batch  930/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.765, Loss:  0.582
    Epoch   0 Batch  931/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.748, Loss:  0.580
    Epoch   0 Batch  932/1378 - Train Accuracy:  0.783, Validation Accuracy:  0.731, Loss:  0.594
    Epoch   0 Batch  933/1378 - Train Accuracy:  0.736, Validation Accuracy:  0.740, Loss:  0.609
    Epoch   0 Batch  934/1378 - Train Accuracy:  0.735, Validation Accuracy:  0.735, Loss:  0.629
    Epoch   0 Batch  935/1378 - Train Accuracy:  0.750, Validation Accuracy:  0.734, Loss:  0.609
    Epoch   0 Batch  936/1378 - Train Accuracy:  0.751, Validation Accuracy:  0.731, Loss:  0.629
    Epoch   0 Batch  937/1378 - Train Accuracy:  0.756, Validation Accuracy:  0.762, Loss:  0.599
    Epoch   0 Batch  938/1378 - Train Accuracy:  0.766, Validation Accuracy:  0.770, Loss:  0.643
    Epoch   0 Batch  939/1378 - Train Accuracy:  0.827, Validation Accuracy:  0.788, Loss:  0.608
    Epoch   0 Batch  940/1378 - Train Accuracy:  0.787, Validation Accuracy:  0.801, Loss:  0.614
    Epoch   0 Batch  941/1378 - Train Accuracy:  0.783, Validation Accuracy:  0.797, Loss:  0.591
    Epoch   0 Batch  942/1378 - Train Accuracy:  0.764, Validation Accuracy:  0.800, Loss:  0.623
    Epoch   0 Batch  943/1378 - Train Accuracy:  0.779, Validation Accuracy:  0.804, Loss:  0.657
    Epoch   0 Batch  944/1378 - Train Accuracy:  0.786, Validation Accuracy:  0.812, Loss:  0.536
    Epoch   0 Batch  945/1378 - Train Accuracy:  0.797, Validation Accuracy:  0.828, Loss:  0.575
    Epoch   0 Batch  946/1378 - Train Accuracy:  0.782, Validation Accuracy:  0.808, Loss:  0.614
    Epoch   0 Batch  947/1378 - Train Accuracy:  0.789, Validation Accuracy:  0.791, Loss:  0.565
    Epoch   0 Batch  948/1378 - Train Accuracy:  0.764, Validation Accuracy:  0.794, Loss:  0.570
    Epoch   0 Batch  949/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.813, Loss:  0.674
    Epoch   0 Batch  950/1378 - Train Accuracy:  0.744, Validation Accuracy:  0.817, Loss:  0.601
    Epoch   0 Batch  951/1378 - Train Accuracy:  0.716, Validation Accuracy:  0.825, Loss:  0.616
    Epoch   0 Batch  952/1378 - Train Accuracy:  0.800, Validation Accuracy:  0.818, Loss:  0.622
    Epoch   0 Batch  953/1378 - Train Accuracy:  0.794, Validation Accuracy:  0.815, Loss:  0.558
    Epoch   0 Batch  954/1378 - Train Accuracy:  0.800, Validation Accuracy:  0.812, Loss:  0.626
    Epoch   0 Batch  955/1378 - Train Accuracy:  0.786, Validation Accuracy:  0.793, Loss:  0.637
    Epoch   0 Batch  956/1378 - Train Accuracy:  0.798, Validation Accuracy:  0.783, Loss:  0.566
    Epoch   0 Batch  957/1378 - Train Accuracy:  0.776, Validation Accuracy:  0.784, Loss:  0.579
    Epoch   0 Batch  958/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.788, Loss:  0.604
    Epoch   0 Batch  959/1378 - Train Accuracy:  0.813, Validation Accuracy:  0.785, Loss:  0.586
    Epoch   0 Batch  960/1378 - Train Accuracy:  0.747, Validation Accuracy:  0.787, Loss:  0.620
    Epoch   0 Batch  961/1378 - Train Accuracy:  0.774, Validation Accuracy:  0.810, Loss:  0.618
    Epoch   0 Batch  962/1378 - Train Accuracy:  0.796, Validation Accuracy:  0.799, Loss:  0.594
    Epoch   0 Batch  963/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.802, Loss:  0.588
    Epoch   0 Batch  964/1378 - Train Accuracy:  0.754, Validation Accuracy:  0.807, Loss:  0.622
    Epoch   0 Batch  965/1378 - Train Accuracy:  0.790, Validation Accuracy:  0.782, Loss:  0.590
    Epoch   0 Batch  966/1378 - Train Accuracy:  0.819, Validation Accuracy:  0.786, Loss:  0.606
    Epoch   0 Batch  967/1378 - Train Accuracy:  0.791, Validation Accuracy:  0.786, Loss:  0.569
    Epoch   0 Batch  968/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.784, Loss:  0.580
    Epoch   0 Batch  969/1378 - Train Accuracy:  0.813, Validation Accuracy:  0.779, Loss:  0.562
    Epoch   0 Batch  970/1378 - Train Accuracy:  0.801, Validation Accuracy:  0.778, Loss:  0.626
    Epoch   0 Batch  971/1378 - Train Accuracy:  0.808, Validation Accuracy:  0.785, Loss:  0.565
    Epoch   0 Batch  972/1378 - Train Accuracy:  0.787, Validation Accuracy:  0.763, Loss:  0.603
    Epoch   0 Batch  973/1378 - Train Accuracy:  0.763, Validation Accuracy:  0.770, Loss:  0.599
    Epoch   0 Batch  974/1378 - Train Accuracy:  0.818, Validation Accuracy:  0.759, Loss:  0.589
    Epoch   0 Batch  975/1378 - Train Accuracy:  0.792, Validation Accuracy:  0.763, Loss:  0.553
    Epoch   0 Batch  976/1378 - Train Accuracy:  0.821, Validation Accuracy:  0.764, Loss:  0.574
    Epoch   0 Batch  977/1378 - Train Accuracy:  0.762, Validation Accuracy:  0.761, Loss:  0.628
    Epoch   0 Batch  978/1378 - Train Accuracy:  0.791, Validation Accuracy:  0.772, Loss:  0.558
    Epoch   0 Batch  979/1378 - Train Accuracy:  0.785, Validation Accuracy:  0.781, Loss:  0.603
    Epoch   0 Batch  980/1378 - Train Accuracy:  0.793, Validation Accuracy:  0.794, Loss:  0.605
    Epoch   0 Batch  981/1378 - Train Accuracy:  0.749, Validation Accuracy:  0.798, Loss:  0.581
    Epoch   0 Batch  982/1378 - Train Accuracy:  0.790, Validation Accuracy:  0.805, Loss:  0.573
    Epoch   0 Batch  983/1378 - Train Accuracy:  0.796, Validation Accuracy:  0.805, Loss:  0.570
    Epoch   0 Batch  984/1378 - Train Accuracy:  0.759, Validation Accuracy:  0.808, Loss:  0.573
    Epoch   0 Batch  985/1378 - Train Accuracy:  0.822, Validation Accuracy:  0.822, Loss:  0.526
    Epoch   0 Batch  986/1378 - Train Accuracy:  0.801, Validation Accuracy:  0.822, Loss:  0.604
    Epoch   0 Batch  987/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.816, Loss:  0.606
    Epoch   0 Batch  988/1378 - Train Accuracy:  0.781, Validation Accuracy:  0.807, Loss:  0.570
    Epoch   0 Batch  989/1378 - Train Accuracy:  0.779, Validation Accuracy:  0.805, Loss:  0.626
    Epoch   0 Batch  990/1378 - Train Accuracy:  0.798, Validation Accuracy:  0.794, Loss:  0.596
    Epoch   0 Batch  991/1378 - Train Accuracy:  0.769, Validation Accuracy:  0.809, Loss:  0.553
    Epoch   0 Batch  992/1378 - Train Accuracy:  0.806, Validation Accuracy:  0.802, Loss:  0.601
    Epoch   0 Batch  993/1378 - Train Accuracy:  0.786, Validation Accuracy:  0.804, Loss:  0.580
    Epoch   0 Batch  994/1378 - Train Accuracy:  0.821, Validation Accuracy:  0.799, Loss:  0.558
    Epoch   0 Batch  995/1378 - Train Accuracy:  0.777, Validation Accuracy:  0.800, Loss:  0.551
    Epoch   0 Batch  996/1378 - Train Accuracy:  0.792, Validation Accuracy:  0.808, Loss:  0.602
    Epoch   0 Batch  997/1378 - Train Accuracy:  0.814, Validation Accuracy:  0.806, Loss:  0.547
    Epoch   0 Batch  998/1378 - Train Accuracy:  0.806, Validation Accuracy:  0.803, Loss:  0.545
    Epoch   0 Batch  999/1378 - Train Accuracy:  0.775, Validation Accuracy:  0.798, Loss:  0.582
    Epoch   0 Batch 1000/1378 - Train Accuracy:  0.827, Validation Accuracy:  0.786, Loss:  0.601
    Epoch   0 Batch 1001/1378 - Train Accuracy:  0.828, Validation Accuracy:  0.772, Loss:  0.598
    Epoch   0 Batch 1002/1378 - Train Accuracy:  0.787, Validation Accuracy:  0.783, Loss:  0.608
    Epoch   0 Batch 1003/1378 - Train Accuracy:  0.830, Validation Accuracy:  0.789, Loss:  0.622
    Epoch   0 Batch 1004/1378 - Train Accuracy:  0.798, Validation Accuracy:  0.785, Loss:  0.570
    Epoch   0 Batch 1005/1378 - Train Accuracy:  0.818, Validation Accuracy:  0.787, Loss:  0.557
    Epoch   0 Batch 1006/1378 - Train Accuracy:  0.770, Validation Accuracy:  0.785, Loss:  0.649
    Epoch   0 Batch 1007/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.776, Loss:  0.638
    Epoch   0 Batch 1008/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.774, Loss:  0.551
    Epoch   0 Batch 1009/1378 - Train Accuracy:  0.762, Validation Accuracy:  0.751, Loss:  0.566
    Epoch   0 Batch 1010/1378 - Train Accuracy:  0.728, Validation Accuracy:  0.762, Loss:  0.598
    Epoch   0 Batch 1011/1378 - Train Accuracy:  0.762, Validation Accuracy:  0.782, Loss:  0.582
    Epoch   0 Batch 1012/1378 - Train Accuracy:  0.794, Validation Accuracy:  0.812, Loss:  0.613
    Epoch   0 Batch 1013/1378 - Train Accuracy:  0.784, Validation Accuracy:  0.812, Loss:  0.573
    Epoch   0 Batch 1014/1378 - Train Accuracy:  0.774, Validation Accuracy:  0.812, Loss:  0.598
    Epoch   0 Batch 1015/1378 - Train Accuracy:  0.773, Validation Accuracy:  0.825, Loss:  0.619
    Epoch   0 Batch 1016/1378 - Train Accuracy:  0.798, Validation Accuracy:  0.847, Loss:  0.549
    Epoch   0 Batch 1017/1378 - Train Accuracy:  0.830, Validation Accuracy:  0.839, Loss:  0.572
    Epoch   0 Batch 1018/1378 - Train Accuracy:  0.805, Validation Accuracy:  0.842, Loss:  0.650
    Epoch   0 Batch 1019/1378 - Train Accuracy:  0.818, Validation Accuracy:  0.836, Loss:  0.610
    Epoch   0 Batch 1020/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.821, Loss:  0.526
    Epoch   0 Batch 1021/1378 - Train Accuracy:  0.787, Validation Accuracy:  0.815, Loss:  0.593
    Epoch   0 Batch 1022/1378 - Train Accuracy:  0.811, Validation Accuracy:  0.821, Loss:  0.588
    Epoch   0 Batch 1023/1378 - Train Accuracy:  0.786, Validation Accuracy:  0.830, Loss:  0.619
    Epoch   0 Batch 1024/1378 - Train Accuracy:  0.809, Validation Accuracy:  0.844, Loss:  0.614
    Epoch   0 Batch 1025/1378 - Train Accuracy:  0.741, Validation Accuracy:  0.845, Loss:  0.615
    Epoch   0 Batch 1026/1378 - Train Accuracy:  0.842, Validation Accuracy:  0.835, Loss:  0.549
    Epoch   0 Batch 1027/1378 - Train Accuracy:  0.820, Validation Accuracy:  0.816, Loss:  0.578
    Epoch   0 Batch 1028/1378 - Train Accuracy:  0.777, Validation Accuracy:  0.819, Loss:  0.645
    Epoch   0 Batch 1029/1378 - Train Accuracy:  0.777, Validation Accuracy:  0.819, Loss:  0.653
    Epoch   0 Batch 1030/1378 - Train Accuracy:  0.830, Validation Accuracy:  0.817, Loss:  0.556
    Epoch   0 Batch 1031/1378 - Train Accuracy:  0.831, Validation Accuracy:  0.822, Loss:  0.584
    Epoch   0 Batch 1032/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.806, Loss:  0.546
    Epoch   0 Batch 1033/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.810, Loss:  0.587
    Epoch   0 Batch 1034/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.804, Loss:  0.618
    Epoch   0 Batch 1035/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.798, Loss:  0.612
    Epoch   0 Batch 1036/1378 - Train Accuracy:  0.758, Validation Accuracy:  0.795, Loss:  0.621
    Epoch   0 Batch 1037/1378 - Train Accuracy:  0.831, Validation Accuracy:  0.785, Loss:  0.547
    Epoch   0 Batch 1038/1378 - Train Accuracy:  0.778, Validation Accuracy:  0.796, Loss:  0.597
    Epoch   0 Batch 1039/1378 - Train Accuracy:  0.777, Validation Accuracy:  0.791, Loss:  0.561
    Epoch   0 Batch 1040/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.815, Loss:  0.545
    Epoch   0 Batch 1041/1378 - Train Accuracy:  0.839, Validation Accuracy:  0.821, Loss:  0.554
    Epoch   0 Batch 1042/1378 - Train Accuracy:  0.843, Validation Accuracy:  0.815, Loss:  0.608
    Epoch   0 Batch 1043/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.824, Loss:  0.573
    Epoch   0 Batch 1044/1378 - Train Accuracy:  0.817, Validation Accuracy:  0.819, Loss:  0.582
    Epoch   0 Batch 1045/1378 - Train Accuracy:  0.807, Validation Accuracy:  0.824, Loss:  0.607
    Epoch   0 Batch 1046/1378 - Train Accuracy:  0.823, Validation Accuracy:  0.830, Loss:  0.615
    Epoch   0 Batch 1047/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.822, Loss:  0.588
    Epoch   0 Batch 1048/1378 - Train Accuracy:  0.817, Validation Accuracy:  0.811, Loss:  0.581
    Epoch   0 Batch 1049/1378 - Train Accuracy:  0.832, Validation Accuracy:  0.806, Loss:  0.547
    Epoch   0 Batch 1050/1378 - Train Accuracy:  0.809, Validation Accuracy:  0.815, Loss:  0.584
    Epoch   0 Batch 1051/1378 - Train Accuracy:  0.845, Validation Accuracy:  0.826, Loss:  0.557
    Epoch   0 Batch 1052/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.811, Loss:  0.562
    Epoch   0 Batch 1053/1378 - Train Accuracy:  0.786, Validation Accuracy:  0.811, Loss:  0.576
    Epoch   0 Batch 1054/1378 - Train Accuracy:  0.835, Validation Accuracy:  0.806, Loss:  0.547
    Epoch   0 Batch 1055/1378 - Train Accuracy:  0.814, Validation Accuracy:  0.826, Loss:  0.571
    Epoch   0 Batch 1056/1378 - Train Accuracy:  0.831, Validation Accuracy:  0.842, Loss:  0.550
    Epoch   0 Batch 1057/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.852, Loss:  0.549
    Epoch   0 Batch 1058/1378 - Train Accuracy:  0.798, Validation Accuracy:  0.856, Loss:  0.563
    Epoch   0 Batch 1059/1378 - Train Accuracy:  0.817, Validation Accuracy:  0.860, Loss:  0.515
    Epoch   0 Batch 1060/1378 - Train Accuracy:  0.823, Validation Accuracy:  0.865, Loss:  0.553
    Epoch   0 Batch 1061/1378 - Train Accuracy:  0.822, Validation Accuracy:  0.856, Loss:  0.551
    Epoch   0 Batch 1062/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.842, Loss:  0.548
    Epoch   0 Batch 1063/1378 - Train Accuracy:  0.797, Validation Accuracy:  0.848, Loss:  0.570
    Epoch   0 Batch 1064/1378 - Train Accuracy:  0.804, Validation Accuracy:  0.837, Loss:  0.572
    Epoch   0 Batch 1065/1378 - Train Accuracy:  0.826, Validation Accuracy:  0.855, Loss:  0.619
    Epoch   0 Batch 1066/1378 - Train Accuracy:  0.835, Validation Accuracy:  0.833, Loss:  0.567
    Epoch   0 Batch 1067/1378 - Train Accuracy:  0.822, Validation Accuracy:  0.839, Loss:  0.582
    Epoch   0 Batch 1068/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.826, Loss:  0.540
    Epoch   0 Batch 1069/1378 - Train Accuracy:  0.819, Validation Accuracy:  0.816, Loss:  0.596
    Epoch   0 Batch 1070/1378 - Train Accuracy:  0.837, Validation Accuracy:  0.816, Loss:  0.577
    Epoch   0 Batch 1071/1378 - Train Accuracy:  0.824, Validation Accuracy:  0.816, Loss:  0.599
    Epoch   0 Batch 1072/1378 - Train Accuracy:  0.842, Validation Accuracy:  0.834, Loss:  0.580
    Epoch   0 Batch 1073/1378 - Train Accuracy:  0.778, Validation Accuracy:  0.826, Loss:  0.583
    Epoch   0 Batch 1074/1378 - Train Accuracy:  0.841, Validation Accuracy:  0.807, Loss:  0.555
    Epoch   0 Batch 1075/1378 - Train Accuracy:  0.855, Validation Accuracy:  0.804, Loss:  0.532
    Epoch   0 Batch 1076/1378 - Train Accuracy:  0.843, Validation Accuracy:  0.815, Loss:  0.531
    Epoch   0 Batch 1077/1378 - Train Accuracy:  0.833, Validation Accuracy:  0.823, Loss:  0.565
    Epoch   0 Batch 1078/1378 - Train Accuracy:  0.837, Validation Accuracy:  0.825, Loss:  0.517
    Epoch   0 Batch 1079/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.852, Loss:  0.542
    Epoch   0 Batch 1080/1378 - Train Accuracy:  0.850, Validation Accuracy:  0.848, Loss:  0.623
    Epoch   0 Batch 1081/1378 - Train Accuracy:  0.837, Validation Accuracy:  0.848, Loss:  0.565
    Epoch   0 Batch 1082/1378 - Train Accuracy:  0.843, Validation Accuracy:  0.845, Loss:  0.567
    Epoch   0 Batch 1083/1378 - Train Accuracy:  0.789, Validation Accuracy:  0.845, Loss:  0.605
    Epoch   0 Batch 1084/1378 - Train Accuracy:  0.848, Validation Accuracy:  0.840, Loss:  0.589
    Epoch   0 Batch 1085/1378 - Train Accuracy:  0.837, Validation Accuracy:  0.841, Loss:  0.591
    Epoch   0 Batch 1086/1378 - Train Accuracy:  0.836, Validation Accuracy:  0.827, Loss:  0.535
    Epoch   0 Batch 1087/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.827, Loss:  0.568
    Epoch   0 Batch 1088/1378 - Train Accuracy:  0.811, Validation Accuracy:  0.827, Loss:  0.595
    Epoch   0 Batch 1089/1378 - Train Accuracy:  0.829, Validation Accuracy:  0.837, Loss:  0.606
    Epoch   0 Batch 1090/1378 - Train Accuracy:  0.800, Validation Accuracy:  0.851, Loss:  0.581
    Epoch   0 Batch 1091/1378 - Train Accuracy:  0.821, Validation Accuracy:  0.843, Loss:  0.581
    Epoch   0 Batch 1092/1378 - Train Accuracy:  0.866, Validation Accuracy:  0.849, Loss:  0.519
    Epoch   0 Batch 1093/1378 - Train Accuracy:  0.850, Validation Accuracy:  0.841, Loss:  0.578
    Epoch   0 Batch 1094/1378 - Train Accuracy:  0.801, Validation Accuracy:  0.820, Loss:  0.606
    Epoch   0 Batch 1095/1378 - Train Accuracy:  0.821, Validation Accuracy:  0.844, Loss:  0.582
    Epoch   0 Batch 1096/1378 - Train Accuracy:  0.826, Validation Accuracy:  0.834, Loss:  0.596
    Epoch   0 Batch 1097/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.834, Loss:  0.575
    Epoch   0 Batch 1098/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.850, Loss:  0.559
    Epoch   0 Batch 1099/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.850, Loss:  0.538
    Epoch   0 Batch 1100/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.849, Loss:  0.572
    Epoch   0 Batch 1101/1378 - Train Accuracy:  0.842, Validation Accuracy:  0.856, Loss:  0.552
    Epoch   0 Batch 1102/1378 - Train Accuracy:  0.800, Validation Accuracy:  0.853, Loss:  0.601
    Epoch   0 Batch 1103/1378 - Train Accuracy:  0.814, Validation Accuracy:  0.841, Loss:  0.528
    Epoch   0 Batch 1104/1378 - Train Accuracy:  0.829, Validation Accuracy:  0.842, Loss:  0.542
    Epoch   0 Batch 1105/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.853, Loss:  0.526
    Epoch   0 Batch 1106/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.851, Loss:  0.544
    Epoch   0 Batch 1107/1378 - Train Accuracy:  0.841, Validation Accuracy:  0.853, Loss:  0.557
    Epoch   0 Batch 1108/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.847, Loss:  0.536
    Epoch   0 Batch 1109/1378 - Train Accuracy:  0.846, Validation Accuracy:  0.848, Loss:  0.585
    Epoch   0 Batch 1110/1378 - Train Accuracy:  0.825, Validation Accuracy:  0.841, Loss:  0.632
    Epoch   0 Batch 1111/1378 - Train Accuracy:  0.826, Validation Accuracy:  0.831, Loss:  0.607
    Epoch   0 Batch 1112/1378 - Train Accuracy:  0.843, Validation Accuracy:  0.832, Loss:  0.583
    Epoch   0 Batch 1113/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.828, Loss:  0.591
    Epoch   0 Batch 1114/1378 - Train Accuracy:  0.795, Validation Accuracy:  0.839, Loss:  0.670
    Epoch   0 Batch 1115/1378 - Train Accuracy:  0.845, Validation Accuracy:  0.855, Loss:  0.524
    Epoch   0 Batch 1116/1378 - Train Accuracy:  0.849, Validation Accuracy:  0.846, Loss:  0.564
    Epoch   0 Batch 1117/1378 - Train Accuracy:  0.854, Validation Accuracy:  0.845, Loss:  0.570
    Epoch   0 Batch 1118/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.855, Loss:  0.612
    Epoch   0 Batch 1119/1378 - Train Accuracy:  0.834, Validation Accuracy:  0.864, Loss:  0.583
    Epoch   0 Batch 1120/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.860, Loss:  0.535
    Epoch   0 Batch 1121/1378 - Train Accuracy:  0.834, Validation Accuracy:  0.860, Loss:  0.607
    Epoch   0 Batch 1122/1378 - Train Accuracy:  0.849, Validation Accuracy:  0.852, Loss:  0.573
    Epoch   0 Batch 1123/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.846, Loss:  0.539
    Epoch   0 Batch 1124/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.850, Loss:  0.519
    Epoch   0 Batch 1125/1378 - Train Accuracy:  0.850, Validation Accuracy:  0.840, Loss:  0.567
    Epoch   0 Batch 1126/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.833, Loss:  0.516
    Epoch   0 Batch 1127/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.840, Loss:  0.559
    Epoch   0 Batch 1128/1378 - Train Accuracy:  0.827, Validation Accuracy:  0.844, Loss:  0.585
    Epoch   0 Batch 1129/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.837, Loss:  0.586
    Epoch   0 Batch 1130/1378 - Train Accuracy:  0.828, Validation Accuracy:  0.843, Loss:  0.560
    Epoch   0 Batch 1131/1378 - Train Accuracy:  0.818, Validation Accuracy:  0.844, Loss:  0.517
    Epoch   0 Batch 1132/1378 - Train Accuracy:  0.838, Validation Accuracy:  0.858, Loss:  0.558
    Epoch   0 Batch 1133/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.871, Loss:  0.546
    Epoch   0 Batch 1134/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.875, Loss:  0.559
    Epoch   0 Batch 1135/1378 - Train Accuracy:  0.833, Validation Accuracy:  0.867, Loss:  0.596
    Epoch   0 Batch 1136/1378 - Train Accuracy:  0.792, Validation Accuracy:  0.869, Loss:  0.607
    Epoch   0 Batch 1137/1378 - Train Accuracy:  0.850, Validation Accuracy:  0.860, Loss:  0.540
    Epoch   0 Batch 1138/1378 - Train Accuracy:  0.833, Validation Accuracy:  0.860, Loss:  0.525
    Epoch   0 Batch 1139/1378 - Train Accuracy:  0.861, Validation Accuracy:  0.853, Loss:  0.573
    Epoch   0 Batch 1140/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.851, Loss:  0.585
    Epoch   0 Batch 1141/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.846, Loss:  0.574
    Epoch   0 Batch 1142/1378 - Train Accuracy:  0.873, Validation Accuracy:  0.833, Loss:  0.527
    Epoch   0 Batch 1143/1378 - Train Accuracy:  0.840, Validation Accuracy:  0.843, Loss:  0.564
    Epoch   0 Batch 1144/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.847, Loss:  0.532
    Epoch   0 Batch 1145/1378 - Train Accuracy:  0.855, Validation Accuracy:  0.855, Loss:  0.577
    Epoch   0 Batch 1146/1378 - Train Accuracy:  0.826, Validation Accuracy:  0.859, Loss:  0.546
    Epoch   0 Batch 1147/1378 - Train Accuracy:  0.850, Validation Accuracy:  0.853, Loss:  0.584
    Epoch   0 Batch 1148/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.860, Loss:  0.521
    Epoch   0 Batch 1149/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.862, Loss:  0.552
    Epoch   0 Batch 1150/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.868, Loss:  0.501
    Epoch   0 Batch 1151/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.869, Loss:  0.527
    Epoch   0 Batch 1152/1378 - Train Accuracy:  0.856, Validation Accuracy:  0.866, Loss:  0.578
    Epoch   0 Batch 1153/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.872, Loss:  0.559
    Epoch   0 Batch 1154/1378 - Train Accuracy:  0.846, Validation Accuracy:  0.867, Loss:  0.560
    Epoch   0 Batch 1155/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.863, Loss:  0.542
    Epoch   0 Batch 1156/1378 - Train Accuracy:  0.836, Validation Accuracy:  0.882, Loss:  0.584
    Epoch   0 Batch 1157/1378 - Train Accuracy:  0.840, Validation Accuracy:  0.877, Loss:  0.586
    Epoch   0 Batch 1158/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.863, Loss:  0.508
    Epoch   0 Batch 1159/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.861, Loss:  0.571
    Epoch   0 Batch 1160/1378 - Train Accuracy:  0.842, Validation Accuracy:  0.859, Loss:  0.544
    Epoch   0 Batch 1161/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.856, Loss:  0.566
    Epoch   0 Batch 1162/1378 - Train Accuracy:  0.839, Validation Accuracy:  0.875, Loss:  0.558
    Epoch   0 Batch 1163/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.872, Loss:  0.525
    Epoch   0 Batch 1164/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.879, Loss:  0.582
    Epoch   0 Batch 1165/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.855, Loss:  0.590
    Epoch   0 Batch 1166/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.849, Loss:  0.509
    Epoch   0 Batch 1167/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.865, Loss:  0.564
    Epoch   0 Batch 1168/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.862, Loss:  0.554
    Epoch   0 Batch 1169/1378 - Train Accuracy:  0.829, Validation Accuracy:  0.861, Loss:  0.560
    Epoch   0 Batch 1170/1378 - Train Accuracy:  0.832, Validation Accuracy:  0.869, Loss:  0.631
    Epoch   0 Batch 1171/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.870, Loss:  0.584
    Epoch   0 Batch 1172/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.885, Loss:  0.555
    Epoch   0 Batch 1173/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.879, Loss:  0.524
    Epoch   0 Batch 1174/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.879, Loss:  0.535
    Epoch   0 Batch 1175/1378 - Train Accuracy:  0.841, Validation Accuracy:  0.871, Loss:  0.599
    Epoch   0 Batch 1176/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.863, Loss:  0.542
    Epoch   0 Batch 1177/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.858, Loss:  0.563
    Epoch   0 Batch 1178/1378 - Train Accuracy:  0.830, Validation Accuracy:  0.870, Loss:  0.533
    Epoch   0 Batch 1179/1378 - Train Accuracy:  0.829, Validation Accuracy:  0.863, Loss:  0.568
    Epoch   0 Batch 1180/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.862, Loss:  0.554
    Epoch   0 Batch 1181/1378 - Train Accuracy:  0.801, Validation Accuracy:  0.861, Loss:  0.568
    Epoch   0 Batch 1182/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.857, Loss:  0.587
    Epoch   0 Batch 1183/1378 - Train Accuracy:  0.841, Validation Accuracy:  0.846, Loss:  0.612
    Epoch   0 Batch 1184/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.833, Loss:  0.540
    Epoch   0 Batch 1185/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.832, Loss:  0.542
    Epoch   0 Batch 1186/1378 - Train Accuracy:  0.812, Validation Accuracy:  0.834, Loss:  0.579
    Epoch   0 Batch 1187/1378 - Train Accuracy:  0.822, Validation Accuracy:  0.830, Loss:  0.588
    Epoch   0 Batch 1188/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.849, Loss:  0.582
    Epoch   0 Batch 1189/1378 - Train Accuracy:  0.827, Validation Accuracy:  0.862, Loss:  0.524
    Epoch   0 Batch 1190/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.855, Loss:  0.485
    Epoch   0 Batch 1191/1378 - Train Accuracy:  0.854, Validation Accuracy:  0.838, Loss:  0.498
    Epoch   0 Batch 1192/1378 - Train Accuracy:  0.866, Validation Accuracy:  0.849, Loss:  0.567
    Epoch   0 Batch 1193/1378 - Train Accuracy:  0.839, Validation Accuracy:  0.842, Loss:  0.582
    Epoch   0 Batch 1194/1378 - Train Accuracy:  0.780, Validation Accuracy:  0.850, Loss:  0.584
    Epoch   0 Batch 1195/1378 - Train Accuracy:  0.828, Validation Accuracy:  0.844, Loss:  0.573
    Epoch   0 Batch 1196/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.849, Loss:  0.532
    Epoch   0 Batch 1197/1378 - Train Accuracy:  0.835, Validation Accuracy:  0.867, Loss:  0.560
    Epoch   0 Batch 1198/1378 - Train Accuracy:  0.851, Validation Accuracy:  0.872, Loss:  0.554
    Epoch   0 Batch 1199/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.859, Loss:  0.610
    Epoch   0 Batch 1200/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.858, Loss:  0.568
    Epoch   0 Batch 1201/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.865, Loss:  0.533
    Epoch   0 Batch 1202/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.853, Loss:  0.530
    Epoch   0 Batch 1203/1378 - Train Accuracy:  0.846, Validation Accuracy:  0.854, Loss:  0.513
    Epoch   0 Batch 1204/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.850, Loss:  0.544
    Epoch   0 Batch 1205/1378 - Train Accuracy:  0.816, Validation Accuracy:  0.855, Loss:  0.530
    Epoch   0 Batch 1206/1378 - Train Accuracy:  0.808, Validation Accuracy:  0.844, Loss:  0.609
    Epoch   0 Batch 1207/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.845, Loss:  0.508
    Epoch   0 Batch 1208/1378 - Train Accuracy:  0.833, Validation Accuracy:  0.856, Loss:  0.481
    Epoch   0 Batch 1209/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.867, Loss:  0.575
    Epoch   0 Batch 1210/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.861, Loss:  0.513
    Epoch   0 Batch 1211/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.856, Loss:  0.585
    Epoch   0 Batch 1212/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.864, Loss:  0.564
    Epoch   0 Batch 1213/1378 - Train Accuracy:  0.824, Validation Accuracy:  0.863, Loss:  0.606
    Epoch   0 Batch 1214/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.863, Loss:  0.567
    Epoch   0 Batch 1215/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.874, Loss:  0.541
    Epoch   0 Batch 1216/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.874, Loss:  0.477
    Epoch   0 Batch 1217/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.872, Loss:  0.540
    Epoch   0 Batch 1218/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.875, Loss:  0.549
    Epoch   0 Batch 1219/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.853, Loss:  0.524
    Epoch   0 Batch 1220/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.845, Loss:  0.511
    Epoch   0 Batch 1221/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.849, Loss:  0.568
    Epoch   0 Batch 1222/1378 - Train Accuracy:  0.831, Validation Accuracy:  0.837, Loss:  0.553
    Epoch   0 Batch 1223/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.843, Loss:  0.537
    Epoch   0 Batch 1224/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.852, Loss:  0.574
    Epoch   0 Batch 1225/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.858, Loss:  0.509
    Epoch   0 Batch 1226/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.863, Loss:  0.576
    Epoch   0 Batch 1227/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.859, Loss:  0.517
    Epoch   0 Batch 1228/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.865, Loss:  0.538
    Epoch   0 Batch 1229/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.865, Loss:  0.532
    Epoch   0 Batch 1230/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.876, Loss:  0.531
    Epoch   0 Batch 1231/1378 - Train Accuracy:  0.849, Validation Accuracy:  0.898, Loss:  0.571
    Epoch   0 Batch 1232/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.880, Loss:  0.527
    Epoch   0 Batch 1233/1378 - Train Accuracy:  0.859, Validation Accuracy:  0.885, Loss:  0.615
    Epoch   0 Batch 1234/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.874, Loss:  0.509
    Epoch   0 Batch 1235/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.865, Loss:  0.548
    Epoch   0 Batch 1236/1378 - Train Accuracy:  0.825, Validation Accuracy:  0.880, Loss:  0.633
    Epoch   0 Batch 1237/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.889, Loss:  0.522
    Epoch   0 Batch 1238/1378 - Train Accuracy:  0.861, Validation Accuracy:  0.880, Loss:  0.510
    Epoch   0 Batch 1239/1378 - Train Accuracy:  0.820, Validation Accuracy:  0.891, Loss:  0.569
    Epoch   0 Batch 1240/1378 - Train Accuracy:  0.829, Validation Accuracy:  0.884, Loss:  0.575
    Epoch   0 Batch 1241/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.884, Loss:  0.537
    Epoch   0 Batch 1242/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.867, Loss:  0.543
    Epoch   0 Batch 1243/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.861, Loss:  0.548
    Epoch   0 Batch 1244/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.863, Loss:  0.570
    Epoch   0 Batch 1245/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.857, Loss:  0.593
    Epoch   0 Batch 1246/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.855, Loss:  0.509
    Epoch   0 Batch 1247/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.837, Loss:  0.539
    Epoch   0 Batch 1248/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.847, Loss:  0.530
    Epoch   0 Batch 1249/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.842, Loss:  0.504
    Epoch   0 Batch 1250/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.845, Loss:  0.542
    Epoch   0 Batch 1251/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.845, Loss:  0.533
    Epoch   0 Batch 1252/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.851, Loss:  0.523
    Epoch   0 Batch 1253/1378 - Train Accuracy:  0.825, Validation Accuracy:  0.858, Loss:  0.609
    Epoch   0 Batch 1254/1378 - Train Accuracy:  0.845, Validation Accuracy:  0.864, Loss:  0.582
    Epoch   0 Batch 1255/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.863, Loss:  0.538
    Epoch   0 Batch 1256/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.854, Loss:  0.584
    Epoch   0 Batch 1257/1378 - Train Accuracy:  0.855, Validation Accuracy:  0.861, Loss:  0.537
    Epoch   0 Batch 1258/1378 - Train Accuracy:  0.851, Validation Accuracy:  0.874, Loss:  0.561
    Epoch   0 Batch 1259/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.883, Loss:  0.540
    Epoch   0 Batch 1260/1378 - Train Accuracy:  0.802, Validation Accuracy:  0.878, Loss:  0.566
    Epoch   0 Batch 1261/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.887, Loss:  0.592
    Epoch   0 Batch 1262/1378 - Train Accuracy:  0.843, Validation Accuracy:  0.897, Loss:  0.576
    Epoch   0 Batch 1263/1378 - Train Accuracy:  0.844, Validation Accuracy:  0.891, Loss:  0.545
    Epoch   0 Batch 1264/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.890, Loss:  0.567
    Epoch   0 Batch 1265/1378 - Train Accuracy:  0.851, Validation Accuracy:  0.901, Loss:  0.621
    Epoch   0 Batch 1266/1378 - Train Accuracy:  0.834, Validation Accuracy:  0.887, Loss:  0.594
    Epoch   0 Batch 1267/1378 - Train Accuracy:  0.844, Validation Accuracy:  0.877, Loss:  0.599
    Epoch   0 Batch 1268/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.880, Loss:  0.568
    Epoch   0 Batch 1269/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.879, Loss:  0.544
    Epoch   0 Batch 1270/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.869, Loss:  0.585
    Epoch   0 Batch 1271/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.867, Loss:  0.518
    Epoch   0 Batch 1272/1378 - Train Accuracy:  0.849, Validation Accuracy:  0.860, Loss:  0.510
    Epoch   0 Batch 1273/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.880, Loss:  0.545
    Epoch   0 Batch 1274/1378 - Train Accuracy:  0.848, Validation Accuracy:  0.870, Loss:  0.560
    Epoch   0 Batch 1275/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.876, Loss:  0.524
    Epoch   0 Batch 1276/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.889, Loss:  0.583
    Epoch   0 Batch 1277/1378 - Train Accuracy:  0.866, Validation Accuracy:  0.869, Loss:  0.493
    Epoch   0 Batch 1278/1378 - Train Accuracy:  0.839, Validation Accuracy:  0.860, Loss:  0.513
    Epoch   0 Batch 1279/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.862, Loss:  0.579
    Epoch   0 Batch 1280/1378 - Train Accuracy:  0.857, Validation Accuracy:  0.857, Loss:  0.506
    Epoch   0 Batch 1281/1378 - Train Accuracy:  0.846, Validation Accuracy:  0.855, Loss:  0.551
    Epoch   0 Batch 1282/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.851, Loss:  0.502
    Epoch   0 Batch 1283/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.862, Loss:  0.502
    Epoch   0 Batch 1284/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.880, Loss:  0.526
    Epoch   0 Batch 1285/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.874, Loss:  0.471
    Epoch   0 Batch 1286/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.861, Loss:  0.533
    Epoch   0 Batch 1287/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.860, Loss:  0.527
    Epoch   0 Batch 1288/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.858, Loss:  0.522
    Epoch   0 Batch 1289/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.838, Loss:  0.556
    Epoch   0 Batch 1290/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.855, Loss:  0.500
    Epoch   0 Batch 1291/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.843, Loss:  0.544
    Epoch   0 Batch 1292/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.842, Loss:  0.547
    Epoch   0 Batch 1293/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.865, Loss:  0.521
    Epoch   0 Batch 1294/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.872, Loss:  0.553
    Epoch   0 Batch 1295/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.879, Loss:  0.541
    Epoch   0 Batch 1296/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.873, Loss:  0.526
    Epoch   0 Batch 1297/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.890, Loss:  0.500
    Epoch   0 Batch 1298/1378 - Train Accuracy:  0.846, Validation Accuracy:  0.881, Loss:  0.578
    Epoch   0 Batch 1299/1378 - Train Accuracy:  0.850, Validation Accuracy:  0.871, Loss:  0.528
    Epoch   0 Batch 1300/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.879, Loss:  0.533
    Epoch   0 Batch 1301/1378 - Train Accuracy:  0.810, Validation Accuracy:  0.860, Loss:  0.545
    Epoch   0 Batch 1302/1378 - Train Accuracy:  0.836, Validation Accuracy:  0.853, Loss:  0.555
    Epoch   0 Batch 1303/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.862, Loss:  0.540
    Epoch   0 Batch 1304/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.849, Loss:  0.543
    Epoch   0 Batch 1305/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.855, Loss:  0.547
    Epoch   0 Batch 1306/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.852, Loss:  0.560
    Epoch   0 Batch 1307/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.839, Loss:  0.576
    Epoch   0 Batch 1308/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.840, Loss:  0.523
    Epoch   0 Batch 1309/1378 - Train Accuracy:  0.856, Validation Accuracy:  0.835, Loss:  0.485
    Epoch   0 Batch 1310/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.844, Loss:  0.516
    Epoch   0 Batch 1311/1378 - Train Accuracy:  0.834, Validation Accuracy:  0.865, Loss:  0.532
    Epoch   0 Batch 1312/1378 - Train Accuracy:  0.855, Validation Accuracy:  0.868, Loss:  0.488
    Epoch   0 Batch 1313/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.875, Loss:  0.487
    Epoch   0 Batch 1314/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.869, Loss:  0.537
    Epoch   0 Batch 1315/1378 - Train Accuracy:  0.845, Validation Accuracy:  0.876, Loss:  0.504
    Epoch   0 Batch 1316/1378 - Train Accuracy:  0.831, Validation Accuracy:  0.878, Loss:  0.550
    Epoch   0 Batch 1317/1378 - Train Accuracy:  0.856, Validation Accuracy:  0.875, Loss:  0.553
    Epoch   0 Batch 1318/1378 - Train Accuracy:  0.859, Validation Accuracy:  0.861, Loss:  0.564
    Epoch   0 Batch 1319/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.860, Loss:  0.597
    Epoch   0 Batch 1320/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.857, Loss:  0.481
    Epoch   0 Batch 1321/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.849, Loss:  0.535
    Epoch   0 Batch 1322/1378 - Train Accuracy:  0.831, Validation Accuracy:  0.860, Loss:  0.548
    Epoch   0 Batch 1323/1378 - Train Accuracy:  0.843, Validation Accuracy:  0.859, Loss:  0.563
    Epoch   0 Batch 1324/1378 - Train Accuracy:  0.816, Validation Accuracy:  0.867, Loss:  0.500
    Epoch   0 Batch 1325/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.861, Loss:  0.520
    Epoch   0 Batch 1326/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.850, Loss:  0.533
    Epoch   0 Batch 1327/1378 - Train Accuracy:  0.822, Validation Accuracy:  0.856, Loss:  0.635
    Epoch   0 Batch 1328/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.857, Loss:  0.510
    Epoch   0 Batch 1329/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.850, Loss:  0.533
    Epoch   0 Batch 1330/1378 - Train Accuracy:  0.848, Validation Accuracy:  0.849, Loss:  0.547
    Epoch   0 Batch 1331/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.843, Loss:  0.506
    Epoch   0 Batch 1332/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.827, Loss:  0.499
    Epoch   0 Batch 1333/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.854, Loss:  0.511
    Epoch   0 Batch 1334/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.844, Loss:  0.536
    Epoch   0 Batch 1335/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.848, Loss:  0.536
    Epoch   0 Batch 1336/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.844, Loss:  0.537
    Epoch   0 Batch 1337/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.845, Loss:  0.578
    Epoch   0 Batch 1338/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.845, Loss:  0.541
    Epoch   0 Batch 1339/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.853, Loss:  0.502
    Epoch   0 Batch 1340/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.860, Loss:  0.553
    Epoch   0 Batch 1341/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.865, Loss:  0.536
    Epoch   0 Batch 1342/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.849, Loss:  0.521
    Epoch   0 Batch 1343/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.854, Loss:  0.544
    Epoch   0 Batch 1344/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.849, Loss:  0.533
    Epoch   0 Batch 1345/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.845, Loss:  0.488
    Epoch   0 Batch 1346/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.848, Loss:  0.479
    Epoch   0 Batch 1347/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.860, Loss:  0.495
    Epoch   0 Batch 1348/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.865, Loss:  0.519
    Epoch   0 Batch 1349/1378 - Train Accuracy:  0.873, Validation Accuracy:  0.863, Loss:  0.507
    Epoch   0 Batch 1350/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.875, Loss:  0.511
    Epoch   0 Batch 1351/1378 - Train Accuracy:  0.842, Validation Accuracy:  0.891, Loss:  0.511
    Epoch   0 Batch 1352/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.888, Loss:  0.515
    Epoch   0 Batch 1353/1378 - Train Accuracy:  0.840, Validation Accuracy:  0.878, Loss:  0.527
    Epoch   0 Batch 1354/1378 - Train Accuracy:  0.873, Validation Accuracy:  0.878, Loss:  0.558
    Epoch   0 Batch 1355/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.879, Loss:  0.491
    Epoch   0 Batch 1356/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.867, Loss:  0.569
    Epoch   0 Batch 1357/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.866, Loss:  0.512
    Epoch   0 Batch 1358/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.865, Loss:  0.547
    Epoch   0 Batch 1359/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.860, Loss:  0.521
    Epoch   0 Batch 1360/1378 - Train Accuracy:  0.855, Validation Accuracy:  0.849, Loss:  0.535
    Epoch   0 Batch 1361/1378 - Train Accuracy:  0.829, Validation Accuracy:  0.862, Loss:  0.521
    Epoch   0 Batch 1362/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.884, Loss:  0.548
    Epoch   0 Batch 1363/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.860, Loss:  0.524
    Epoch   0 Batch 1364/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.861, Loss:  0.532
    Epoch   0 Batch 1365/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.865, Loss:  0.549
    Epoch   0 Batch 1366/1378 - Train Accuracy:  0.856, Validation Accuracy:  0.869, Loss:  0.559
    Epoch   0 Batch 1367/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.873, Loss:  0.515
    Epoch   0 Batch 1368/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.866, Loss:  0.460
    Epoch   0 Batch 1369/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.872, Loss:  0.469
    Epoch   0 Batch 1370/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.864, Loss:  0.547
    Epoch   0 Batch 1371/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.873, Loss:  0.502
    Epoch   0 Batch 1372/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.865, Loss:  0.507
    Epoch   0 Batch 1373/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.879, Loss:  0.510
    Epoch   0 Batch 1374/1378 - Train Accuracy:  0.848, Validation Accuracy:  0.866, Loss:  0.508
    Epoch   0 Batch 1375/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.859, Loss:  0.489
    Epoch   0 Batch 1376/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.869, Loss:  0.475
    Epoch   1 Batch    0/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.879, Loss:  0.508
    Epoch   1 Batch    1/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.890, Loss:  0.514
    Epoch   1 Batch    2/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.890, Loss:  0.531
    Epoch   1 Batch    3/1378 - Train Accuracy:  0.834, Validation Accuracy:  0.881, Loss:  0.546
    Epoch   1 Batch    4/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.871, Loss:  0.499
    Epoch   1 Batch    5/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.862, Loss:  0.540
    Epoch   1 Batch    6/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.872, Loss:  0.465
    Epoch   1 Batch    7/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.873, Loss:  0.533
    Epoch   1 Batch    8/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.864, Loss:  0.522
    Epoch   1 Batch    9/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.876, Loss:  0.514
    Epoch   1 Batch   10/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.864, Loss:  0.480
    Epoch   1 Batch   11/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.863, Loss:  0.484
    Epoch   1 Batch   12/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.870, Loss:  0.521
    Epoch   1 Batch   13/1378 - Train Accuracy:  0.861, Validation Accuracy:  0.880, Loss:  0.520
    Epoch   1 Batch   14/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.873, Loss:  0.506
    Epoch   1 Batch   15/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.874, Loss:  0.519
    Epoch   1 Batch   16/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.874, Loss:  0.521
    Epoch   1 Batch   17/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.876, Loss:  0.508
    Epoch   1 Batch   18/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.879, Loss:  0.477
    Epoch   1 Batch   19/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.880, Loss:  0.472
    Epoch   1 Batch   20/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.879, Loss:  0.521
    Epoch   1 Batch   21/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.879, Loss:  0.538
    Epoch   1 Batch   22/1378 - Train Accuracy:  0.851, Validation Accuracy:  0.873, Loss:  0.516
    Epoch   1 Batch   23/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.870, Loss:  0.460
    Epoch   1 Batch   24/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.875, Loss:  0.490
    Epoch   1 Batch   25/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.875, Loss:  0.516
    Epoch   1 Batch   26/1378 - Train Accuracy:  0.873, Validation Accuracy:  0.867, Loss:  0.490
    Epoch   1 Batch   27/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.869, Loss:  0.513
    Epoch   1 Batch   28/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.865, Loss:  0.530
    Epoch   1 Batch   29/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.852, Loss:  0.533
    Epoch   1 Batch   30/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.852, Loss:  0.505
    Epoch   1 Batch   31/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.880, Loss:  0.534
    Epoch   1 Batch   32/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.891, Loss:  0.522
    Epoch   1 Batch   33/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.892, Loss:  0.491
    Epoch   1 Batch   34/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.891, Loss:  0.511
    Epoch   1 Batch   35/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.897, Loss:  0.452
    Epoch   1 Batch   36/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.899, Loss:  0.448
    Epoch   1 Batch   37/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.899, Loss:  0.481
    Epoch   1 Batch   38/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.894, Loss:  0.472
    Epoch   1 Batch   39/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.880, Loss:  0.476
    Epoch   1 Batch   40/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.883, Loss:  0.510
    Epoch   1 Batch   41/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.889, Loss:  0.485
    Epoch   1 Batch   42/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.896, Loss:  0.475
    Epoch   1 Batch   43/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.894, Loss:  0.482
    Epoch   1 Batch   44/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.893, Loss:  0.482
    Epoch   1 Batch   45/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.892, Loss:  0.495
    Epoch   1 Batch   46/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.886, Loss:  0.444
    Epoch   1 Batch   47/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.877, Loss:  0.553
    Epoch   1 Batch   48/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.899, Loss:  0.515
    Epoch   1 Batch   49/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.893, Loss:  0.566
    Epoch   1 Batch   50/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.890, Loss:  0.486
    Epoch   1 Batch   51/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.888, Loss:  0.493
    Epoch   1 Batch   52/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.885, Loss:  0.481
    Epoch   1 Batch   53/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.898, Loss:  0.542
    Epoch   1 Batch   54/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.888, Loss:  0.512
    Epoch   1 Batch   55/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.894, Loss:  0.495
    Epoch   1 Batch   56/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.891, Loss:  0.499
    Epoch   1 Batch   57/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.895, Loss:  0.512
    Epoch   1 Batch   58/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.888, Loss:  0.533
    Epoch   1 Batch   59/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.882, Loss:  0.493
    Epoch   1 Batch   60/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.885, Loss:  0.557
    Epoch   1 Batch   61/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.879, Loss:  0.507
    Epoch   1 Batch   62/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.886, Loss:  0.521
    Epoch   1 Batch   63/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.555
    Epoch   1 Batch   64/1378 - Train Accuracy:  0.854, Validation Accuracy:  0.885, Loss:  0.578
    Epoch   1 Batch   65/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.893, Loss:  0.485
    Epoch   1 Batch   66/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.895, Loss:  0.545
    Epoch   1 Batch   67/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.886, Loss:  0.527
    Epoch   1 Batch   68/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.892, Loss:  0.514
    Epoch   1 Batch   69/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.892, Loss:  0.505
    Epoch   1 Batch   70/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.893, Loss:  0.506
    Epoch   1 Batch   71/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.878, Loss:  0.466
    Epoch   1 Batch   72/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.869, Loss:  0.501
    Epoch   1 Batch   73/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.867, Loss:  0.470
    Epoch   1 Batch   74/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.857, Loss:  0.451
    Epoch   1 Batch   75/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.853, Loss:  0.523
    Epoch   1 Batch   76/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.854, Loss:  0.499
    Epoch   1 Batch   77/1378 - Train Accuracy:  0.859, Validation Accuracy:  0.865, Loss:  0.469
    Epoch   1 Batch   78/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.859, Loss:  0.527
    Epoch   1 Batch   79/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.868, Loss:  0.528
    Epoch   1 Batch   80/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.865, Loss:  0.502
    Epoch   1 Batch   81/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.862, Loss:  0.437
    Epoch   1 Batch   82/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.862, Loss:  0.494
    Epoch   1 Batch   83/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.862, Loss:  0.473
    Epoch   1 Batch   84/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.850, Loss:  0.496
    Epoch   1 Batch   85/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.849, Loss:  0.455
    Epoch   1 Batch   86/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.855, Loss:  0.467
    Epoch   1 Batch   87/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.865, Loss:  0.502
    Epoch   1 Batch   88/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.871, Loss:  0.528
    Epoch   1 Batch   89/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.864, Loss:  0.448
    Epoch   1 Batch   90/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.868, Loss:  0.500
    Epoch   1 Batch   91/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.868, Loss:  0.469
    Epoch   1 Batch   92/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.864, Loss:  0.482
    Epoch   1 Batch   93/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.870, Loss:  0.453
    Epoch   1 Batch   94/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.872, Loss:  0.504
    Epoch   1 Batch   95/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.862, Loss:  0.492
    Epoch   1 Batch   96/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.866, Loss:  0.496
    Epoch   1 Batch   97/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.883, Loss:  0.477
    Epoch   1 Batch   98/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.878, Loss:  0.466
    Epoch   1 Batch   99/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.873, Loss:  0.484
    Epoch   1 Batch  100/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.871, Loss:  0.495
    Epoch   1 Batch  101/1378 - Train Accuracy:  0.860, Validation Accuracy:  0.862, Loss:  0.509
    Epoch   1 Batch  102/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.875, Loss:  0.576
    Epoch   1 Batch  103/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.879, Loss:  0.487
    Epoch   1 Batch  104/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.888, Loss:  0.525
    Epoch   1 Batch  105/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.895, Loss:  0.427
    Epoch   1 Batch  106/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.890, Loss:  0.494
    Epoch   1 Batch  107/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.892, Loss:  0.484
    Epoch   1 Batch  108/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.889, Loss:  0.496
    Epoch   1 Batch  109/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.536
    Epoch   1 Batch  110/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.875, Loss:  0.467
    Epoch   1 Batch  111/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.887, Loss:  0.456
    Epoch   1 Batch  112/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.898, Loss:  0.504
    Epoch   1 Batch  113/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.900, Loss:  0.491
    Epoch   1 Batch  114/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.891, Loss:  0.570
    Epoch   1 Batch  115/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.884, Loss:  0.514
    Epoch   1 Batch  116/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.885, Loss:  0.455
    Epoch   1 Batch  117/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.877, Loss:  0.467
    Epoch   1 Batch  118/1378 - Train Accuracy:  0.853, Validation Accuracy:  0.872, Loss:  0.508
    Epoch   1 Batch  119/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.887, Loss:  0.523
    Epoch   1 Batch  120/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.887, Loss:  0.538
    Epoch   1 Batch  121/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.886, Loss:  0.497
    Epoch   1 Batch  122/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.890, Loss:  0.552
    Epoch   1 Batch  123/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.885, Loss:  0.468
    Epoch   1 Batch  124/1378 - Train Accuracy:  0.845, Validation Accuracy:  0.878, Loss:  0.535
    Epoch   1 Batch  125/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.874, Loss:  0.489
    Epoch   1 Batch  126/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.864, Loss:  0.512
    Epoch   1 Batch  127/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.862, Loss:  0.526
    Epoch   1 Batch  128/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.861, Loss:  0.520
    Epoch   1 Batch  129/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.856, Loss:  0.506
    Epoch   1 Batch  130/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.854, Loss:  0.506
    Epoch   1 Batch  131/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.864, Loss:  0.469
    Epoch   1 Batch  132/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.865, Loss:  0.519
    Epoch   1 Batch  133/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.869, Loss:  0.535
    Epoch   1 Batch  134/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.876, Loss:  0.499
    Epoch   1 Batch  135/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.882, Loss:  0.516
    Epoch   1 Batch  136/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.885, Loss:  0.523
    Epoch   1 Batch  137/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.892, Loss:  0.481
    Epoch   1 Batch  138/1378 - Train Accuracy:  0.847, Validation Accuracy:  0.878, Loss:  0.526
    Epoch   1 Batch  139/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.876, Loss:  0.503
    Epoch   1 Batch  140/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.879, Loss:  0.450
    Epoch   1 Batch  141/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.877, Loss:  0.481
    Epoch   1 Batch  142/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.884, Loss:  0.523
    Epoch   1 Batch  143/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.890, Loss:  0.508
    Epoch   1 Batch  144/1378 - Train Accuracy:  0.861, Validation Accuracy:  0.896, Loss:  0.509
    Epoch   1 Batch  145/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.895, Loss:  0.451
    Epoch   1 Batch  146/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.896, Loss:  0.544
    Epoch   1 Batch  147/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.895, Loss:  0.495
    Epoch   1 Batch  148/1378 - Train Accuracy:  0.858, Validation Accuracy:  0.898, Loss:  0.528
    Epoch   1 Batch  149/1378 - Train Accuracy:  0.852, Validation Accuracy:  0.905, Loss:  0.507
    Epoch   1 Batch  150/1378 - Train Accuracy:  0.848, Validation Accuracy:  0.912, Loss:  0.519
    Epoch   1 Batch  151/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.907, Loss:  0.489
    Epoch   1 Batch  152/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.900, Loss:  0.457
    Epoch   1 Batch  153/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.902, Loss:  0.545
    Epoch   1 Batch  154/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.913, Loss:  0.492
    Epoch   1 Batch  155/1378 - Train Accuracy:  0.854, Validation Accuracy:  0.901, Loss:  0.478
    Epoch   1 Batch  156/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.908, Loss:  0.545
    Epoch   1 Batch  157/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.904, Loss:  0.516
    Epoch   1 Batch  158/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.897, Loss:  0.467
    Epoch   1 Batch  159/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.897, Loss:  0.526
    Epoch   1 Batch  160/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.885, Loss:  0.510
    Epoch   1 Batch  161/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.885, Loss:  0.487
    Epoch   1 Batch  162/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.879, Loss:  0.464
    Epoch   1 Batch  163/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.867, Loss:  0.467
    Epoch   1 Batch  164/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.871, Loss:  0.504
    Epoch   1 Batch  165/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.874, Loss:  0.461
    Epoch   1 Batch  166/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.865, Loss:  0.494
    Epoch   1 Batch  167/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.863, Loss:  0.463
    Epoch   1 Batch  168/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.875, Loss:  0.486
    Epoch   1 Batch  169/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.874, Loss:  0.522
    Epoch   1 Batch  170/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.871, Loss:  0.477
    Epoch   1 Batch  171/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.870, Loss:  0.520
    Epoch   1 Batch  172/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.865, Loss:  0.415
    Epoch   1 Batch  173/1378 - Train Accuracy:  0.861, Validation Accuracy:  0.866, Loss:  0.471
    Epoch   1 Batch  174/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.870, Loss:  0.520
    Epoch   1 Batch  175/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.872, Loss:  0.501
    Epoch   1 Batch  176/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.879, Loss:  0.453
    Epoch   1 Batch  177/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.883, Loss:  0.483
    Epoch   1 Batch  178/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.882, Loss:  0.470
    Epoch   1 Batch  179/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.896, Loss:  0.497
    Epoch   1 Batch  180/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.901, Loss:  0.434
    Epoch   1 Batch  181/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.895, Loss:  0.439
    Epoch   1 Batch  182/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.892, Loss:  0.511
    Epoch   1 Batch  183/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.891, Loss:  0.507
    Epoch   1 Batch  184/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.872, Loss:  0.431
    Epoch   1 Batch  185/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.871, Loss:  0.550
    Epoch   1 Batch  186/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.868, Loss:  0.491
    Epoch   1 Batch  187/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.856, Loss:  0.457
    Epoch   1 Batch  188/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.844, Loss:  0.543
    Epoch   1 Batch  189/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.854, Loss:  0.517
    Epoch   1 Batch  190/1378 - Train Accuracy:  0.866, Validation Accuracy:  0.876, Loss:  0.546
    Epoch   1 Batch  191/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.875, Loss:  0.487
    Epoch   1 Batch  192/1378 - Train Accuracy:  0.855, Validation Accuracy:  0.887, Loss:  0.514
    Epoch   1 Batch  193/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.892, Loss:  0.513
    Epoch   1 Batch  194/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.888, Loss:  0.489
    Epoch   1 Batch  195/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.894, Loss:  0.512
    Epoch   1 Batch  196/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.895, Loss:  0.470
    Epoch   1 Batch  197/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.885, Loss:  0.474
    Epoch   1 Batch  198/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.890, Loss:  0.481
    Epoch   1 Batch  199/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.880, Loss:  0.501
    Epoch   1 Batch  200/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.879, Loss:  0.478
    Epoch   1 Batch  201/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.877, Loss:  0.467
    Epoch   1 Batch  202/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.878, Loss:  0.485
    Epoch   1 Batch  203/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.883, Loss:  0.497
    Epoch   1 Batch  204/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.890, Loss:  0.446
    Epoch   1 Batch  205/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.881, Loss:  0.536
    Epoch   1 Batch  206/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.885, Loss:  0.507
    Epoch   1 Batch  207/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.875, Loss:  0.422
    Epoch   1 Batch  208/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.874, Loss:  0.501
    Epoch   1 Batch  209/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.868, Loss:  0.469
    Epoch   1 Batch  210/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.868, Loss:  0.473
    Epoch   1 Batch  211/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.861, Loss:  0.544
    Epoch   1 Batch  212/1378 - Train Accuracy:  0.841, Validation Accuracy:  0.861, Loss:  0.501
    Epoch   1 Batch  213/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.865, Loss:  0.579
    Epoch   1 Batch  214/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.876, Loss:  0.442
    Epoch   1 Batch  215/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.880, Loss:  0.500
    Epoch   1 Batch  216/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.873, Loss:  0.471
    Epoch   1 Batch  217/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.871, Loss:  0.440
    Epoch   1 Batch  218/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.880, Loss:  0.487
    Epoch   1 Batch  219/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.873, Loss:  0.465
    Epoch   1 Batch  220/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.880, Loss:  0.487
    Epoch   1 Batch  221/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.880, Loss:  0.499
    Epoch   1 Batch  222/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.887, Loss:  0.446
    Epoch   1 Batch  223/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.892, Loss:  0.494
    Epoch   1 Batch  224/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.891, Loss:  0.481
    Epoch   1 Batch  225/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.897, Loss:  0.502
    Epoch   1 Batch  226/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.895, Loss:  0.511
    Epoch   1 Batch  227/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.897, Loss:  0.504
    Epoch   1 Batch  228/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.900, Loss:  0.481
    Epoch   1 Batch  229/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.895, Loss:  0.514
    Epoch   1 Batch  230/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.900, Loss:  0.506
    Epoch   1 Batch  231/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.902, Loss:  0.450
    Epoch   1 Batch  232/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.897, Loss:  0.493
    Epoch   1 Batch  233/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.890, Loss:  0.510
    Epoch   1 Batch  234/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.885, Loss:  0.441
    Epoch   1 Batch  235/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.881, Loss:  0.450
    Epoch   1 Batch  236/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.881, Loss:  0.546
    Epoch   1 Batch  237/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.882, Loss:  0.503
    Epoch   1 Batch  238/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.884, Loss:  0.508
    Epoch   1 Batch  239/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.876, Loss:  0.490
    Epoch   1 Batch  240/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.872, Loss:  0.454
    Epoch   1 Batch  241/1378 - Train Accuracy:  0.869, Validation Accuracy:  0.865, Loss:  0.492
    Epoch   1 Batch  242/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.877, Loss:  0.482
    Epoch   1 Batch  243/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.875, Loss:  0.473
    Epoch   1 Batch  244/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.879, Loss:  0.454
    Epoch   1 Batch  245/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.885, Loss:  0.458
    Epoch   1 Batch  246/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.889, Loss:  0.457
    Epoch   1 Batch  247/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.899, Loss:  0.529
    Epoch   1 Batch  248/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.890, Loss:  0.457
    Epoch   1 Batch  249/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.898, Loss:  0.476
    Epoch   1 Batch  250/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.888, Loss:  0.515
    Epoch   1 Batch  251/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.875, Loss:  0.460
    Epoch   1 Batch  252/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.891, Loss:  0.470
    Epoch   1 Batch  253/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.894, Loss:  0.460
    Epoch   1 Batch  254/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.892, Loss:  0.497
    Epoch   1 Batch  255/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.887, Loss:  0.481
    Epoch   1 Batch  256/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.899, Loss:  0.499
    Epoch   1 Batch  257/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.898, Loss:  0.466
    Epoch   1 Batch  258/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.892, Loss:  0.453
    Epoch   1 Batch  259/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.888, Loss:  0.501
    Epoch   1 Batch  260/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.881, Loss:  0.466
    Epoch   1 Batch  261/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.876, Loss:  0.527
    Epoch   1 Batch  262/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.879, Loss:  0.518
    Epoch   1 Batch  263/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.879, Loss:  0.506
    Epoch   1 Batch  264/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.882, Loss:  0.495
    Epoch   1 Batch  265/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.888, Loss:  0.436
    Epoch   1 Batch  266/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.884, Loss:  0.447
    Epoch   1 Batch  267/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.888, Loss:  0.482
    Epoch   1 Batch  268/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.900, Loss:  0.494
    Epoch   1 Batch  269/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.913, Loss:  0.484
    Epoch   1 Batch  270/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.908, Loss:  0.398
    Epoch   1 Batch  271/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.914, Loss:  0.489
    Epoch   1 Batch  272/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.904, Loss:  0.456
    Epoch   1 Batch  273/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.891, Loss:  0.442
    Epoch   1 Batch  274/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.896, Loss:  0.470
    Epoch   1 Batch  275/1378 - Train Accuracy:  0.849, Validation Accuracy:  0.883, Loss:  0.458
    Epoch   1 Batch  276/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.889, Loss:  0.451
    Epoch   1 Batch  277/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.880, Loss:  0.498
    Epoch   1 Batch  278/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.880, Loss:  0.413
    Epoch   1 Batch  279/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.881, Loss:  0.505
    Epoch   1 Batch  280/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.881, Loss:  0.547
    Epoch   1 Batch  281/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.872, Loss:  0.475
    Epoch   1 Batch  282/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.881, Loss:  0.446
    Epoch   1 Batch  283/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.884, Loss:  0.477
    Epoch   1 Batch  284/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.898, Loss:  0.451
    Epoch   1 Batch  285/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.909, Loss:  0.461
    Epoch   1 Batch  286/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.903, Loss:  0.460
    Epoch   1 Batch  287/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.903, Loss:  0.446
    Epoch   1 Batch  288/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.888, Loss:  0.487
    Epoch   1 Batch  289/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.885, Loss:  0.455
    Epoch   1 Batch  290/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.886, Loss:  0.501
    Epoch   1 Batch  291/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.880, Loss:  0.504
    Epoch   1 Batch  292/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.865, Loss:  0.474
    Epoch   1 Batch  293/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.860, Loss:  0.494
    Epoch   1 Batch  294/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.861, Loss:  0.409
    Epoch   1 Batch  295/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.860, Loss:  0.477
    Epoch   1 Batch  296/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.855, Loss:  0.459
    Epoch   1 Batch  297/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.857, Loss:  0.455
    Epoch   1 Batch  298/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.870, Loss:  0.521
    Epoch   1 Batch  299/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.897, Loss:  0.504
    Epoch   1 Batch  300/1378 - Train Accuracy:  0.877, Validation Accuracy:  0.888, Loss:  0.459
    Epoch   1 Batch  301/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.895, Loss:  0.482
    Epoch   1 Batch  302/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.901, Loss:  0.447
    Epoch   1 Batch  303/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.901, Loss:  0.442
    Epoch   1 Batch  304/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.897, Loss:  0.444
    Epoch   1 Batch  305/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.898, Loss:  0.504
    Epoch   1 Batch  306/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.878, Loss:  0.458
    Epoch   1 Batch  307/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.881, Loss:  0.462
    Epoch   1 Batch  308/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.875, Loss:  0.502
    Epoch   1 Batch  309/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.875, Loss:  0.418
    Epoch   1 Batch  310/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.875, Loss:  0.486
    Epoch   1 Batch  311/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.866, Loss:  0.472
    Epoch   1 Batch  312/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.869, Loss:  0.470
    Epoch   1 Batch  313/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.865, Loss:  0.489
    Epoch   1 Batch  314/1378 - Train Accuracy:  0.880, Validation Accuracy:  0.872, Loss:  0.473
    Epoch   1 Batch  315/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.883, Loss:  0.455
    Epoch   1 Batch  316/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.891, Loss:  0.460
    Epoch   1 Batch  317/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.888, Loss:  0.479
    Epoch   1 Batch  318/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.887, Loss:  0.474
    Epoch   1 Batch  319/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.886, Loss:  0.533
    Epoch   1 Batch  320/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.894, Loss:  0.490
    Epoch   1 Batch  321/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.906, Loss:  0.455
    Epoch   1 Batch  322/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.906, Loss:  0.528
    Epoch   1 Batch  323/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.904, Loss:  0.502
    Epoch   1 Batch  324/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.907, Loss:  0.483
    Epoch   1 Batch  325/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.914, Loss:  0.493
    Epoch   1 Batch  326/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.907, Loss:  0.443
    Epoch   1 Batch  327/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.904, Loss:  0.426
    Epoch   1 Batch  328/1378 - Train Accuracy:  0.871, Validation Accuracy:  0.897, Loss:  0.497
    Epoch   1 Batch  329/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.902, Loss:  0.445
    Epoch   1 Batch  330/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.902, Loss:  0.471
    Epoch   1 Batch  331/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.900, Loss:  0.469
    Epoch   1 Batch  332/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.896, Loss:  0.451
    Epoch   1 Batch  333/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.902, Loss:  0.453
    Epoch   1 Batch  334/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.896, Loss:  0.500
    Epoch   1 Batch  335/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.902, Loss:  0.465
    Epoch   1 Batch  336/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.917, Loss:  0.467
    Epoch   1 Batch  337/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.917, Loss:  0.439
    Epoch   1 Batch  338/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.920, Loss:  0.427
    Epoch   1 Batch  339/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.911, Loss:  0.486
    Epoch   1 Batch  340/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.913, Loss:  0.411
    Epoch   1 Batch  341/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.906, Loss:  0.492
    Epoch   1 Batch  342/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.916, Loss:  0.467
    Epoch   1 Batch  343/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.905, Loss:  0.474
    Epoch   1 Batch  344/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.905, Loss:  0.515
    Epoch   1 Batch  345/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.903, Loss:  0.500
    Epoch   1 Batch  346/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.907, Loss:  0.443
    Epoch   1 Batch  347/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.895, Loss:  0.511
    Epoch   1 Batch  348/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.884, Loss:  0.487
    Epoch   1 Batch  349/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.875, Loss:  0.510
    Epoch   1 Batch  350/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.876, Loss:  0.473
    Epoch   1 Batch  351/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.872, Loss:  0.406
    Epoch   1 Batch  352/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.863, Loss:  0.462
    Epoch   1 Batch  353/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.857, Loss:  0.440
    Epoch   1 Batch  354/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.855, Loss:  0.465
    Epoch   1 Batch  355/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.867, Loss:  0.463
    Epoch   1 Batch  356/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.874, Loss:  0.397
    Epoch   1 Batch  357/1378 - Train Accuracy:  0.863, Validation Accuracy:  0.874, Loss:  0.484
    Epoch   1 Batch  358/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.873, Loss:  0.481
    Epoch   1 Batch  359/1378 - Train Accuracy:  0.862, Validation Accuracy:  0.884, Loss:  0.495
    Epoch   1 Batch  360/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.885, Loss:  0.487
    Epoch   1 Batch  361/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.891, Loss:  0.473
    Epoch   1 Batch  362/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.904, Loss:  0.524
    Epoch   1 Batch  363/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.901, Loss:  0.431
    Epoch   1 Batch  364/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.902, Loss:  0.439
    Epoch   1 Batch  365/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.896, Loss:  0.439
    Epoch   1 Batch  366/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.902, Loss:  0.437
    Epoch   1 Batch  367/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.902, Loss:  0.438
    Epoch   1 Batch  368/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.910, Loss:  0.423
    Epoch   1 Batch  369/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.894, Loss:  0.465
    Epoch   1 Batch  370/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.888, Loss:  0.422
    Epoch   1 Batch  371/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.896, Loss:  0.511
    Epoch   1 Batch  372/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.896, Loss:  0.495
    Epoch   1 Batch  373/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.895, Loss:  0.468
    Epoch   1 Batch  374/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.896, Loss:  0.485
    Epoch   1 Batch  375/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.896, Loss:  0.451
    Epoch   1 Batch  376/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.891, Loss:  0.446
    Epoch   1 Batch  377/1378 - Train Accuracy:  0.884, Validation Accuracy:  0.892, Loss:  0.415
    Epoch   1 Batch  378/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.906, Loss:  0.491
    Epoch   1 Batch  379/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.907, Loss:  0.475
    Epoch   1 Batch  380/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.900, Loss:  0.444
    Epoch   1 Batch  381/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.898, Loss:  0.500
    Epoch   1 Batch  382/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.889, Loss:  0.509
    Epoch   1 Batch  383/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.889, Loss:  0.465
    Epoch   1 Batch  384/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.901, Loss:  0.452
    Epoch   1 Batch  385/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.896, Loss:  0.454
    Epoch   1 Batch  386/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.896, Loss:  0.473
    Epoch   1 Batch  387/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.909, Loss:  0.480
    Epoch   1 Batch  388/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.910, Loss:  0.498
    Epoch   1 Batch  389/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.911, Loss:  0.519
    Epoch   1 Batch  390/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.914, Loss:  0.492
    Epoch   1 Batch  391/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.914, Loss:  0.466
    Epoch   1 Batch  392/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.895, Loss:  0.499
    Epoch   1 Batch  393/1378 - Train Accuracy:  0.870, Validation Accuracy:  0.894, Loss:  0.481
    Epoch   1 Batch  394/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.895, Loss:  0.453
    Epoch   1 Batch  395/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.900, Loss:  0.444
    Epoch   1 Batch  396/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.892, Loss:  0.472
    Epoch   1 Batch  397/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.890, Loss:  0.452
    Epoch   1 Batch  398/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.890, Loss:  0.492
    Epoch   1 Batch  399/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.889, Loss:  0.457
    Epoch   1 Batch  400/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.899, Loss:  0.469
    Epoch   1 Batch  401/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.899, Loss:  0.505
    Epoch   1 Batch  402/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.907, Loss:  0.448
    Epoch   1 Batch  403/1378 - Train Accuracy:  0.961, Validation Accuracy:  0.904, Loss:  0.474
    Epoch   1 Batch  404/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.905, Loss:  0.476
    Epoch   1 Batch  405/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.908, Loss:  0.451
    Epoch   1 Batch  406/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.904, Loss:  0.520
    Epoch   1 Batch  407/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.904, Loss:  0.458
    Epoch   1 Batch  408/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.900, Loss:  0.449
    Epoch   1 Batch  409/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.897, Loss:  0.469
    Epoch   1 Batch  410/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.905, Loss:  0.431
    Epoch   1 Batch  411/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.917, Loss:  0.445
    Epoch   1 Batch  412/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.911, Loss:  0.446
    Epoch   1 Batch  413/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.900, Loss:  0.484
    Epoch   1 Batch  414/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.901, Loss:  0.441
    Epoch   1 Batch  415/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.907, Loss:  0.451
    Epoch   1 Batch  416/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.903, Loss:  0.425
    Epoch   1 Batch  417/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.905, Loss:  0.478
    Epoch   1 Batch  418/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.899, Loss:  0.437
    Epoch   1 Batch  419/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.890, Loss:  0.497
    Epoch   1 Batch  420/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.884, Loss:  0.515
    Epoch   1 Batch  421/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.882, Loss:  0.439
    Epoch   1 Batch  422/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.873, Loss:  0.469
    Epoch   1 Batch  423/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.867, Loss:  0.492
    Epoch   1 Batch  424/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.872, Loss:  0.497
    Epoch   1 Batch  425/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.866, Loss:  0.432
    Epoch   1 Batch  426/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.865, Loss:  0.454
    Epoch   1 Batch  427/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.865, Loss:  0.512
    Epoch   1 Batch  428/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.873, Loss:  0.413
    Epoch   1 Batch  429/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.878, Loss:  0.463
    Epoch   1 Batch  430/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.880, Loss:  0.518
    Epoch   1 Batch  431/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.893, Loss:  0.479
    Epoch   1 Batch  432/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.893, Loss:  0.504
    Epoch   1 Batch  433/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.889, Loss:  0.488
    Epoch   1 Batch  434/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.888, Loss:  0.404
    Epoch   1 Batch  435/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.903, Loss:  0.449
    Epoch   1 Batch  436/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.890, Loss:  0.450
    Epoch   1 Batch  437/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.899, Loss:  0.451
    Epoch   1 Batch  438/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.901, Loss:  0.447
    Epoch   1 Batch  439/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.907, Loss:  0.495
    Epoch   1 Batch  440/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.904, Loss:  0.486
    Epoch   1 Batch  441/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.902, Loss:  0.469
    Epoch   1 Batch  442/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.889, Loss:  0.450
    Epoch   1 Batch  443/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.879, Loss:  0.447
    Epoch   1 Batch  444/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.873, Loss:  0.468
    Epoch   1 Batch  445/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.875, Loss:  0.442
    Epoch   1 Batch  446/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.879, Loss:  0.429
    Epoch   1 Batch  447/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.879, Loss:  0.488
    Epoch   1 Batch  448/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.892, Loss:  0.420
    Epoch   1 Batch  449/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.898, Loss:  0.488
    Epoch   1 Batch  450/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.894, Loss:  0.445
    Epoch   1 Batch  451/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.900, Loss:  0.472
    Epoch   1 Batch  452/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.900, Loss:  0.499
    Epoch   1 Batch  453/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.900, Loss:  0.462
    Epoch   1 Batch  454/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.896, Loss:  0.450
    Epoch   1 Batch  455/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.893, Loss:  0.456
    Epoch   1 Batch  456/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.900, Loss:  0.467
    Epoch   1 Batch  457/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.888, Loss:  0.473
    Epoch   1 Batch  458/1378 - Train Accuracy:  0.866, Validation Accuracy:  0.880, Loss:  0.521
    Epoch   1 Batch  459/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.890, Loss:  0.458
    Epoch   1 Batch  460/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.897, Loss:  0.434
    Epoch   1 Batch  461/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.908, Loss:  0.418
    Epoch   1 Batch  462/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.917, Loss:  0.426
    Epoch   1 Batch  463/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.911, Loss:  0.456
    Epoch   1 Batch  464/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.895, Loss:  0.452
    Epoch   1 Batch  465/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.901, Loss:  0.491
    Epoch   1 Batch  466/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.900, Loss:  0.475
    Epoch   1 Batch  467/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.900, Loss:  0.474
    Epoch   1 Batch  468/1378 - Train Accuracy:  0.967, Validation Accuracy:  0.895, Loss:  0.398
    Epoch   1 Batch  469/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.895, Loss:  0.412
    Epoch   1 Batch  470/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.889, Loss:  0.410
    Epoch   1 Batch  471/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.906, Loss:  0.464
    Epoch   1 Batch  472/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.910, Loss:  0.439
    Epoch   1 Batch  473/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.916, Loss:  0.437
    Epoch   1 Batch  474/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.914, Loss:  0.474
    Epoch   1 Batch  475/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.910, Loss:  0.429
    Epoch   1 Batch  476/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.902, Loss:  0.485
    Epoch   1 Batch  477/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.900, Loss:  0.444
    Epoch   1 Batch  478/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.900, Loss:  0.453
    Epoch   1 Batch  479/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.905, Loss:  0.479
    Epoch   1 Batch  480/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.911, Loss:  0.443
    Epoch   1 Batch  481/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.912, Loss:  0.476
    Epoch   1 Batch  482/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.905, Loss:  0.465
    Epoch   1 Batch  483/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.905, Loss:  0.438
    Epoch   1 Batch  484/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.906, Loss:  0.448
    Epoch   1 Batch  485/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.908, Loss:  0.455
    Epoch   1 Batch  486/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.906, Loss:  0.446
    Epoch   1 Batch  487/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.902, Loss:  0.447
    Epoch   1 Batch  488/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.900, Loss:  0.490
    Epoch   1 Batch  489/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.907, Loss:  0.460
    Epoch   1 Batch  490/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.907, Loss:  0.436
    Epoch   1 Batch  491/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.902, Loss:  0.419
    Epoch   1 Batch  492/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.894, Loss:  0.415
    Epoch   1 Batch  493/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.918, Loss:  0.471
    Epoch   1 Batch  494/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.916, Loss:  0.471
    Epoch   1 Batch  495/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.925, Loss:  0.485
    Epoch   1 Batch  496/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.921, Loss:  0.447
    Epoch   1 Batch  497/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.915, Loss:  0.412
    Epoch   1 Batch  498/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.910, Loss:  0.449
    Epoch   1 Batch  499/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.915, Loss:  0.470
    Epoch   1 Batch  500/1378 - Train Accuracy:  0.886, Validation Accuracy:  0.914, Loss:  0.446
    Epoch   1 Batch  501/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.911, Loss:  0.428
    Epoch   1 Batch  502/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.914, Loss:  0.476
    Epoch   1 Batch  503/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.917, Loss:  0.484
    Epoch   1 Batch  504/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.918, Loss:  0.394
    Epoch   1 Batch  505/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.917, Loss:  0.460
    Epoch   1 Batch  506/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.919, Loss:  0.543
    Epoch   1 Batch  507/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.923, Loss:  0.437
    Epoch   1 Batch  508/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.916, Loss:  0.486
    Epoch   1 Batch  509/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.905, Loss:  0.486
    Epoch   1 Batch  510/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.892, Loss:  0.444
    Epoch   1 Batch  511/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.897, Loss:  0.487
    Epoch   1 Batch  512/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.891, Loss:  0.415
    Epoch   1 Batch  513/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.884, Loss:  0.424
    Epoch   1 Batch  514/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.890, Loss:  0.432
    Epoch   1 Batch  515/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.896, Loss:  0.438
    Epoch   1 Batch  516/1378 - Train Accuracy:  0.859, Validation Accuracy:  0.898, Loss:  0.459
    Epoch   1 Batch  517/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.899, Loss:  0.471
    Epoch   1 Batch  518/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.897, Loss:  0.439
    Epoch   1 Batch  519/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.894, Loss:  0.457
    Epoch   1 Batch  520/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.897, Loss:  0.475
    Epoch   1 Batch  521/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.894, Loss:  0.452
    Epoch   1 Batch  522/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.896, Loss:  0.427
    Epoch   1 Batch  523/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.898, Loss:  0.436
    Epoch   1 Batch  524/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.918, Loss:  0.455
    Epoch   1 Batch  525/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.917, Loss:  0.418
    Epoch   1 Batch  526/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.925, Loss:  0.428
    Epoch   1 Batch  527/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.923, Loss:  0.455
    Epoch   1 Batch  528/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.924, Loss:  0.464
    Epoch   1 Batch  529/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.915, Loss:  0.438
    Epoch   1 Batch  530/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.917, Loss:  0.420
    Epoch   1 Batch  531/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.915, Loss:  0.506
    Epoch   1 Batch  532/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.911, Loss:  0.393
    Epoch   1 Batch  533/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.906, Loss:  0.498
    Epoch   1 Batch  534/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.906, Loss:  0.474
    Epoch   1 Batch  535/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.895, Loss:  0.478
    Epoch   1 Batch  536/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.905, Loss:  0.410
    Epoch   1 Batch  537/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.910, Loss:  0.465
    Epoch   1 Batch  538/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.915, Loss:  0.478
    Epoch   1 Batch  539/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.916, Loss:  0.448
    Epoch   1 Batch  540/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.904, Loss:  0.442
    Epoch   1 Batch  541/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.908, Loss:  0.483
    Epoch   1 Batch  542/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.897, Loss:  0.433
    Epoch   1 Batch  543/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.895, Loss:  0.445
    Epoch   1 Batch  544/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.902, Loss:  0.394
    Epoch   1 Batch  545/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.891, Loss:  0.442
    Epoch   1 Batch  546/1378 - Train Accuracy:  0.867, Validation Accuracy:  0.893, Loss:  0.439
    Epoch   1 Batch  547/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.896, Loss:  0.435
    Epoch   1 Batch  548/1378 - Train Accuracy:  0.958, Validation Accuracy:  0.896, Loss:  0.373
    Epoch   1 Batch  549/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.895, Loss:  0.422
    Epoch   1 Batch  550/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.893, Loss:  0.479
    Epoch   1 Batch  551/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.894, Loss:  0.427
    Epoch   1 Batch  552/1378 - Train Accuracy:  0.965, Validation Accuracy:  0.893, Loss:  0.440
    Epoch   1 Batch  553/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.887, Loss:  0.439
    Epoch   1 Batch  554/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.897, Loss:  0.502
    Epoch   1 Batch  555/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.907, Loss:  0.475
    Epoch   1 Batch  556/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.922, Loss:  0.452
    Epoch   1 Batch  557/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.922, Loss:  0.472
    Epoch   1 Batch  558/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.915, Loss:  0.406
    Epoch   1 Batch  559/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.895, Loss:  0.465
    Epoch   1 Batch  560/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.897, Loss:  0.437
    Epoch   1 Batch  561/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.898, Loss:  0.463
    Epoch   1 Batch  562/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.904, Loss:  0.426
    Epoch   1 Batch  563/1378 - Train Accuracy:  0.883, Validation Accuracy:  0.897, Loss:  0.481
    Epoch   1 Batch  564/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.896, Loss:  0.433
    Epoch   1 Batch  565/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.887, Loss:  0.440
    Epoch   1 Batch  566/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.875, Loss:  0.417
    Epoch   1 Batch  567/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.871, Loss:  0.402
    Epoch   1 Batch  568/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.876, Loss:  0.426
    Epoch   1 Batch  569/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.890, Loss:  0.405
    Epoch   1 Batch  570/1378 - Train Accuracy:  0.866, Validation Accuracy:  0.886, Loss:  0.468
    Epoch   1 Batch  571/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.885, Loss:  0.454
    Epoch   1 Batch  572/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.884, Loss:  0.408
    Epoch   1 Batch  573/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.883, Loss:  0.473
    Epoch   1 Batch  574/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.892, Loss:  0.455
    Epoch   1 Batch  575/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.893, Loss:  0.462
    Epoch   1 Batch  576/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.876, Loss:  0.423
    Epoch   1 Batch  577/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.884, Loss:  0.493
    Epoch   1 Batch  578/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.894, Loss:  0.430
    Epoch   1 Batch  579/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.905, Loss:  0.467
    Epoch   1 Batch  580/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.911, Loss:  0.383
    Epoch   1 Batch  581/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.923, Loss:  0.453
    Epoch   1 Batch  582/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.925, Loss:  0.426
    Epoch   1 Batch  583/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.932, Loss:  0.414
    Epoch   1 Batch  584/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.921, Loss:  0.465
    Epoch   1 Batch  585/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.925, Loss:  0.465
    Epoch   1 Batch  586/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.933, Loss:  0.429
    Epoch   1 Batch  587/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.932, Loss:  0.492
    Epoch   1 Batch  588/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.931, Loss:  0.433
    Epoch   1 Batch  589/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.932, Loss:  0.456
    Epoch   1 Batch  590/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.925, Loss:  0.418
    Epoch   1 Batch  591/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.925, Loss:  0.405
    Epoch   1 Batch  592/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.921, Loss:  0.450
    Epoch   1 Batch  593/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.919, Loss:  0.489
    Epoch   1 Batch  594/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.914, Loss:  0.410
    Epoch   1 Batch  595/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.910, Loss:  0.421
    Epoch   1 Batch  596/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.908, Loss:  0.463
    Epoch   1 Batch  597/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.906, Loss:  0.453
    Epoch   1 Batch  598/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.897, Loss:  0.454
    Epoch   1 Batch  599/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.892, Loss:  0.417
    Epoch   1 Batch  600/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.895, Loss:  0.473
    Epoch   1 Batch  601/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.889, Loss:  0.496
    Epoch   1 Batch  602/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.897, Loss:  0.472
    Epoch   1 Batch  603/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.905, Loss:  0.405
    Epoch   1 Batch  604/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.908, Loss:  0.430
    Epoch   1 Batch  605/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.921, Loss:  0.464
    Epoch   1 Batch  606/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.927, Loss:  0.516
    Epoch   1 Batch  607/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.918, Loss:  0.377
    Epoch   1 Batch  608/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.917, Loss:  0.495
    Epoch   1 Batch  609/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.908, Loss:  0.485
    Epoch   1 Batch  610/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.915, Loss:  0.430
    Epoch   1 Batch  611/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.908, Loss:  0.455
    Epoch   1 Batch  612/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.893, Loss:  0.430
    Epoch   1 Batch  613/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.895, Loss:  0.439
    Epoch   1 Batch  614/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.893, Loss:  0.448
    Epoch   1 Batch  615/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.890, Loss:  0.493
    Epoch   1 Batch  616/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.891, Loss:  0.449
    Epoch   1 Batch  617/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.895, Loss:  0.457
    Epoch   1 Batch  618/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.888, Loss:  0.470
    Epoch   1 Batch  619/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.893, Loss:  0.440
    Epoch   1 Batch  620/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.905, Loss:  0.416
    Epoch   1 Batch  621/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.905, Loss:  0.455
    Epoch   1 Batch  622/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.899, Loss:  0.432
    Epoch   1 Batch  623/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.905, Loss:  0.443
    Epoch   1 Batch  624/1378 - Train Accuracy:  0.957, Validation Accuracy:  0.903, Loss:  0.402
    Epoch   1 Batch  625/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.901, Loss:  0.444
    Epoch   1 Batch  626/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.897, Loss:  0.452
    Epoch   1 Batch  627/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.893, Loss:  0.494
    Epoch   1 Batch  628/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.881, Loss:  0.463
    Epoch   1 Batch  629/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.890, Loss:  0.433
    Epoch   1 Batch  630/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.890, Loss:  0.450
    Epoch   1 Batch  631/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.896, Loss:  0.429
    Epoch   1 Batch  632/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.892, Loss:  0.422
    Epoch   1 Batch  633/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.917, Loss:  0.440
    Epoch   1 Batch  634/1378 - Train Accuracy:  0.872, Validation Accuracy:  0.917, Loss:  0.480
    Epoch   1 Batch  635/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.916, Loss:  0.420
    Epoch   1 Batch  636/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.890, Loss:  0.452
    Epoch   1 Batch  637/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.890, Loss:  0.384
    Epoch   1 Batch  638/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.897, Loss:  0.468
    Epoch   1 Batch  639/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.902, Loss:  0.398
    Epoch   1 Batch  640/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.911, Loss:  0.409
    Epoch   1 Batch  641/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.914, Loss:  0.430
    Epoch   1 Batch  642/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.921, Loss:  0.472
    Epoch   1 Batch  643/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.918, Loss:  0.433
    Epoch   1 Batch  644/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.917, Loss:  0.437
    Epoch   1 Batch  645/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.922, Loss:  0.444
    Epoch   1 Batch  646/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.922, Loss:  0.426
    Epoch   1 Batch  647/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.930, Loss:  0.409
    Epoch   1 Batch  648/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.918, Loss:  0.492
    Epoch   1 Batch  649/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.918, Loss:  0.485
    Epoch   1 Batch  650/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.920, Loss:  0.438
    Epoch   1 Batch  651/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.923, Loss:  0.416
    Epoch   1 Batch  652/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.915, Loss:  0.482
    Epoch   1 Batch  653/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.905, Loss:  0.452
    Epoch   1 Batch  654/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.902, Loss:  0.453
    Epoch   1 Batch  655/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.889, Loss:  0.464
    Epoch   1 Batch  656/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.895, Loss:  0.470
    Epoch   1 Batch  657/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.890, Loss:  0.455
    Epoch   1 Batch  658/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.890, Loss:  0.499
    Epoch   1 Batch  659/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.911, Loss:  0.447
    Epoch   1 Batch  660/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.912, Loss:  0.482
    Epoch   1 Batch  661/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.905, Loss:  0.402
    Epoch   1 Batch  662/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.902, Loss:  0.502
    Epoch   1 Batch  663/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.914, Loss:  0.421
    Epoch   1 Batch  664/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.914, Loss:  0.441
    Epoch   1 Batch  665/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.921, Loss:  0.469
    Epoch   1 Batch  666/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.920, Loss:  0.480
    Epoch   1 Batch  667/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.928, Loss:  0.466
    Epoch   1 Batch  668/1378 - Train Accuracy:  0.875, Validation Accuracy:  0.932, Loss:  0.478
    Epoch   1 Batch  669/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.925, Loss:  0.448
    Epoch   1 Batch  670/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.910, Loss:  0.445
    Epoch   1 Batch  671/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.910, Loss:  0.392
    Epoch   1 Batch  672/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.920, Loss:  0.474
    Epoch   1 Batch  673/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.913, Loss:  0.442
    Epoch   1 Batch  674/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.912, Loss:  0.470
    Epoch   1 Batch  675/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.919, Loss:  0.468
    Epoch   1 Batch  676/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.923, Loss:  0.410
    Epoch   1 Batch  677/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.923, Loss:  0.448
    Epoch   1 Batch  678/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.904, Loss:  0.492
    Epoch   1 Batch  679/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.891, Loss:  0.483
    Epoch   1 Batch  680/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.886, Loss:  0.461
    Epoch   1 Batch  681/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.900, Loss:  0.438
    Epoch   1 Batch  682/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.894, Loss:  0.500
    Epoch   1 Batch  683/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.898, Loss:  0.442
    Epoch   1 Batch  684/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.899, Loss:  0.461
    Epoch   1 Batch  685/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.898, Loss:  0.429
    Epoch   1 Batch  686/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.876, Loss:  0.474
    Epoch   1 Batch  687/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.892, Loss:  0.435
    Epoch   1 Batch  688/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.871, Loss:  0.435
    Epoch   1 Batch  689/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.894, Loss:  0.452
    Epoch   1 Batch  690/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.897, Loss:  0.512
    Epoch   1 Batch  691/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.886, Loss:  0.509
    Epoch   1 Batch  692/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.887, Loss:  0.468
    Epoch   1 Batch  693/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.892, Loss:  0.452
    Epoch   1 Batch  694/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.890, Loss:  0.462
    Epoch   1 Batch  695/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.886, Loss:  0.488
    Epoch   1 Batch  696/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.885, Loss:  0.382
    Epoch   1 Batch  697/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.888, Loss:  0.444
    Epoch   1 Batch  698/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.878, Loss:  0.444
    Epoch   1 Batch  699/1378 - Train Accuracy:  0.889, Validation Accuracy:  0.888, Loss:  0.482
    Epoch   1 Batch  700/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.892, Loss:  0.417
    Epoch   1 Batch  701/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.887, Loss:  0.508
    Epoch   1 Batch  702/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.887, Loss:  0.445
    Epoch   1 Batch  703/1378 - Train Accuracy:  0.879, Validation Accuracy:  0.894, Loss:  0.474
    Epoch   1 Batch  704/1378 - Train Accuracy:  0.888, Validation Accuracy:  0.903, Loss:  0.458
    Epoch   1 Batch  705/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.905, Loss:  0.489
    Epoch   1 Batch  706/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.885, Loss:  0.475
    Epoch   1 Batch  707/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.878, Loss:  0.535
    Epoch   1 Batch  708/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.890, Loss:  0.438
    Epoch   1 Batch  709/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.886, Loss:  0.427
    Epoch   1 Batch  710/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.888, Loss:  0.453
    Epoch   1 Batch  711/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.895, Loss:  0.407
    Epoch   1 Batch  712/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.890, Loss:  0.447
    Epoch   1 Batch  713/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.906, Loss:  0.506
    Epoch   1 Batch  714/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.907, Loss:  0.426
    Epoch   1 Batch  715/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.912, Loss:  0.471
    Epoch   1 Batch  716/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.907, Loss:  0.425
    Epoch   1 Batch  717/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.908, Loss:  0.459
    Epoch   1 Batch  718/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.900, Loss:  0.454
    Epoch   1 Batch  719/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.886, Loss:  0.450
    Epoch   1 Batch  720/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.862, Loss:  0.470
    Epoch   1 Batch  721/1378 - Train Accuracy:  0.881, Validation Accuracy:  0.865, Loss:  0.458
    Epoch   1 Batch  722/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.852, Loss:  0.399
    Epoch   1 Batch  723/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.843, Loss:  0.423
    Epoch   1 Batch  724/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.859, Loss:  0.495
    Epoch   1 Batch  725/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.873, Loss:  0.402
    Epoch   1 Batch  726/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.872, Loss:  0.411
    Epoch   1 Batch  727/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.875, Loss:  0.412
    Epoch   1 Batch  728/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.883, Loss:  0.464
    Epoch   1 Batch  729/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.890, Loss:  0.465
    Epoch   1 Batch  730/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.890, Loss:  0.462
    Epoch   1 Batch  731/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.884, Loss:  0.444
    Epoch   1 Batch  732/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.887, Loss:  0.433
    Epoch   1 Batch  733/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.887, Loss:  0.485
    Epoch   1 Batch  734/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.888, Loss:  0.474
    Epoch   1 Batch  735/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.888, Loss:  0.511
    Epoch   1 Batch  736/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.877, Loss:  0.422
    Epoch   1 Batch  737/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.883, Loss:  0.432
    Epoch   1 Batch  738/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.883, Loss:  0.415
    Epoch   1 Batch  739/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.889, Loss:  0.437
    Epoch   1 Batch  740/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.890, Loss:  0.416
    Epoch   1 Batch  741/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.890, Loss:  0.431
    Epoch   1 Batch  742/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.897, Loss:  0.478
    Epoch   1 Batch  743/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.891, Loss:  0.428
    Epoch   1 Batch  744/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.888, Loss:  0.421
    Epoch   1 Batch  745/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.882, Loss:  0.474
    Epoch   1 Batch  746/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.876, Loss:  0.406
    Epoch   1 Batch  747/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.887, Loss:  0.417
    Epoch   1 Batch  748/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.883, Loss:  0.394
    Epoch   1 Batch  749/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.882, Loss:  0.409
    Epoch   1 Batch  750/1378 - Train Accuracy:  0.971, Validation Accuracy:  0.885, Loss:  0.463
    Epoch   1 Batch  751/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.903, Loss:  0.438
    Epoch   1 Batch  752/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.908, Loss:  0.434
    Epoch   1 Batch  753/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.920, Loss:  0.421
    Epoch   1 Batch  754/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.909, Loss:  0.472
    Epoch   1 Batch  755/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.910, Loss:  0.498
    Epoch   1 Batch  756/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.916, Loss:  0.471
    Epoch   1 Batch  757/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.910, Loss:  0.468
    Epoch   1 Batch  758/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.900, Loss:  0.462
    Epoch   1 Batch  759/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.907, Loss:  0.470
    Epoch   1 Batch  760/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.912, Loss:  0.429
    Epoch   1 Batch  761/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.902, Loss:  0.414
    Epoch   1 Batch  762/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.909, Loss:  0.440
    Epoch   1 Batch  763/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.915, Loss:  0.457
    Epoch   1 Batch  764/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.901, Loss:  0.422
    Epoch   1 Batch  765/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.911, Loss:  0.432
    Epoch   1 Batch  766/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.913, Loss:  0.425
    Epoch   1 Batch  767/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.915, Loss:  0.437
    Epoch   1 Batch  768/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.915, Loss:  0.427
    Epoch   1 Batch  769/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.909, Loss:  0.463
    Epoch   1 Batch  770/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.910, Loss:  0.437
    Epoch   1 Batch  771/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.909, Loss:  0.459
    Epoch   1 Batch  772/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.915, Loss:  0.475
    Epoch   1 Batch  773/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.903, Loss:  0.392
    Epoch   1 Batch  774/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.891, Loss:  0.472
    Epoch   1 Batch  775/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.907, Loss:  0.455
    Epoch   1 Batch  776/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.901, Loss:  0.433
    Epoch   1 Batch  777/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.887, Loss:  0.437
    Epoch   1 Batch  778/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.885, Loss:  0.470
    Epoch   1 Batch  779/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.872, Loss:  0.461
    Epoch   1 Batch  780/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.876, Loss:  0.426
    Epoch   1 Batch  781/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.884, Loss:  0.479
    Epoch   1 Batch  782/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.894, Loss:  0.401
    Epoch   1 Batch  783/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.899, Loss:  0.432
    Epoch   1 Batch  784/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.885, Loss:  0.415
    Epoch   1 Batch  785/1378 - Train Accuracy:  0.856, Validation Accuracy:  0.893, Loss:  0.455
    Epoch   1 Batch  786/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.894, Loss:  0.467
    Epoch   1 Batch  787/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.894, Loss:  0.413
    Epoch   1 Batch  788/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.899, Loss:  0.445
    Epoch   1 Batch  789/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.899, Loss:  0.424
    Epoch   1 Batch  790/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.905, Loss:  0.438
    Epoch   1 Batch  791/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.928, Loss:  0.501
    Epoch   1 Batch  792/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.926, Loss:  0.466
    Epoch   1 Batch  793/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.923, Loss:  0.421
    Epoch   1 Batch  794/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.936, Loss:  0.463
    Epoch   1 Batch  795/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.929, Loss:  0.478
    Epoch   1 Batch  796/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.928, Loss:  0.424
    Epoch   1 Batch  797/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.922, Loss:  0.446
    Epoch   1 Batch  798/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.922, Loss:  0.475
    Epoch   1 Batch  799/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.912, Loss:  0.460
    Epoch   1 Batch  800/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.913, Loss:  0.434
    Epoch   1 Batch  801/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.913, Loss:  0.452
    Epoch   1 Batch  802/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.920, Loss:  0.417
    Epoch   1 Batch  803/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.919, Loss:  0.413
    Epoch   1 Batch  804/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.917, Loss:  0.461
    Epoch   1 Batch  805/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.926, Loss:  0.416
    Epoch   1 Batch  806/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.923, Loss:  0.472
    Epoch   1 Batch  807/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.929, Loss:  0.413
    Epoch   1 Batch  808/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.931, Loss:  0.444
    Epoch   1 Batch  809/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.930, Loss:  0.410
    Epoch   1 Batch  810/1378 - Train Accuracy:  0.953, Validation Accuracy:  0.924, Loss:  0.360
    Epoch   1 Batch  811/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.924, Loss:  0.419
    Epoch   1 Batch  812/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.924, Loss:  0.425
    Epoch   1 Batch  813/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.923, Loss:  0.424
    Epoch   1 Batch  814/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.910, Loss:  0.457
    Epoch   1 Batch  815/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.915, Loss:  0.427
    Epoch   1 Batch  816/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.925, Loss:  0.439
    Epoch   1 Batch  817/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.934, Loss:  0.465
    Epoch   1 Batch  818/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.935, Loss:  0.474
    Epoch   1 Batch  819/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.935, Loss:  0.419
    Epoch   1 Batch  820/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.945, Loss:  0.443
    Epoch   1 Batch  821/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.946, Loss:  0.474
    Epoch   1 Batch  822/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.946, Loss:  0.411
    Epoch   1 Batch  823/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.943, Loss:  0.408
    Epoch   1 Batch  824/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.944, Loss:  0.400
    Epoch   1 Batch  825/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.944, Loss:  0.407
    Epoch   1 Batch  826/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.946, Loss:  0.435
    Epoch   1 Batch  827/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.948, Loss:  0.407
    Epoch   1 Batch  828/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.935, Loss:  0.442
    Epoch   1 Batch  829/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.938, Loss:  0.442
    Epoch   1 Batch  830/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.934, Loss:  0.374
    Epoch   1 Batch  831/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.927, Loss:  0.429
    Epoch   1 Batch  832/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.928, Loss:  0.478
    Epoch   1 Batch  833/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.925, Loss:  0.451
    Epoch   1 Batch  834/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.928, Loss:  0.445
    Epoch   1 Batch  835/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.916, Loss:  0.422
    Epoch   1 Batch  836/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.910, Loss:  0.425
    Epoch   1 Batch  837/1378 - Train Accuracy:  0.962, Validation Accuracy:  0.916, Loss:  0.425
    Epoch   1 Batch  838/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.920, Loss:  0.428
    Epoch   1 Batch  839/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.919, Loss:  0.467
    Epoch   1 Batch  840/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.922, Loss:  0.479
    Epoch   1 Batch  841/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.910, Loss:  0.395
    Epoch   1 Batch  842/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.903, Loss:  0.416
    Epoch   1 Batch  843/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.904, Loss:  0.434
    Epoch   1 Batch  844/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.906, Loss:  0.448
    Epoch   1 Batch  845/1378 - Train Accuracy:  0.949, Validation Accuracy:  0.914, Loss:  0.436
    Epoch   1 Batch  846/1378 - Train Accuracy:  0.958, Validation Accuracy:  0.906, Loss:  0.425
    Epoch   1 Batch  847/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.903, Loss:  0.478
    Epoch   1 Batch  848/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.905, Loss:  0.407
    Epoch   1 Batch  849/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.900, Loss:  0.470
    Epoch   1 Batch  850/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.902, Loss:  0.404
    Epoch   1 Batch  851/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.900, Loss:  0.411
    Epoch   1 Batch  852/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.888, Loss:  0.400
    Epoch   1 Batch  853/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.888, Loss:  0.493
    Epoch   1 Batch  854/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.907, Loss:  0.439
    Epoch   1 Batch  855/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.921, Loss:  0.426
    Epoch   1 Batch  856/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.940, Loss:  0.421
    Epoch   1 Batch  857/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.935, Loss:  0.452
    Epoch   1 Batch  858/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.938, Loss:  0.453
    Epoch   1 Batch  859/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.945, Loss:  0.403
    Epoch   1 Batch  860/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.945, Loss:  0.440
    Epoch   1 Batch  861/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.948, Loss:  0.434
    Epoch   1 Batch  862/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.943, Loss:  0.443
    Epoch   1 Batch  863/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.926, Loss:  0.440
    Epoch   1 Batch  864/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.919, Loss:  0.425
    Epoch   1 Batch  865/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.924, Loss:  0.428
    Epoch   1 Batch  866/1378 - Train Accuracy:  0.898, Validation Accuracy:  0.933, Loss:  0.437
    Epoch   1 Batch  867/1378 - Train Accuracy:  0.887, Validation Accuracy:  0.943, Loss:  0.465
    Epoch   1 Batch  868/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.934, Loss:  0.408
    Epoch   1 Batch  869/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.933, Loss:  0.452
    Epoch   1 Batch  870/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.930, Loss:  0.474
    Epoch   1 Batch  871/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.911, Loss:  0.363
    Epoch   1 Batch  872/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.893, Loss:  0.433
    Epoch   1 Batch  873/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.909, Loss:  0.442
    Epoch   1 Batch  874/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.930, Loss:  0.415
    Epoch   1 Batch  875/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.931, Loss:  0.412
    Epoch   1 Batch  876/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.928, Loss:  0.411
    Epoch   1 Batch  877/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.926, Loss:  0.450
    Epoch   1 Batch  878/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.926, Loss:  0.427
    Epoch   1 Batch  879/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.926, Loss:  0.424
    Epoch   1 Batch  880/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.927, Loss:  0.410
    Epoch   1 Batch  881/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.926, Loss:  0.451
    Epoch   1 Batch  882/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.915, Loss:  0.399
    Epoch   1 Batch  883/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.916, Loss:  0.445
    Epoch   1 Batch  884/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.915, Loss:  0.449
    Epoch   1 Batch  885/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.926, Loss:  0.492
    Epoch   1 Batch  886/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.938, Loss:  0.436
    Epoch   1 Batch  887/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.943, Loss:  0.422
    Epoch   1 Batch  888/1378 - Train Accuracy:  0.882, Validation Accuracy:  0.936, Loss:  0.473
    Epoch   1 Batch  889/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.934, Loss:  0.418
    Epoch   1 Batch  890/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.937, Loss:  0.440
    Epoch   1 Batch  891/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.934, Loss:  0.441
    Epoch   1 Batch  892/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.935, Loss:  0.454
    Epoch   1 Batch  893/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.934, Loss:  0.464
    Epoch   1 Batch  894/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.937, Loss:  0.425
    Epoch   1 Batch  895/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.941, Loss:  0.425
    Epoch   1 Batch  896/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.933, Loss:  0.428
    Epoch   1 Batch  897/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.926, Loss:  0.409
    Epoch   1 Batch  898/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.938, Loss:  0.357
    Epoch   1 Batch  899/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.940, Loss:  0.446
    Epoch   1 Batch  900/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.936, Loss:  0.459
    Epoch   1 Batch  901/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.946, Loss:  0.472
    Epoch   1 Batch  902/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.955, Loss:  0.413
    Epoch   1 Batch  903/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.949, Loss:  0.461
    Epoch   1 Batch  904/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.936, Loss:  0.461
    Epoch   1 Batch  905/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.934, Loss:  0.456
    Epoch   1 Batch  906/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.933, Loss:  0.434
    Epoch   1 Batch  907/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.933, Loss:  0.454
    Epoch   1 Batch  908/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.934, Loss:  0.491
    Epoch   1 Batch  909/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.924, Loss:  0.447
    Epoch   1 Batch  910/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.917, Loss:  0.453
    Epoch   1 Batch  911/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.917, Loss:  0.457
    Epoch   1 Batch  912/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.917, Loss:  0.443
    Epoch   1 Batch  913/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.915, Loss:  0.392
    Epoch   1 Batch  914/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.918, Loss:  0.474
    Epoch   1 Batch  915/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.935, Loss:  0.465
    Epoch   1 Batch  916/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.942, Loss:  0.431
    Epoch   1 Batch  917/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.931, Loss:  0.368
    Epoch   1 Batch  918/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.936, Loss:  0.465
    Epoch   1 Batch  919/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.930, Loss:  0.422
    Epoch   1 Batch  920/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.935, Loss:  0.479
    Epoch   1 Batch  921/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.933, Loss:  0.459
    Epoch   1 Batch  922/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.926, Loss:  0.435
    Epoch   1 Batch  923/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.920, Loss:  0.415
    Epoch   1 Batch  924/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.919, Loss:  0.440
    Epoch   1 Batch  925/1378 - Train Accuracy:  0.876, Validation Accuracy:  0.917, Loss:  0.439
    Epoch   1 Batch  926/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.912, Loss:  0.463
    Epoch   1 Batch  927/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.916, Loss:  0.472
    Epoch   1 Batch  928/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.922, Loss:  0.401
    Epoch   1 Batch  929/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.926, Loss:  0.452
    Epoch   1 Batch  930/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.929, Loss:  0.409
    Epoch   1 Batch  931/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.935, Loss:  0.470
    Epoch   1 Batch  932/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.940, Loss:  0.398
    Epoch   1 Batch  933/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.931, Loss:  0.440
    Epoch   1 Batch  934/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.927, Loss:  0.458
    Epoch   1 Batch  935/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.928, Loss:  0.431
    Epoch   1 Batch  936/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.920, Loss:  0.400
    Epoch   1 Batch  937/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.922, Loss:  0.442
    Epoch   1 Batch  938/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.913, Loss:  0.463
    Epoch   1 Batch  939/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.915, Loss:  0.418
    Epoch   1 Batch  940/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.920, Loss:  0.454
    Epoch   1 Batch  941/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.930, Loss:  0.452
    Epoch   1 Batch  942/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.945, Loss:  0.459
    Epoch   1 Batch  943/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.951, Loss:  0.475
    Epoch   1 Batch  944/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.953, Loss:  0.425
    Epoch   1 Batch  945/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.938, Loss:  0.455
    Epoch   1 Batch  946/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.442
    Epoch   1 Batch  947/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.935, Loss:  0.465
    Epoch   1 Batch  948/1378 - Train Accuracy:  0.878, Validation Accuracy:  0.936, Loss:  0.413
    Epoch   1 Batch  949/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.938, Loss:  0.465
    Epoch   1 Batch  950/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.931, Loss:  0.423
    Epoch   1 Batch  951/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.931, Loss:  0.424
    Epoch   1 Batch  952/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.931, Loss:  0.411
    Epoch   1 Batch  953/1378 - Train Accuracy:  0.960, Validation Accuracy:  0.930, Loss:  0.427
    Epoch   1 Batch  954/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.929, Loss:  0.416
    Epoch   1 Batch  955/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.908, Loss:  0.429
    Epoch   1 Batch  956/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.916, Loss:  0.417
    Epoch   1 Batch  957/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.913, Loss:  0.438
    Epoch   1 Batch  958/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.913, Loss:  0.407
    Epoch   1 Batch  959/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.911, Loss:  0.426
    Epoch   1 Batch  960/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.911, Loss:  0.445
    Epoch   1 Batch  961/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.898, Loss:  0.426
    Epoch   1 Batch  962/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.907, Loss:  0.442
    Epoch   1 Batch  963/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.906, Loss:  0.406
    Epoch   1 Batch  964/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.910, Loss:  0.474
    Epoch   1 Batch  965/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.916, Loss:  0.429
    Epoch   1 Batch  966/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.902, Loss:  0.512
    Epoch   1 Batch  967/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.900, Loss:  0.437
    Epoch   1 Batch  968/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.899, Loss:  0.410
    Epoch   1 Batch  969/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.901, Loss:  0.424
    Epoch   1 Batch  970/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.902, Loss:  0.386
    Epoch   1 Batch  971/1378 - Train Accuracy:  0.962, Validation Accuracy:  0.896, Loss:  0.385
    Epoch   1 Batch  972/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.905, Loss:  0.454
    Epoch   1 Batch  973/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.906, Loss:  0.470
    Epoch   1 Batch  974/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.920, Loss:  0.407
    Epoch   1 Batch  975/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.924, Loss:  0.418
    Epoch   1 Batch  976/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.936, Loss:  0.454
    Epoch   1 Batch  977/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.931, Loss:  0.454
    Epoch   1 Batch  978/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.923, Loss:  0.391
    Epoch   1 Batch  979/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.934, Loss:  0.489
    Epoch   1 Batch  980/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.910, Loss:  0.485
    Epoch   1 Batch  981/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.929, Loss:  0.430
    Epoch   1 Batch  982/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.921, Loss:  0.413
    Epoch   1 Batch  983/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.935, Loss:  0.431
    Epoch   1 Batch  984/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.942, Loss:  0.388
    Epoch   1 Batch  985/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.942, Loss:  0.423
    Epoch   1 Batch  986/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.942, Loss:  0.444
    Epoch   1 Batch  987/1378 - Train Accuracy:  0.949, Validation Accuracy:  0.937, Loss:  0.474
    Epoch   1 Batch  988/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.945, Loss:  0.399
    Epoch   1 Batch  989/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.939, Loss:  0.459
    Epoch   1 Batch  990/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.937, Loss:  0.473
    Epoch   1 Batch  991/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.938, Loss:  0.418
    Epoch   1 Batch  992/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.941, Loss:  0.443
    Epoch   1 Batch  993/1378 - Train Accuracy:  0.949, Validation Accuracy:  0.953, Loss:  0.474
    Epoch   1 Batch  994/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.951, Loss:  0.456
    Epoch   1 Batch  995/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.947, Loss:  0.426
    Epoch   1 Batch  996/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.952, Loss:  0.442
    Epoch   1 Batch  997/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.939, Loss:  0.403
    Epoch   1 Batch  998/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.948, Loss:  0.441
    Epoch   1 Batch  999/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.942, Loss:  0.430
    Epoch   1 Batch 1000/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.929, Loss:  0.462
    Epoch   1 Batch 1001/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.932, Loss:  0.423
    Epoch   1 Batch 1002/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.912, Loss:  0.446
    Epoch   1 Batch 1003/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.915, Loss:  0.486
    Epoch   1 Batch 1004/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.908, Loss:  0.512
    Epoch   1 Batch 1005/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.896, Loss:  0.425
    Epoch   1 Batch 1006/1378 - Train Accuracy:  0.865, Validation Accuracy:  0.900, Loss:  0.442
    Epoch   1 Batch 1007/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.904, Loss:  0.423
    Epoch   1 Batch 1008/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.905, Loss:  0.425
    Epoch   1 Batch 1009/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.900, Loss:  0.462
    Epoch   1 Batch 1010/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.912, Loss:  0.429
    Epoch   1 Batch 1011/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.911, Loss:  0.407
    Epoch   1 Batch 1012/1378 - Train Accuracy:  0.874, Validation Accuracy:  0.907, Loss:  0.427
    Epoch   1 Batch 1013/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.914, Loss:  0.418
    Epoch   1 Batch 1014/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.920, Loss:  0.472
    Epoch   1 Batch 1015/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.929, Loss:  0.453
    Epoch   1 Batch 1016/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.938, Loss:  0.429
    Epoch   1 Batch 1017/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.926, Loss:  0.424
    Epoch   1 Batch 1018/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.914, Loss:  0.482
    Epoch   1 Batch 1019/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.900, Loss:  0.466
    Epoch   1 Batch 1020/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.895, Loss:  0.408
    Epoch   1 Batch 1021/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.896, Loss:  0.432
    Epoch   1 Batch 1022/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.896, Loss:  0.373
    Epoch   1 Batch 1023/1378 - Train Accuracy:  0.885, Validation Accuracy:  0.896, Loss:  0.406
    Epoch   1 Batch 1024/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.901, Loss:  0.515
    Epoch   1 Batch 1025/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.918, Loss:  0.418
    Epoch   1 Batch 1026/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.924, Loss:  0.459
    Epoch   1 Batch 1027/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.930, Loss:  0.425
    Epoch   1 Batch 1028/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.923, Loss:  0.482
    Epoch   1 Batch 1029/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.921, Loss:  0.357
    Epoch   1 Batch 1030/1378 - Train Accuracy:  0.973, Validation Accuracy:  0.915, Loss:  0.403
    Epoch   1 Batch 1031/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.917, Loss:  0.443
    Epoch   1 Batch 1032/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.922, Loss:  0.434
    Epoch   1 Batch 1033/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.928, Loss:  0.434
    Epoch   1 Batch 1034/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.915, Loss:  0.429
    Epoch   1 Batch 1035/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.916, Loss:  0.423
    Epoch   1 Batch 1036/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.915, Loss:  0.450
    Epoch   1 Batch 1037/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.917, Loss:  0.414
    Epoch   1 Batch 1038/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.918, Loss:  0.415
    Epoch   1 Batch 1039/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.922, Loss:  0.464
    Epoch   1 Batch 1040/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.917, Loss:  0.423
    Epoch   1 Batch 1041/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.920, Loss:  0.454
    Epoch   1 Batch 1042/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.915, Loss:  0.428
    Epoch   1 Batch 1043/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.921, Loss:  0.475
    Epoch   1 Batch 1044/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.922, Loss:  0.419
    Epoch   1 Batch 1045/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.912, Loss:  0.479
    Epoch   1 Batch 1046/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.913, Loss:  0.432
    Epoch   1 Batch 1047/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.914, Loss:  0.431
    Epoch   1 Batch 1048/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.930, Loss:  0.425
    Epoch   1 Batch 1049/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.938, Loss:  0.418
    Epoch   1 Batch 1050/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.941, Loss:  0.455
    Epoch   1 Batch 1051/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.922, Loss:  0.396
    Epoch   1 Batch 1052/1378 - Train Accuracy:  0.959, Validation Accuracy:  0.925, Loss:  0.433
    Epoch   1 Batch 1053/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.943, Loss:  0.426
    Epoch   1 Batch 1054/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.942, Loss:  0.413
    Epoch   1 Batch 1055/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.942, Loss:  0.473
    Epoch   1 Batch 1056/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.939, Loss:  0.444
    Epoch   1 Batch 1057/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.930, Loss:  0.424
    Epoch   1 Batch 1058/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.920, Loss:  0.446
    Epoch   1 Batch 1059/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.931, Loss:  0.418
    Epoch   1 Batch 1060/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.933, Loss:  0.492
    Epoch   1 Batch 1061/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.951, Loss:  0.470
    Epoch   1 Batch 1062/1378 - Train Accuracy:  0.905, Validation Accuracy:  0.943, Loss:  0.438
    Epoch   1 Batch 1063/1378 - Train Accuracy:  0.868, Validation Accuracy:  0.936, Loss:  0.475
    Epoch   1 Batch 1064/1378 - Train Accuracy:  0.864, Validation Accuracy:  0.938, Loss:  0.431
    Epoch   1 Batch 1065/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.909, Loss:  0.460
    Epoch   1 Batch 1066/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.913, Loss:  0.464
    Epoch   1 Batch 1067/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.913, Loss:  0.426
    Epoch   1 Batch 1068/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.914, Loss:  0.407
    Epoch   1 Batch 1069/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.913, Loss:  0.413
    Epoch   1 Batch 1070/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.921, Loss:  0.401
    Epoch   1 Batch 1071/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.927, Loss:  0.444
    Epoch   1 Batch 1072/1378 - Train Accuracy:  0.907, Validation Accuracy:  0.933, Loss:  0.451
    Epoch   1 Batch 1073/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.940, Loss:  0.449
    Epoch   1 Batch 1074/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.933, Loss:  0.405
    Epoch   1 Batch 1075/1378 - Train Accuracy:  0.963, Validation Accuracy:  0.924, Loss:  0.399
    Epoch   1 Batch 1076/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.915, Loss:  0.409
    Epoch   1 Batch 1077/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.915, Loss:  0.457
    Epoch   1 Batch 1078/1378 - Train Accuracy:  0.967, Validation Accuracy:  0.911, Loss:  0.429
    Epoch   1 Batch 1079/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.902, Loss:  0.423
    Epoch   1 Batch 1080/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.895, Loss:  0.414
    Epoch   1 Batch 1081/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.912, Loss:  0.423
    Epoch   1 Batch 1082/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.913, Loss:  0.429
    Epoch   1 Batch 1083/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.925, Loss:  0.453
    Epoch   1 Batch 1084/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.919, Loss:  0.434
    Epoch   1 Batch 1085/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.916, Loss:  0.466
    Epoch   1 Batch 1086/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.917, Loss:  0.403
    Epoch   1 Batch 1087/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.921, Loss:  0.417
    Epoch   1 Batch 1088/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.922, Loss:  0.475
    Epoch   1 Batch 1089/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.911, Loss:  0.454
    Epoch   1 Batch 1090/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.910, Loss:  0.446
    Epoch   1 Batch 1091/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.923, Loss:  0.411
    Epoch   1 Batch 1092/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.921, Loss:  0.433
    Epoch   1 Batch 1093/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.922, Loss:  0.442
    Epoch   1 Batch 1094/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.929, Loss:  0.475
    Epoch   1 Batch 1095/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.923, Loss:  0.420
    Epoch   1 Batch 1096/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.902, Loss:  0.403
    Epoch   1 Batch 1097/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.896, Loss:  0.470
    Epoch   1 Batch 1098/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.902, Loss:  0.456
    Epoch   1 Batch 1099/1378 - Train Accuracy:  0.951, Validation Accuracy:  0.893, Loss:  0.422
    Epoch   1 Batch 1100/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.900, Loss:  0.438
    Epoch   1 Batch 1101/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.902, Loss:  0.473
    Epoch   1 Batch 1102/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.902, Loss:  0.434
    Epoch   1 Batch 1103/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.914, Loss:  0.476
    Epoch   1 Batch 1104/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.920, Loss:  0.433
    Epoch   1 Batch 1105/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.920, Loss:  0.437
    Epoch   1 Batch 1106/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.913, Loss:  0.393
    Epoch   1 Batch 1107/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.912, Loss:  0.483
    Epoch   1 Batch 1108/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.925, Loss:  0.445
    Epoch   1 Batch 1109/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.900, Loss:  0.417
    Epoch   1 Batch 1110/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.872, Loss:  0.510
    Epoch   1 Batch 1111/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.851, Loss:  0.458
    Epoch   1 Batch 1112/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.855, Loss:  0.456
    Epoch   1 Batch 1113/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.866, Loss:  0.415
    Epoch   1 Batch 1114/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.880, Loss:  0.411
    Epoch   1 Batch 1115/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.909, Loss:  0.389
    Epoch   1 Batch 1116/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.916, Loss:  0.423
    Epoch   1 Batch 1117/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.905, Loss:  0.391
    Epoch   1 Batch 1118/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.911, Loss:  0.466
    Epoch   1 Batch 1119/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.914, Loss:  0.481
    Epoch   1 Batch 1120/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.906, Loss:  0.428
    Epoch   1 Batch 1121/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.900, Loss:  0.433
    Epoch   1 Batch 1122/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.901, Loss:  0.403
    Epoch   1 Batch 1123/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.893, Loss:  0.449
    Epoch   1 Batch 1124/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.889, Loss:  0.468
    Epoch   1 Batch 1125/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.885, Loss:  0.417
    Epoch   1 Batch 1126/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.886, Loss:  0.437
    Epoch   1 Batch 1127/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.886, Loss:  0.432
    Epoch   1 Batch 1128/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.904, Loss:  0.454
    Epoch   1 Batch 1129/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.903, Loss:  0.423
    Epoch   1 Batch 1130/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.914, Loss:  0.429
    Epoch   1 Batch 1131/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.915, Loss:  0.473
    Epoch   1 Batch 1132/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.916, Loss:  0.409
    Epoch   1 Batch 1133/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.917, Loss:  0.488
    Epoch   1 Batch 1134/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.913, Loss:  0.423
    Epoch   1 Batch 1135/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.907, Loss:  0.440
    Epoch   1 Batch 1136/1378 - Train Accuracy:  0.896, Validation Accuracy:  0.901, Loss:  0.451
    Epoch   1 Batch 1137/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.914, Loss:  0.445
    Epoch   1 Batch 1138/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.887, Loss:  0.440
    Epoch   1 Batch 1139/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.886, Loss:  0.431
    Epoch   1 Batch 1140/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.881, Loss:  0.439
    Epoch   1 Batch 1141/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.887, Loss:  0.425
    Epoch   1 Batch 1142/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.902, Loss:  0.433
    Epoch   1 Batch 1143/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.907, Loss:  0.430
    Epoch   1 Batch 1144/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.912, Loss:  0.447
    Epoch   1 Batch 1145/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.917, Loss:  0.404
    Epoch   1 Batch 1146/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.915, Loss:  0.471
    Epoch   1 Batch 1147/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.916, Loss:  0.441
    Epoch   1 Batch 1148/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.925, Loss:  0.428
    Epoch   1 Batch 1149/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.931, Loss:  0.407
    Epoch   1 Batch 1150/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.930, Loss:  0.398
    Epoch   1 Batch 1151/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.934, Loss:  0.460
    Epoch   1 Batch 1152/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.933, Loss:  0.441
    Epoch   1 Batch 1153/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.936, Loss:  0.397
    Epoch   1 Batch 1154/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.942, Loss:  0.408
    Epoch   1 Batch 1155/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.944, Loss:  0.460
    Epoch   1 Batch 1156/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.933, Loss:  0.440
    Epoch   1 Batch 1157/1378 - Train Accuracy:  0.947, Validation Accuracy:  0.933, Loss:  0.434
    Epoch   1 Batch 1158/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.932, Loss:  0.433
    Epoch   1 Batch 1159/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.924, Loss:  0.402
    Epoch   1 Batch 1160/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.905, Loss:  0.419
    Epoch   1 Batch 1161/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.905, Loss:  0.415
    Epoch   1 Batch 1162/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.909, Loss:  0.435
    Epoch   1 Batch 1163/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.909, Loss:  0.417
    Epoch   1 Batch 1164/1378 - Train Accuracy:  0.893, Validation Accuracy:  0.913, Loss:  0.420
    Epoch   1 Batch 1165/1378 - Train Accuracy:  0.958, Validation Accuracy:  0.924, Loss:  0.482
    Epoch   1 Batch 1166/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.918, Loss:  0.414
    Epoch   1 Batch 1167/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.925, Loss:  0.422
    Epoch   1 Batch 1168/1378 - Train Accuracy:  0.949, Validation Accuracy:  0.929, Loss:  0.465
    Epoch   1 Batch 1169/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.929, Loss:  0.521
    Epoch   1 Batch 1170/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.923, Loss:  0.444
    Epoch   1 Batch 1171/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.925, Loss:  0.422
    Epoch   1 Batch 1172/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.929, Loss:  0.455
    Epoch   1 Batch 1173/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.935, Loss:  0.430
    Epoch   1 Batch 1174/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.936, Loss:  0.449
    Epoch   1 Batch 1175/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.925, Loss:  0.431
    Epoch   1 Batch 1176/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.911, Loss:  0.490
    Epoch   1 Batch 1177/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.904, Loss:  0.437
    Epoch   1 Batch 1178/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.916, Loss:  0.421
    Epoch   1 Batch 1179/1378 - Train Accuracy:  0.897, Validation Accuracy:  0.927, Loss:  0.477
    Epoch   1 Batch 1180/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.931, Loss:  0.458
    Epoch   1 Batch 1181/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.935, Loss:  0.437
    Epoch   1 Batch 1182/1378 - Train Accuracy:  0.979, Validation Accuracy:  0.928, Loss:  0.451
    Epoch   1 Batch 1183/1378 - Train Accuracy:  0.918, Validation Accuracy:  0.910, Loss:  0.419
    Epoch   1 Batch 1184/1378 - Train Accuracy:  0.960, Validation Accuracy:  0.907, Loss:  0.410
    Epoch   1 Batch 1185/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.909, Loss:  0.400
    Epoch   1 Batch 1186/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.901, Loss:  0.413
    Epoch   1 Batch 1187/1378 - Train Accuracy:  0.899, Validation Accuracy:  0.905, Loss:  0.435
    Epoch   1 Batch 1188/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.920, Loss:  0.464
    Epoch   1 Batch 1189/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.930, Loss:  0.408
    Epoch   1 Batch 1190/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.917, Loss:  0.416
    Epoch   1 Batch 1191/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.915, Loss:  0.428
    Epoch   1 Batch 1192/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.901, Loss:  0.440
    Epoch   1 Batch 1193/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.916, Loss:  0.415
    Epoch   1 Batch 1194/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.923, Loss:  0.412
    Epoch   1 Batch 1195/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.917, Loss:  0.422
    Epoch   1 Batch 1196/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.918, Loss:  0.441
    Epoch   1 Batch 1197/1378 - Train Accuracy:  0.953, Validation Accuracy:  0.919, Loss:  0.412
    Epoch   1 Batch 1198/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.929, Loss:  0.446
    Epoch   1 Batch 1199/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.936, Loss:  0.457
    Epoch   1 Batch 1200/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.940, Loss:  0.422
    Epoch   1 Batch 1201/1378 - Train Accuracy:  0.953, Validation Accuracy:  0.940, Loss:  0.425
    Epoch   1 Batch 1202/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.934, Loss:  0.475
    Epoch   1 Batch 1203/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.935, Loss:  0.438
    Epoch   1 Batch 1204/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.942, Loss:  0.391
    Epoch   1 Batch 1205/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.927, Loss:  0.451
    Epoch   1 Batch 1206/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.932, Loss:  0.438
    Epoch   1 Batch 1207/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.930, Loss:  0.422
    Epoch   1 Batch 1208/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.922, Loss:  0.426
    Epoch   1 Batch 1209/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.927, Loss:  0.464
    Epoch   1 Batch 1210/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.933, Loss:  0.433
    Epoch   1 Batch 1211/1378 - Train Accuracy:  0.968, Validation Accuracy:  0.940, Loss:  0.480
    Epoch   1 Batch 1212/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.940, Loss:  0.401
    Epoch   1 Batch 1213/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.920, Loss:  0.452
    Epoch   1 Batch 1214/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.924, Loss:  0.400
    Epoch   1 Batch 1215/1378 - Train Accuracy:  0.961, Validation Accuracy:  0.918, Loss:  0.434
    Epoch   1 Batch 1216/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.923, Loss:  0.414
    Epoch   1 Batch 1217/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.930, Loss:  0.399
    Epoch   1 Batch 1218/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.925, Loss:  0.419
    Epoch   1 Batch 1219/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.926, Loss:  0.409
    Epoch   1 Batch 1220/1378 - Train Accuracy:  0.919, Validation Accuracy:  0.920, Loss:  0.391
    Epoch   1 Batch 1221/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.917, Loss:  0.442
    Epoch   1 Batch 1222/1378 - Train Accuracy:  0.954, Validation Accuracy:  0.929, Loss:  0.395
    Epoch   1 Batch 1223/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.929, Loss:  0.453
    Epoch   1 Batch 1224/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.935, Loss:  0.465
    Epoch   1 Batch 1225/1378 - Train Accuracy:  0.952, Validation Accuracy:  0.935, Loss:  0.392
    Epoch   1 Batch 1226/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.937, Loss:  0.375
    Epoch   1 Batch 1227/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.938, Loss:  0.435
    Epoch   1 Batch 1228/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.935, Loss:  0.396
    Epoch   1 Batch 1229/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.931, Loss:  0.424
    Epoch   1 Batch 1230/1378 - Train Accuracy:  0.963, Validation Accuracy:  0.935, Loss:  0.404
    Epoch   1 Batch 1231/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.929, Loss:  0.451
    Epoch   1 Batch 1232/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.935, Loss:  0.419
    Epoch   1 Batch 1233/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.943, Loss:  0.476
    Epoch   1 Batch 1234/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.937, Loss:  0.407
    Epoch   1 Batch 1235/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.951, Loss:  0.425
    Epoch   1 Batch 1236/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.952, Loss:  0.457
    Epoch   1 Batch 1237/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.952, Loss:  0.398
    Epoch   1 Batch 1238/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.951, Loss:  0.431
    Epoch   1 Batch 1239/1378 - Train Accuracy:  0.903, Validation Accuracy:  0.951, Loss:  0.461
    Epoch   1 Batch 1240/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.939, Loss:  0.413
    Epoch   1 Batch 1241/1378 - Train Accuracy:  0.890, Validation Accuracy:  0.950, Loss:  0.438
    Epoch   1 Batch 1242/1378 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.419
    Epoch   1 Batch 1243/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.946, Loss:  0.469
    Epoch   1 Batch 1244/1378 - Train Accuracy:  0.964, Validation Accuracy:  0.947, Loss:  0.419
    Epoch   1 Batch 1245/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.946, Loss:  0.463
    Epoch   1 Batch 1246/1378 - Train Accuracy:  0.967, Validation Accuracy:  0.946, Loss:  0.436
    Epoch   1 Batch 1247/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.952, Loss:  0.418
    Epoch   1 Batch 1248/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.951, Loss:  0.425
    Epoch   1 Batch 1249/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.952, Loss:  0.389
    Epoch   1 Batch 1250/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.943, Loss:  0.428
    Epoch   1 Batch 1251/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.935, Loss:  0.452
    Epoch   1 Batch 1252/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.942, Loss:  0.462
    Epoch   1 Batch 1253/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.929, Loss:  0.492
    Epoch   1 Batch 1254/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.934, Loss:  0.426
    Epoch   1 Batch 1255/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.934, Loss:  0.429
    Epoch   1 Batch 1256/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.934, Loss:  0.402
    Epoch   1 Batch 1257/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.935, Loss:  0.441
    Epoch   1 Batch 1258/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.930, Loss:  0.416
    Epoch   1 Batch 1259/1378 - Train Accuracy:  0.915, Validation Accuracy:  0.954, Loss:  0.390
    Epoch   1 Batch 1260/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.954, Loss:  0.415
    Epoch   1 Batch 1261/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.952, Loss:  0.399
    Epoch   1 Batch 1262/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.953, Loss:  0.414
    Epoch   1 Batch 1263/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.962, Loss:  0.453
    Epoch   1 Batch 1264/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.951, Loss:  0.410
    Epoch   1 Batch 1265/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.947, Loss:  0.417
    Epoch   1 Batch 1266/1378 - Train Accuracy:  0.926, Validation Accuracy:  0.941, Loss:  0.487
    Epoch   1 Batch 1267/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.945, Loss:  0.458
    Epoch   1 Batch 1268/1378 - Train Accuracy:  0.963, Validation Accuracy:  0.945, Loss:  0.452
    Epoch   1 Batch 1269/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.945, Loss:  0.412
    Epoch   1 Batch 1270/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.924, Loss:  0.433
    Epoch   1 Batch 1271/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.905, Loss:  0.434
    Epoch   1 Batch 1272/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.913, Loss:  0.415
    Epoch   1 Batch 1273/1378 - Train Accuracy:  0.973, Validation Accuracy:  0.924, Loss:  0.431
    Epoch   1 Batch 1274/1378 - Train Accuracy:  0.925, Validation Accuracy:  0.924, Loss:  0.441
    Epoch   1 Batch 1275/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.933, Loss:  0.429
    Epoch   1 Batch 1276/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.933, Loss:  0.457
    Epoch   1 Batch 1277/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.921, Loss:  0.416
    Epoch   1 Batch 1278/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.906, Loss:  0.456
    Epoch   1 Batch 1279/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.900, Loss:  0.430
    Epoch   1 Batch 1280/1378 - Train Accuracy:  0.949, Validation Accuracy:  0.910, Loss:  0.416
    Epoch   1 Batch 1281/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.910, Loss:  0.465
    Epoch   1 Batch 1282/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.910, Loss:  0.408
    Epoch   1 Batch 1283/1378 - Train Accuracy:  0.965, Validation Accuracy:  0.912, Loss:  0.380
    Epoch   1 Batch 1284/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.917, Loss:  0.427
    Epoch   1 Batch 1285/1378 - Train Accuracy:  0.979, Validation Accuracy:  0.916, Loss:  0.435
    Epoch   1 Batch 1286/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.917, Loss:  0.442
    Epoch   1 Batch 1287/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.915, Loss:  0.444
    Epoch   1 Batch 1288/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.917, Loss:  0.411
    Epoch   1 Batch 1289/1378 - Train Accuracy:  0.956, Validation Accuracy:  0.927, Loss:  0.442
    Epoch   1 Batch 1290/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.932, Loss:  0.427
    Epoch   1 Batch 1291/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.932, Loss:  0.483
    Epoch   1 Batch 1292/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.938, Loss:  0.355
    Epoch   1 Batch 1293/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.941, Loss:  0.420
    Epoch   1 Batch 1294/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.935, Loss:  0.446
    Epoch   1 Batch 1295/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.919, Loss:  0.414
    Epoch   1 Batch 1296/1378 - Train Accuracy:  0.940, Validation Accuracy:  0.918, Loss:  0.400
    Epoch   1 Batch 1297/1378 - Train Accuracy:  0.942, Validation Accuracy:  0.918, Loss:  0.401
    Epoch   1 Batch 1298/1378 - Train Accuracy:  0.944, Validation Accuracy:  0.911, Loss:  0.431
    Epoch   1 Batch 1299/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.919, Loss:  0.403
    Epoch   1 Batch 1300/1378 - Train Accuracy:  0.941, Validation Accuracy:  0.926, Loss:  0.414
    Epoch   1 Batch 1301/1378 - Train Accuracy:  0.906, Validation Accuracy:  0.921, Loss:  0.397
    Epoch   1 Batch 1302/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.929, Loss:  0.423
    Epoch   1 Batch 1303/1378 - Train Accuracy:  0.927, Validation Accuracy:  0.930, Loss:  0.422
    Epoch   1 Batch 1304/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.930, Loss:  0.419
    Epoch   1 Batch 1305/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.930, Loss:  0.429
    Epoch   1 Batch 1306/1378 - Train Accuracy:  0.932, Validation Accuracy:  0.926, Loss:  0.407
    Epoch   1 Batch 1307/1378 - Train Accuracy:  0.921, Validation Accuracy:  0.935, Loss:  0.451
    Epoch   1 Batch 1308/1378 - Train Accuracy:  0.955, Validation Accuracy:  0.930, Loss:  0.416
    Epoch   1 Batch 1309/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.942, Loss:  0.510
    Epoch   1 Batch 1310/1378 - Train Accuracy:  0.904, Validation Accuracy:  0.947, Loss:  0.456
    Epoch   1 Batch 1311/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.948, Loss:  0.421
    Epoch   1 Batch 1312/1378 - Train Accuracy:  0.924, Validation Accuracy:  0.947, Loss:  0.472
    Epoch   1 Batch 1313/1378 - Train Accuracy:  0.950, Validation Accuracy:  0.946, Loss:  0.408
    Epoch   1 Batch 1314/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.480
    Epoch   1 Batch 1315/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.937, Loss:  0.483
    Epoch   1 Batch 1316/1378 - Train Accuracy:  0.891, Validation Accuracy:  0.934, Loss:  0.426
    Epoch   1 Batch 1317/1378 - Train Accuracy:  0.929, Validation Accuracy:  0.911, Loss:  0.418
    Epoch   1 Batch 1318/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.904, Loss:  0.413
    Epoch   1 Batch 1319/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.904, Loss:  0.469
    Epoch   1 Batch 1320/1378 - Train Accuracy:  0.937, Validation Accuracy:  0.908, Loss:  0.428
    Epoch   1 Batch 1321/1378 - Train Accuracy:  0.914, Validation Accuracy:  0.911, Loss:  0.399
    Epoch   1 Batch 1322/1378 - Train Accuracy:  0.930, Validation Accuracy:  0.911, Loss:  0.380
    Epoch   1 Batch 1323/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.910, Loss:  0.404
    Epoch   1 Batch 1324/1378 - Train Accuracy:  0.892, Validation Accuracy:  0.918, Loss:  0.469
    Epoch   1 Batch 1325/1378 - Train Accuracy:  0.958, Validation Accuracy:  0.912, Loss:  0.408
    Epoch   1 Batch 1326/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.911, Loss:  0.426
    Epoch   1 Batch 1327/1378 - Train Accuracy:  0.917, Validation Accuracy:  0.920, Loss:  0.425
    Epoch   1 Batch 1328/1378 - Train Accuracy:  0.894, Validation Accuracy:  0.928, Loss:  0.484
    Epoch   1 Batch 1329/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.935, Loss:  0.415
    Epoch   1 Batch 1330/1378 - Train Accuracy:  0.901, Validation Accuracy:  0.934, Loss:  0.474
    Epoch   1 Batch 1331/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.928, Loss:  0.406
    Epoch   1 Batch 1332/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.914, Loss:  0.406
    Epoch   1 Batch 1333/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.910, Loss:  0.419
    Epoch   1 Batch 1334/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.904, Loss:  0.416
    Epoch   1 Batch 1335/1378 - Train Accuracy:  0.939, Validation Accuracy:  0.910, Loss:  0.395
    Epoch   1 Batch 1336/1378 - Train Accuracy:  0.959, Validation Accuracy:  0.916, Loss:  0.427
    Epoch   1 Batch 1337/1378 - Train Accuracy:  0.911, Validation Accuracy:  0.916, Loss:  0.461
    Epoch   1 Batch 1338/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.916, Loss:  0.431
    Epoch   1 Batch 1339/1378 - Train Accuracy:  0.958, Validation Accuracy:  0.908, Loss:  0.386
    Epoch   1 Batch 1340/1378 - Train Accuracy:  0.900, Validation Accuracy:  0.908, Loss:  0.427
    Epoch   1 Batch 1341/1378 - Train Accuracy:  0.961, Validation Accuracy:  0.915, Loss:  0.420
    Epoch   1 Batch 1342/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.909, Loss:  0.438
    Epoch   1 Batch 1343/1378 - Train Accuracy:  0.912, Validation Accuracy:  0.909, Loss:  0.426
    Epoch   1 Batch 1344/1378 - Train Accuracy:  0.953, Validation Accuracy:  0.920, Loss:  0.406
    Epoch   1 Batch 1345/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.923, Loss:  0.442
    Epoch   1 Batch 1346/1378 - Train Accuracy:  0.895, Validation Accuracy:  0.918, Loss:  0.426
    Epoch   1 Batch 1347/1378 - Train Accuracy:  0.953, Validation Accuracy:  0.925, Loss:  0.359
    Epoch   1 Batch 1348/1378 - Train Accuracy:  0.908, Validation Accuracy:  0.927, Loss:  0.439
    Epoch   1 Batch 1349/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.919, Loss:  0.410
    Epoch   1 Batch 1350/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.920, Loss:  0.439
    Epoch   1 Batch 1351/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.921, Loss:  0.431
    Epoch   1 Batch 1352/1378 - Train Accuracy:  0.913, Validation Accuracy:  0.919, Loss:  0.384
    Epoch   1 Batch 1353/1378 - Train Accuracy:  0.936, Validation Accuracy:  0.937, Loss:  0.430
    Epoch   1 Batch 1354/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.935, Loss:  0.447
    Epoch   1 Batch 1355/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.935, Loss:  0.443
    Epoch   1 Batch 1356/1378 - Train Accuracy:  0.909, Validation Accuracy:  0.925, Loss:  0.436
    Epoch   1 Batch 1357/1378 - Train Accuracy:  0.933, Validation Accuracy:  0.930, Loss:  0.418
    Epoch   1 Batch 1358/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.923, Loss:  0.462
    Epoch   1 Batch 1359/1378 - Train Accuracy:  0.948, Validation Accuracy:  0.931, Loss:  0.420
    Epoch   1 Batch 1360/1378 - Train Accuracy:  0.902, Validation Accuracy:  0.926, Loss:  0.439
    Epoch   1 Batch 1361/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.934, Loss:  0.440
    Epoch   1 Batch 1362/1378 - Train Accuracy:  0.920, Validation Accuracy:  0.932, Loss:  0.425
    Epoch   1 Batch 1363/1378 - Train Accuracy:  0.945, Validation Accuracy:  0.932, Loss:  0.440
    Epoch   1 Batch 1364/1378 - Train Accuracy:  0.928, Validation Accuracy:  0.926, Loss:  0.459
    Epoch   1 Batch 1365/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.926, Loss:  0.409
    Epoch   1 Batch 1366/1378 - Train Accuracy:  0.923, Validation Accuracy:  0.928, Loss:  0.455
    Epoch   1 Batch 1367/1378 - Train Accuracy:  0.931, Validation Accuracy:  0.922, Loss:  0.412
    Epoch   1 Batch 1368/1378 - Train Accuracy:  0.938, Validation Accuracy:  0.924, Loss:  0.415
    Epoch   1 Batch 1369/1378 - Train Accuracy:  0.943, Validation Accuracy:  0.922, Loss:  0.425
    Epoch   1 Batch 1370/1378 - Train Accuracy:  0.946, Validation Accuracy:  0.917, Loss:  0.436
    Epoch   1 Batch 1371/1378 - Train Accuracy:  0.916, Validation Accuracy:  0.917, Loss:  0.407
    Epoch   1 Batch 1372/1378 - Train Accuracy:  0.934, Validation Accuracy:  0.930, Loss:  0.435
    Epoch   1 Batch 1373/1378 - Train Accuracy:  0.958, Validation Accuracy:  0.944, Loss:  0.445
    Epoch   1 Batch 1374/1378 - Train Accuracy:  0.910, Validation Accuracy:  0.929, Loss:  0.437
    Epoch   1 Batch 1375/1378 - Train Accuracy:  0.935, Validation Accuracy:  0.934, Loss:  0.416
    Epoch   1 Batch 1376/1378 - Train Accuracy:  0.922, Validation Accuracy:  0.941, Loss:  0.430
    Model Trained and Saved


### 保存参数

保存 `batch_size` 和 `save_path` 参数以进行推论（for inference）。


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# 检查点


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## 句子到序列

要向模型提供要翻译的句子，你首先需要预处理该句子。实现函数 `sentence_to_seq()` 以预处理新的句子。

- 将句子转换为小写形式
- 使用 `vocab_to_int` 将单词转换为 id
 - 如果单词不在词汇表中，将其转换为`<UNK>` 单词 id


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    seq = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.lower().split()]
    return seq


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## 翻译

将 `translate_sentence` 从英语翻译成法语。


```python
translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [83, 153, 195, 198, 32, 99, 59]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [82, 253, 207, 180, 308, 258, 129, 198, 1]
      French Words: ['il', 'a', 'vu', 'un', 'nouveau', 'camion', 'jaune', '.', '<EOS>']


## 不完美的翻译

你可能注意到了，某些句子的翻译质量比其他的要好。因为你使用的数据集只有 227 个英语单词，但实际生活中有数千个单词，只有使用这些单词的句子结果才会比较理想。对于此项目，不需要达到完美的翻译。但是，如果你想创建更好的翻译模型，则需要更好的数据。

你可以使用 [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar) 语料库训练模型。该数据集拥有更多的词汇，讨论的话题也更丰富。但是，训练时间要好多天的时间，所以确保你有 GPU 并且对于我们提供的数据集，你的神经网络性能很棒。提交此项目后，别忘了研究下 WMT10 语料库。


## 提交项目

提交项目时，确保先运行所有单元，然后再保存记事本。保存记事本文件为 “dlnd_language_translation.ipynb”，再通过菜单中的“文件” ->“下载为”将其另存为 HTML 格式。提交的项目文档中需包含“helper.py”和“problem_unittests.py”文件。

