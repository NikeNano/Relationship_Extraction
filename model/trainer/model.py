import os 

import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, metadata_io,dataset_schema 
from tensorflow_transform.beam.tft_beam_io import transform_fn_io

# TO DO
# 1) Do the traning and eval functions
# 2) Test train the model 
# 3) Add the serving function 
# 4) 

def transform_metadata(folder="gs://relation_extraction/beam/"):
    """Read the transform metadata"""
    transformed_metadata = metadata_io.read_metadata(
    os.path.join(
        folder, transform_fn_io.TRANSFORMED_METADATA_DIR
        )
    )
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()
    return transformed_feature_spec


def train_input_fn(train_folder=None,model_dir_beam=None,batch_size=None):
    """Function to generate the input traning data"""
    transformed_feature_spec = transform_metadata()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=os.path.join(train_folder,"TRAIN*"),
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=True
        )
    transformed_features = dataset.make_one_shot_iterator().get_next()
    # itterator as output here
    # Should i replace this step with a parsing function instead? 
    # I belive so, think that should make it faster as well. 
    # That way it should be way better I think at least .... 
    transformed_labels = {key: value for (key, value) in transformed_features.items() if key in ["labels"]}
    transformed_features = {key: value for (key, value) in transformed_features.items() if key not in ["labels"]}    
    return transformed_features, transformed_labels


def eval_input_fn(test_folder=None,model_dir_beam=None,batch_size=None):
    transformed_feature_spec = transform_metadata()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=os.path.join(test_folder,"TEST*"),
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=True
        )
    transformed_features = dataset.make_one_shot_iterator().get_next()
    transformed_labels = {key: value for (key, value) in transformed_features.items() if key in ["labels"]}
    transformed_features = {key: value for (key, value) in transformed_features.items() if key not in ["labels"]}    
    return transformed_features, transformed_labels


def serve_input_fn():
    pass


def padd(input_tensor=None,tensor_length=None):
    """Functon to padd the inputs with. Assume the data to be of shaoe 2."""
    paddings = tf.constant([[0, 0,], [0,tensor_length]])
    input_tensor = tf.pad(input_tensor,paddings,"CONSTANT")
    output_tensor = tf.slice(input_tensor,[0,0],[tf.shape(input_tensor)[0],tensor_length])
    return output_tensor


def cnn_model(features,labels,mode,params):
    DEFAULT_WORD_VALUE = params.default_word_value
    VOCAB_SIZE = params.vocab_size
    MAX_SENTENCE_LENGTH = params.max_sentence_length # NEED TO SET THIS IN THE BEAM JOB
    EMBED_WORD_POSITIONS = int(3) # CHANGE THIS LATER
    embed_dim_words=max(8,int(VOCAB_SIZE**(1/4))) #
    N_CLASSES = params.n_classes
    conv1_filter=5
    conv2_filter=4
    conv3_filter=2 
    sentence = tf.sparse.to_dense(features["word_representation"],default_value=DEFAULT_WORD_VALUE)
    input_sentences = padd(sentence,MAX_SENTENCE_LENGTH)
    input_tail_positions = padd(features["distance_to_tail"],MAX_SENTENCE_LENGTH)
    input_head_positions = padd(features["distance_to_head"],MAX_SENTENCE_LENGTH)
    word_embedding = tf.contrib.layers.embed_sequence(
        input_sentences, vocab_size=VOCAB_SIZE, embed_dim=embed_dim_words)
    head_position_embedding = tf.contrib.layers.embed_sequence(
        input_head_positions, vocab_size=2*(MAX_SENTENCE_LENGTH-1)-1, embed_dim=EMBED_WORD_POSITIONS)
    tail_position_embedding = tf.contrib.layers.embed_sequence(
        input_tail_positions, vocab_size=2*(MAX_SENTENCE_LENGTH-1)-1, embed_dim=EMBED_WORD_POSITIONS)
    input_tensor = tf.concat([word_embedding,head_position_embedding,tail_position_embedding],2)
    input_tensor = tf.expand_dims(input_tensor, 3)
    complete_embed_size = embed_dim_words + 2 * EMBED_WORD_POSITIONS
    conv1 = tf.layers.conv2d(
      inputs=input_tensor,
      filters=2,
      kernel_size=[conv1_filter, complete_embed_size],
      activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[MAX_SENTENCE_LENGTH-conv1_filter+1, 1], strides=1)
    conv2 = tf.layers.conv2d(
      inputs=input_tensor,
      filters=2,
      kernel_size=[conv2_filter, complete_embed_size],
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[MAX_SENTENCE_LENGTH-conv2_filter+1, 1], strides=1)
    conv3 = tf.layers.conv2d(
      inputs=input_tensor,
      filters=2,
      kernel_size=[conv3_filter, complete_embed_size],
      activation=tf.nn.relu,)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[MAX_SENTENCE_LENGTH-conv3_filter+1, 1], strides=1)
    concat = tf.reshape(tf.concat([pool1,pool2,pool3],1),[-1,6])
    head = tf.contrib.estimator.multi_class_head(
        n_classes = N_CLASSES
        )
    logits = tf.layers.dense(concat,head.logits_dimension,activation=None)    
    optimizer = tf.train.AdamOptimizer() 
    
    
    def _train_op_fn(loss):
        tf.summary.scalar('loss', loss)
        return optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
    return head.create_estimator_spec(
        features=features,
        labels=features["relation"],
        mode=mode,
        logits=logits, 
        train_op_fn=_train_op_fn)

