import os
import argparse
from datetime import datetime
import ast
import numpy as np
import re
import json
import random

import tensorflow_transform as tft
import apache_beam as beam
import tensorflow as tf
from random import randint
from apache_beam.io import tfrecordio
from apache_beam.io.gcp.internal.clients import bigquery
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata, metadata_io,dataset_schema 

DELIMITERS_WORDS=' '
SENTENCE_MAX_LENGTH=50

TRAIN_DATA_SCHEMA = dataset_schema.from_feature_spec({
    "text":  tf.FixedLenFeature(shape=[], dtype=tf.string),
    "head":  tf.FixedLenFeature(shape=[], dtype=tf.string),
    "taill": tf.FixedLenFeature(shape=[], dtype=tf.string),
    "distance_to_head": tf.FixedLenFeature(shape=[SENTENCE_MAX_LENGTH], dtype=tf.int64),
    "distance_to_tail": tf.FixedLenFeature(shape=[SENTENCE_MAX_LENGTH], dtype=tf.int64),
    "sentence_length": tf.FixedLenFeature(shape=[],dtype=tf.int64)
    })
train_metadata = dataset_metadata.DatasetMetadata(schema=TRAIN_DATA_SCHEMA)

class PadList(list):
    """ The super padding list used for padding data
    """
    def inner_pad(self, pad_length, pad_value=0):
        """Do inner padding of the list
        
        Paramters:
            padd_length -- How long should the list be
            padd_value -- What value should be used for the padding
        
        Return:
            self -- the list 
        """
        nbr_pad = pad_length - len(self)
        if nbr_pad>0:
            self = self + [pad_value] * nbr_pad
        else:
            self=self[:pad_length]
        return self
    
    
    def outer_pad(self,padded_list_length,pad_length,pad_value=0):
        """
        Out padding of a list e.g append a list to a list. 
        Args:
            padded_list_length -- how long should the appended list be
            pad_lenght -- how long should the list be e.g how much should we append
            padd_value -- What should the appended list have as values
            
        """
        nbr_pad = pad_length-len(self)
        if nbr_pad > 0:
            for _ in range(nbr_pad):
                self.append([pad_value] * padded_list_length)
        else:
            self = self[:pad_length]
        return self



def get_cloud_pipeline_options():
    """Get apache beam pipeline options to run with Dataflow on the cloud
    Args:
        project (str): GCP project to which job will be submitted
    Returns:
        beam.pipeline.PipelineOptions
    """
    options = {
        'runner': 'DataflowRunner',
        'job_name': ('preprocessdigitaltwin-{}'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))),
        'staging_location': "gs://tommy_dummy_ml_engine/binaries/",
        'temp_location':  "gs://tommy_dummy_ml_engine/tmp/",
        'project': "iotpubsub-1536350750202",
        'region': 'europe-west1',
        'zone': 'europe-west1-b',
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'save_main_session': True,
        'setup_file': './setup.py',
    }
    return beam.pipeline.PipelineOptions(flags=[], **options)


class SplitSentence(beam.DoFn):
    def process(self,element):
        sentence = element["sentence"].split()
        index_head = [sentence.index(word) for word in element["head"].split()]
        index_tail = [sentence.index(word) for word in element["tail"].split()]
        distance_to_head = PadList(abs(index_head[0]-index)for index,word in enumerate(sentence)).inner_pad(50,-1)
        distance_to_tail = PadList(abs(index_tail[0]-index)for index,word in enumerate(sentence)).inner_pad(50,-1)
        return [{
            "text":element["sentence"].lower(),
            "head":element["head"],
            "taill":element["tail"],
            "distance_to_head":distance_to_head,
            "distance_to_tail":distance_to_tail,
            "sentence_length":len(sentence)
        }]


class ReadBigQuery(beam.PTransform):
    """Read from BigQuery and do split to list"""
    def expand(self,pcoll):
        table_spec = bigquery.TableReference(
        projectId='iotpubsub-1536350750202',
        datasetId='baybenames',
        tableId='relation_extraction_data')
        return  (
            pcoll
            |'Read in put table' >> beam.io.Read(beam.io.BigQuerySource(table_spec))
            |'Split words' >> beam.ParDo(SplitSentence())
            |'Split test and training data' >>  beam.Partition(lambda element, _: 0 if randint(0,100)<80 else 0,2 )
            )


def printy(element):
    print(element)


def preprocessing_fn(inputs):
    words = tf.string_split(inputs['text'],DELIMITERS_WORDS)
    word_representation = tft.compute_and_apply_vocabulary(words,default_value=0,top_k=10000)
    outputs = inputs.copy()
    outputs["word_representation"] = word_representation
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', type=str, help='y' )
    args = parser.parse_args(argv) # Parse the arguments 
    if args.cloud=="y":
        pipeline_options = get_cloud_pipeline_options()
    else:
        pipeline_options = beam.pipeline.PipelineOptions(flags=[],**{'project': "iotpubsub-1536350750202"})
    with beam_impl.Context(temp_dir="gs://named_entity_recognition/beam/"):
        p = beam.Pipeline(options=pipeline_options)
        train_data, test_data = (p | "Read from bigquery" >> ReadBigQuery())
        train_data = (train_data, train_metadata)
        train_dataset, transform_fn = (train_data
                                            | 'AnalyzeAndTransform' >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn)
                                            )
        test_data = (test_data, train_metadata)
        test_data, _ = ((test_data, transform_fn) | 'Transform test data' >> beam_impl.TransformDataset())
        train_data, transformed_metadata = train_dataset
        transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)
        _ = (train_data
                | 'Encode train data to save it' >> beam.Map(transformed_data_coder.encode)
                | 'Write the train data to tfrecords' >> tfrecordio.WriteToTFRecord(os.path.join("gs://relation_extraction/beam/","TRAIN"))
                )
        _ = (test_data
                | 'Encode test data to save it' >> beam.Map(transformed_data_coder.encode)
                | 'Write the test data to tfrecords' >> tfrecordio.WriteToTFRecord(os.path.join("gs://relation_extraction/beam/","TEST"))
                )
        _ = (transform_fn | "WriteTransformFn" >> transform_fn_io.WriteTransformFn("gs://relation_extraction/beam/"))

        p.run().wait_until_finish()

if __name__=="__main__":
    main()
