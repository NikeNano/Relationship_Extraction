import os
import argparse
from datetime import datetime
import ast
import numpy as np
import re
import json

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import tfrecordio
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata, metadata_io,dataset_schema 
from apache_beam.io.gcp.internal.clients import bigquery

DATASET_ID = "baybenames"
PROJECT_ID = "iotpubsub-1536350750202"

# TO DO
# 1) Start on the second preprocessing that access the BigQuery table
# 2) Need to run it on a larger dataset later on. 
#  

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

class Parse_json(beam.DoFn):
    def process(self,element):
        data = ast.literal_eval(element)
        output={}
        output["head"] = data["head"]["word"].decode('utf8', 'ignore')
        output["head_type"] = data["head"]["type"].decode('utf8', 'ignore')
        output["relation"] = data["relation"].decode('utf8', 'ignore')
        output["tail"] = data["tail"]["word"].decode('utf8', 'ignore')
        output["tail_type"] = data["tail"]["type"].decode('utf8', 'ignore')
        output["sentence"] = data["sentence"].decode('utf8', 'ignore')
        return [output]

class SaveToBigQuery(beam.PTransform):
    def expand(self,pcoll):
        table_schema = bigquery.TableSchema()

        head_schema = bigquery.TableFieldSchema()
        head_schema.name = 'head'
        head_schema.type = 'string'
        head_schema.mode = 'nullable'
        table_schema.fields.append(head_schema)

        head_type_schema = bigquery.TableFieldSchema()
        head_type_schema.name = 'head_type'
        head_type_schema.type = 'string'
        head_type_schema.mode = 'nullable'
        table_schema.fields.append(head_type_schema)
    
        relation_schema = bigquery.TableFieldSchema()
        relation_schema.name = 'relation'
        relation_schema.type = 'string'
        relation_schema.mode = 'nullable'
        table_schema.fields.append(relation_schema)

        tail_schema = bigquery.TableFieldSchema()
        tail_schema.name = 'tail'
        tail_schema.type = 'string'
        tail_schema.mode = 'nullable'
        table_schema.fields.append(tail_schema)

        tail_type_schema = bigquery.TableFieldSchema()
        tail_type_schema.name = 'tail_type'
        tail_type_schema.type = 'string'
        tail_type_schema.mode = 'nullable'
        table_schema.fields.append(tail_type_schema)

        sentence_schema = bigquery.TableFieldSchema()
        sentence_schema.name = 'sentence'
        sentence_schema.type = 'string'
        sentence_schema.mode = 'nullable'
        table_schema.fields.append(sentence_schema)

        return (pcoll 
                | 'Parse the json lines' >> beam.ParDo(Parse_json())
                | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                table='relation_extraction_data',
                dataset=DATASET_ID,
                project=PROJECT_ID,
                schema=table_schema,  # Pass the defined table_schema
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
                )


# HERE https://stackoverflow.com/questions/45998157/apache-beam-python-streaming-to-bigquery-writes-no-data-to-the-table 


def printy(element):
    print(element)


def main(argv=None):
    """Run preprocessing as a Dataflow pipeline.
    Args:
        argv (list): list of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', type=str, help='y' )
    args = parser.parse_args(argv) # Parse the arguments 

    if args.cloud=="y":
        pipeline_options = get_cloud_pipeline_options()
    else:
        pipeline_options = None
    p = beam.Pipeline(options=pipeline_options)

    with beam_impl.Context(temp_dir="tmp"):
        documents = (p
            | 'Read the documents' >> beam.io.ReadFromText("/Users/Niklas/Documents/Relationship_Extraction/Data/Sample_data.txt")
            | 'Store in bigquery' >> SaveToBigQuery()
            )
        p.run().wait_until_finish()


if __name__=="__main__":
    main()