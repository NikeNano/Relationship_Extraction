import os
import argparse
from datetime import datetime
import ast

import apache_beam as beam
from tensorflow_transform.beam import impl as beam_impl

def get_cloud_pipeline_options():
    """Get apache beam pipeline options to run with Dataflow on the cloud
    Args:
        project (str): GCP project to which job will be submitted
    Returns:
        beam.pipeline.PipelineOptions
    """
    logging.warning('Start running in the cloud')

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

def data_folders(path_data="/Users/Niklas/Documents/Relationship_Extraction/Data"):
    """Get the paths for the data"""
    folders=["Documents","Entities","Relations"]
    paths={}
    for folder in folders:
        paths[folder]=(os.path.join(path_data,folder,"*.json"))
    return paths


class Split_key(beam.DoFn):
    def process(self,element):
        input = ast.literal_eval(element)
        id=input["_id"].split("-")[0]
        return [(id,input)]


class Create_key_value(beam.DoFn):
    def process(self,element):
        input = ast.literal_eval(element)
        return [(input["_id"],input)]


def printer(x):
    """The dummy printer we use """
    print("The printer")
    print(x)


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

    folders = data_folders()

    with beam_impl.Context(temp_dir="tmp"):
        documents = (p
            | 'Read the documents' >> beam.io.ReadFromText(folders["Documents"])
            | 'Create key value_pair documents' >> beam.ParDo(Create_key_value())
            )
        relations = (p
            | 'Read the relations' >> beam.io.ReadFromText(folders["Relations"])
            | 'Split the key' >> beam.ParDo(Split_key())
            | 'Group all relations' >> beam.GroupByKey()
            )
        entities = (p
            | 'Read the entities' >> beam.io.ReadFromText(folders["Entities"])
            | 'Split the entitie keys' >> beam.ParDo(Split_key())
            | 'Group all entities' >> beam.GroupByKey()
            )
        complete_data = ({'entities': entities, 'relations': relations,'documents': documents}
           | beam.CoGroupByKey())
        
        complete_data | "print the shit" >> beam.Map(printer)

        # To do now break it up and make sure it is in the correct format 
        # Break it up and fix it
        p.run().wait_until_finish()


if __name__=="__main__":
    main()