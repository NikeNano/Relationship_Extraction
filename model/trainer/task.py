import argparse
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.contrib.training.python.training import hparam
from trainer.model import cnn_model,train_input_fn,eval_input_fn #,build_serving_input_fn


def model_setup(params=None):
	"""Model set up function returns an custom estimator"""
	run_config = tf.estimator.RunConfig(save_checkpoints_secs = 120, 
	                                    keep_checkpoint_max = 3)
	cnn_classifier = tf.estimator.Estimator(model_fn = cnn_model,
	                                        model_dir = params.job_dir,
	                                        params = hparams,
	                                        config = run_config)
	return cnn_classifier


def train_setup(hparams = None,classifier=None):
	#hook = tf.contrib.estimator.stop_if_no_increase_hook(classifier, 'f1', 500, min_steps = 800, run_every_secs = 120)
	train_spec = tf.estimator.TrainSpec(input_fn = lambda:train_input_fn(
			train_folder = hparams.train_folder,
			model_dir_beam = hparams.model_dir_beam,
			batch_size = int(hparams.batch_size)
			),
		max_steps = int(hparams.max_steps),
        #hooks=[hook]
		)
	return train_spec


def eval_setup(hparams = None):
	#tf_transform_beam = tft.TFTransformOutput(hparams.model_dir_beam)
	#serving_input_fn = build_serving_input_fn(tf_transform_beam = tf_transform_beam,params = hparams)
	#exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
	eval_spec = tf.estimator.EvalSpec(input_fn = lambda:eval_input_fn(
			test_folder = hparams.eval_folder,
			model_dir_beam = hparams.model_dir_beam,
			batch_size = int(hparams.batch_size)
			),
		start_delay_secs = 10, 
		throttle_secs = 20,
		#exporters = exporter
		)
	return eval_spec


def main(hparams):
	tf.logging.set_verbosity(tf.logging.INFO)
	classifier = model_setup(params = hparams)
	train_spec = train_setup(hparams=hparams,classifier=classifier)
	eval_spec = eval_setup(hparams=hparams)   
	tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help = 'GCS location to write checkpoints and export models',
        default = "gs://relation_extraction/ml_engine"
        )
    parser.add_argument(
        '--model-dir-beam',
        help = 'path to the model dir, where the beam job stored the outputs',
        default = "gs://relation_extraction/beam"
        )
    parser.add_argument(
        '--train-folder',
        help='path to the train folder',
        default = "gs://relation_extraction/beam/Train"
        )
    parser.add_argument(
        '--eval-folder',
        help='path to the eval folder',
        default = "gs://relation_extraction/beam/Test"
        )
    parser.add_argument(
        '--batch-size',
        help = 'the batch size',
        default = 20)
    parser.add_argument(
        '--max-sentence-length',
        help='the max length of a sentence',
        default = 50)
    parser.add_argument(
        '--vocab_size',
        help='this is based upon the preprocessing job',
        default = 1e4
        )
    parser.add_argument(
        '--default-word-value',
        help='what should be used during padding as defualt value',
        default = 0)
    parser.add_argument(
        '--n-classes',
        help='How many classes do we have',
        default = 56)
    parser.add_argument(
        '--max-steps',
        help='How many classes do we have',
        default = 200)      
        
    args, _ = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparam.HParams(**args.__dict__)
    main(hparams=hparams)