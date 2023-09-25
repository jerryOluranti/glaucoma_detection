from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from evaluate import load, combine
import numpy as np

accuracy = load("accuracy")
f1 = load("f1")
precision = load("precision")
recall = load("recall")

metrics = combine([accuracy, f1, precision, recall])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)
    # predictions = np.argmax([pred.detach().numpy() for pred in pred.logits], axis=-1)
    # return metrics.compute(predictions=predictions)

train, test, id2label, label2id = VisionDataset.fromImageFolder(
  "./dataset_lag/",
  test_ratio   = 0.25,
  balanced     = True,
  augmentation = True,
)

model_path = "./out/GLAUCOMAVIT/20_2023-08-04-11-02-36/"

trainer = VisionClassifierTrainer(
	model_name   = "GlaucomaViT-Enhanced",
	train        = train,
	test         = test,
	output_dir   = "./out_enhanced/",
	max_epochs   = 10,
	batch_size   = 32, # On RTX 2080 Ti
	lr	     = 2e-5,
	fp16	     = False,
	model = ViTForImageClassification.from_pretrained(
	    model_path + "model",
	    num_labels = len(label2id),
	    label2id   = label2id,
	    id2label   = id2label
	),
	feature_extractor = ViTFeatureExtractor.from_pretrained(
		model_path + "feature_extractor",
	),
  # compute_metrics = compute_metrics
)


trainer.evaluate()
ref, hyp = trainer.evaluate_f1_score()

print(ref, hyp)
