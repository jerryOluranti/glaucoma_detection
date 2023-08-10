from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification

train, test, id2label, label2id = VisionDataset.fromImageFolder(
  "./grouped/",
  test_ratio   = 0.20,
  balanced     = True,
  augmentation = True,
)

huggingface_model = 'google/vit-base-patch16-224-in21k'

trainer = VisionClassifierTrainer(
	model_name   = "GlaucomaViT",
	train        = train,
	test         = test,
	output_dir   = "./out/",
	max_epochs   = 20,
	batch_size   = 32, # On RTX 2080 Ti
	lr	     = 2e-5,
	fp16	     = False,
	model = ViTForImageClassification.from_pretrained(
	    huggingface_model,
	    num_labels = len(label2id),
	    label2id   = label2id,
	    id2label   = id2label
	),
	feature_extractor = ViTFeatureExtractor.from_pretrained(
		huggingface_model,
	),
)
