from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTForImageClassification, TrainingArguments
import numpy as np
from evaluate import load
from datasets import load_dataset, load_metric

metric = load("accuracy")

model_path = "./out/GLAUCOMAVIT/20_2023-08-04-11-02-36/"


def compute_metrics(pred):
    predictions = np.argmax([pred.detach().numpy() for pred in pred.logits], axis=-1)
    print(len([1 for p in predictions if p == 1]), len([0 for p in predictions if p == 0]))
    return metric.compute(predictions=predictions)

# tokenizer = ViTFeatureExtractor.from_pretrained(model_path + "feature_extractor")
tokenizer = ViTImageProcessor.from_pretrained(model_path + "feature_extractor")
model = ViTForImageClassification.from_pretrained(model_path + "model")

test_images = load_dataset("imagefolder", data_dir="./test/")

image = tokenizer(images=test_images["test"]['image'], return_tensors="pt")

# print(model.config)

outputs = model(**image)

# print(outputs)

print(compute_metrics(outputs))

