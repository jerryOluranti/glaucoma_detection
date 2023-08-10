from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
from evaluate import load

metric = load("accuracy")

model_path = "./out/GLAUCOMAVIT/20_2023-08-04-11-02-36/"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenizer = ViTFeatureExtractor.from_pretrained(model_path + "feature_extractor")
model = ViTForImageClassification.from_pretrained(model_path + "model", return_dict=False)

test_image = "./test/refuge_test_4.jpg"
image = tokenizer(images=test_image, return_tensors="pt")

preds = model(**image)

print(compute_metrics(preds))