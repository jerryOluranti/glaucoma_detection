from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import SwinForImageClassification, AutoFeatureExtractor
from evaluate import load, combine
# from datasets import load_dataset
import numpy as np

accuracy = load("accuracy")
metrics = combine([accuracy])

preds = []
labels = []


def compute_metrics(predictions, _labels):
    return metrics.compute(predictions=predictions, references=_labels)


def eval(image, label):
    images = tokenizer(images=image, return_tensors="pt")


    outputs = model(**images)

    prediction = np.argmax(outputs.logits.detach().numpy(), axis=-1);
    
    preds.append(prediction)
    labels.append(label)


tokenizer = AutoFeatureExtractor.from_pretrained("swin_odir_cropped_out/GLAUCOMASWINODIRCROPPED/10_2023-11-11-21-48-02/feature_extractor")
model = SwinForImageClassification.from_pretrained("swin_odir_cropped_out/GLAUCOMASWINODIRCROPPED/10_2023-11-11-21-48-02/model")


test_images, _, id2label, label2id = VisionDataset.fromImageFolder(
    "./eval/segmented",
    test_ratio=0,
    balanced=True,
    augmentation=True,
)

for image in test_images.dataset:
    eval(image[0], image[1])

accuracy = compute_metrics(preds, labels)["accuracy"]

print("accuracy: ", str(accuracy * 100), "%")