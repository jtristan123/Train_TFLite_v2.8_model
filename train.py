#this is the lastest version of the code
#this train the model ANND exports it to exported-model-v2
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader
import tensorflow as tf

# Load training data
train_data = DataLoader.from_pascal_voc(
    images_dir='images',
    annotations_dir='images',
    label_map={1: 'cone'}
)

# Create the model
spec = object_detector.EfficientDetLite0Spec()  # For Coral
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, epochs=100)

# Evaluate
model.evaluate(train_data)

# Export
model.export(export_dir='exported-model-v2')
