import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import ResNet152 as EncoderModel
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization, Input
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import absl.logging
import re
import random
from bs4 import BeautifulSoup
from tqdm import tqdm
import scipy.io

####################################################################################################
###> Remove warnings & info message...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)

####################################################################################################
###> Programm variables

# ACTIONS
TODO = [ 
    "from_scratch",
    "preprocess", 
    "train", 
    "evaluate", 
    "test", 
]

# DATASETS
DATASET_FOLDER     = os.path.join("datasets", "caltech-101")
IMAGES_FOLDER      = os.path.join(DATASET_FOLDER, "101_ObjectCategories")
ANNOTATIONS_FOLDER = os.path.join(DATASET_FOLDER, "Annotations")
CLASS_LIST         = [
    { "name": "Airplane", "imagesName": "airplanes", "annotationsName": "Airplanes_Side_2"},
    { "name": "Moto", "imagesName": "Motorbikes", "annotationsName": "Motorbikes_16"}
]
CLASSES            = { i : CLASS_LIST[i] | { "id": i } for i in range(len(CLASS_LIST)) }
ALLOWED_EXTENSIONS = [ ".jpg", ".jpeg", ".png" ]
TRAINING_RATIO     = .9
VALIDATION_RATIO   = .2
IMAGE_SIZE         = (224, 224)

# SAVES
SAVES_PATH                          = "saves"
GRAPHS_PATH                         = os.path.join(SAVES_PATH, "graphs/")
GRAPHS_TRAINING_LOSS_FILE_NAME            = "training_loss_history.png"
GRAPHS_CLASS_TRAINING_LOSS_FILE_NAME      = "class_training_loss_history.png"
GRAPHS_CLASS_TRAINING_ACCURACY_FILE_NAME  = "class_training_accuracy_history.png"
GRAPHS_BOX_TRAINING_LOSS_FILE_NAME        = "box_training_loss_history.png"
GRAPHS_BOX_TRAINING_ACCURACY_FILE_NAME    = "box_training_accuracy_history.png"
CHECKPOINTS_PATH                    = os.path.join(SAVES_PATH, "checkpoints/")
CHECKPOINTS_FILE_NAME               = "best_weights"

# TRAIN
TRAINING_PATIENCE   = 2
EPOCHS              = 25
BATCH_SIZE          = 32
DROPOUT             = .5
LEARNING_RATE       = 1e-4

# TEST
NUMBER_OF_IMAGES_TO_TEST    = 5
MIN_ID_TO_TEST              = 1
MAX_ID_TO_TEST              = 200

####################################################################################################
###> Launching the programm

print()
print("Starting...")
print("Actions todo: ", TODO)
print()

####################################################################################################
###> Clean previously saved files

if "from_scratch" in TODO:

    print("Removing files...")
    def removeFilesMatching(path):
        files = glob.glob(path)
        for file in files:
            os.remove(file)
        print("%d files removed matching pattern %s" % (len(files), path))
    removeFilesMatching(CHECKPOINTS_PATH + "/*")
    removeFilesMatching(GRAPHS_PATH + "/*")

if "train" in TODO:
    def createFolderIfNotExists(folder):
        if not(os.path.isdir(folder)):
            os.makedirs(folder)
    createFolderIfNotExists(CHECKPOINTS_PATH)
    createFolderIfNotExists(GRAPHS_PATH)

####################################################################################################
###> Filter images

if "preprocess" in TODO:

    def isCorrupted(fileimage):

        # with open(fileimage, "rb") as fobj:
        #     if not tf.compat.as_bytes("JFIF") in fobj.peek(10):
        #         return True

        try:
            with Image.open(fileimage) as img:
                img.verify()
            return False
        except:
            return True

    def removeInvalidImages(folder):

        num_total   = 0
        num_skipped = 0

        # Foreach images
        for f in os.listdir(folder):
            file = os.path.join(folder, f)
            if os.path.isfile(file):
                if not(
                    any([file.endswith(ext) for ext in ALLOWED_EXTENSIONS]) 
                        and 
                    not(isCorrupted(file))
                ):
                    os.remove(file)
                    num_skipped += 1
                    
                num_total += 1

        print("\nRemove bad formatted files...")
        print("Deleted %d / %d invalid images" % (num_skipped, num_total))

    for d_class in CLASSES.values():
        removeInvalidImages(os.path.join(IMAGES_FOLDER, d_class["imagesName"]))

####################################################################################################
###> Generate the datasets

def getPathsByIdFromFolder(folder):

    paths = {}

    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            currentId = list(map(int, re.findall(r'\d+', f)))[-1]
            paths[currentId] = full_path
    
    return paths

DATASETS_IMAGES_PATHS = {}
DATASETS_ANNOTATIONS_PATHS = {}
for d_class in CLASSES.values():
    DATASETS_IMAGES_PATHS |= getPathsByIdFromFolder(os.path.join(IMAGES_FOLDER, d_class["imagesName"]))
    DATASETS_ANNOTATIONS_PATHS |= getPathsByIdFromFolder(os.path.join(ANNOTATIONS_FOLDER, d_class["annotationsName"]))

def getPathsFromId(id):

    imagePath      = DATASETS_IMAGES_PATHS[id] if id in DATASETS_IMAGES_PATHS else None
    annotationPath = DATASETS_ANNOTATIONS_PATHS[id] if id in DATASETS_ANNOTATIONS_PATHS else None

    if not(imagePath) or not(annotationPath):
        print("\nWarning: Can't find the image with the id " + str(image_id_to_test))
        return None

    return imagePath, annotationPath

def getDClassIdFromAnnotationPath(annotation_path):
    for d_class in CLASSES.values():
        if d_class["annotationsName"] == annotation_path.split(os.sep)[-2]:
            vector = [ 0 for i in range(len(CLASSES))]
            vector[d_class["id"]] = 1
            return d_class["id"], np.array(vector, dtype="int32") 
    return None, None

def extractDataFromId(id):

    image_path, annotation_path = getPathsFromId(id)
        
    # Extract image
    image = keras.utils.load_img(image_path)
    image_w, image_h = image.size[:2]

    # Image processing
    processed_image = keras.utils.img_to_array(image.resize(IMAGE_SIZE))
    processed_image = np.array(processed_image, dtype="float32") / 255.0

    # Extract boxes
    box_data = tuple(scipy.io.loadmat(annotation_path)["box_coord"][0])
    min_x, min_y, max_x, max_y = box_data[3], box_data[1], box_data[2], box_data[0]
    box_data = np.array([ min_x, min_y, max_x, max_y ], dtype="float32")

    # Boxes processing : store position in %
    processed_box_data =  np.array([ min_x / image_w, min_y / image_h, max_x / image_w, max_y / image_h ], dtype="float32")

    # Extract class
    d_class_id, d_vector = getDClassIdFromAnnotationPath(annotation_path)

    return ( ( image, box_data ), (d_class_id, d_vector), ( processed_image, processed_box_data ) )

def plotRectangle(min_x, min_y, max_x, max_y, color, linestyle="solid"):

    plt.plot(
        [ min_x, max_x, max_x, min_x, min_x ],
        [ min_y, min_y, max_y, max_y, min_y ], 
        color=color, linestyle=linestyle
    )

def displayImageWithTargets(img, img_dim, tgts=None, preds=None, show=True, figure_name="Image with targets"):
    
    plt.figure(figure_name)
    plt.imshow(img.resize(img_dim))

    # Image dimentions
    img_w, img_h = img_dim
    plotRectangle(0, 0, img_w, img_h, color="grey", linestyle="solid")

    # Targets
    if not(tgts is None):    
        plt.text(25, 25, tgts["class"], bbox=dict(facecolor='black', alpha=0.5))
        min_x, min_y, max_x, max_y = tgts["box"]
        plotRectangle(min_x*img_w, min_y*img_h, max_x*img_w, max_y*img_h, color="black", linestyle="dashed")

    # Pred
    if not(preds is None):    
        plt.text(25, 50, preds["class"], bbox=dict(facecolor='red', alpha=0.5))
        min_x, min_y, max_x, max_y = preds["box"]
        plotRectangle(min_x*img_w, min_y*img_h, max_x*img_w, max_y*img_h, color="red", linestyle="dashed")

    if show:
        plt.show()

if "train" in TODO or "evaluate" in TODO:

    print("\nLoading the datasets...")

    images  = []
    boxes = []
    classes = []
    ids = list(DATASETS_IMAGES_PATHS.keys())
    random.shuffle(ids)

    for id in tqdm(ids):

        ( ( image, box_data ), (d_class_id, d_vector), ( processed_image, processed_box ) ) = extractDataFromId(id)
        images.append(processed_image)
        boxes.append(processed_box)
        classes.append(d_vector)

        # display_data = {
        #     "class": CLASSES[d_class_id]["name"],
        #     "box": processed_box
        # }
        # displayImageWithTargets(image, image.size, display_data, figure_name="Original (" + str(id) + ")", show=False)
        # displayImageWithTargets(image, IMAGE_SIZE, display_data, figure_name="Resized (" + str(id) + ")", show=False)
        # plt.show()

        pass
    
    # Separate training and testing data thanks to r1 ratio
    # Convert the list to numpy array, split to train and test dataset
    r = int(len(images)*TRAINING_RATIO)

    if "train" in TODO:
        x_train, y_boxes_train, y_classes_train = ( np.asarray(images[:r]), np.asarray(boxes[:r]), np.asarray(classes[:r]) )
        print("Working with %s images, including %s for training and %s for validation" % (len(images), int(len(x_train)*(1-VALIDATION_RATIO)), int(len(x_train)*VALIDATION_RATIO)))
    
    if "evaluate" in TODO:
        x_test, y_boxes_test, y_classes_test = ( np.asarray(images[r:]), np.asarray(boxes[r:]), np.asarray(classes[r:]) )
        print("Working with %s images, including %s for tests" % (len(images), len(x_test)))

####################################################################################################
###> Build a model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:
    
    def make_model(input_shape):
        
        encoder = EncoderModel(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
        encoder.trainable = False

        i = encoder.input
        vector = encoder.output
        vector = Flatten()(vector)

        box = Dense(128, activation="relu")(vector)
        box = Dense(64, activation="relu")(box)
        box = Dense(32, activation="relu")(box)
        box = Dense(4, activation="sigmoid", name="box")(box)

        d_class = Dense(512, activation="relu")(vector)
        d_class = Dropout(DROPOUT)(d_class)
        d_class = Dense(512, activation="relu")(d_class)
        d_class = Dropout(DROPOUT)(d_class)
        d_class = Dense(len(CLASSES), activation="softmax", name="class")(d_class)

        # construct the model we will fine-tune for bounding box regression
        return Model(inputs=i, outputs=(box, d_class))

    model = make_model(input_shape=IMAGE_SIZE + (3,))
    # keras.utils.plot_model(model, show_shapes=True)

    pass

####################################################################################################
###> Train the model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss = {
            "class": "categorical_crossentropy",
            "box": "mean_squared_error"
        },
        loss_weights = {
            "class": 1.0,
            "box": 1.0
        },
        metrics=["accuracy"]
    )

    # keras.utils.plot_model(model, show_shapes=True)

    pass

if "train" in TODO:
    
    print("\nTraining the model...")

    history = model.fit(
        x_train, { "class": y_classes_train, "box": y_boxes_train },
        validation_data=(x_test, { "class": y_classes_test, "box": y_boxes_test }),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_RATIO,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINTS_PATH, CHECKPOINTS_FILE_NAME), save_best_only=True, save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor="val_box_loss", restore_best_weights=True, patience=TRAINING_PATIENCE),
        ],
    )

    def visualizeLearningHistory(history, show=True):

        h = history.history

        plt.figure("Loss history")
        plt.plot(h["loss",],         color='red', label='Train loss')
        plt.plot(h['val_loss'],      color='green', label='Val loss')
        plt.legend()
        plt.title('Training and validation loss over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(GRAPHS_PATH, GRAPHS_TRAINING_LOSS_FILE_NAME))
  
        plt.figure("Class loss history")
        plt.plot(h["class_loss",],         color='red', label='Class train loss')
        plt.plot(h['val_class_loss'],      color='green', label='Class val loss')
        plt.legend()
        plt.title('Class training and validation loss over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(GRAPHS_PATH, GRAPHS_CLASS_TRAINING_LOSS_FILE_NAME))
  
        plt.figure("Box loss history")
        plt.plot(h["box_loss",],         color='red', label='Box train loss')
        plt.plot(h['val_box_loss'],      color='green', label='Box val loss')
        plt.legend()
        plt.title('Box training and validation loss over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(GRAPHS_PATH, GRAPHS_CLASS_TRAINING_LOSS_FILE_NAME))

        plt.figure("Class accuracy history")
        plt.plot(h['class_accuracy'],     color='red',   label='Class train accuracy')
        plt.plot(h['val_class_accuracy'], color='green', label='Class val accuracy')
        plt.legend()
        plt.title('Class training and validation accuracy over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(GRAPHS_PATH, GRAPHS_BOX_TRAINING_ACCURACY_FILE_NAME))

        plt.figure("Box accuracy history")
        plt.plot(h['box_accuracy'],     color='red',   label='Box train accuracy')
        plt.plot(h['val_box_accuracy'], color='green', label='Box val accuracy')
        plt.legend()
        plt.title('Box training and validation accuracy over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(GRAPHS_PATH, GRAPHS_BOX_TRAINING_ACCURACY_FILE_NAME))

        if show:
            plt.show()

    visualizeLearningHistory(history, show=True)

####################################################################################################
###> Load best model

if "evaluate" in TODO or "test" in TODO:

    print("\nLoading checkpoint %s" % (os.path.join(CHECKPOINTS_PATH, CHECKPOINTS_FILE_NAME)))
    model.load_weights(os.path.join(CHECKPOINTS_PATH, CHECKPOINTS_FILE_NAME))

####################################################################################################
###> Evaluate model

if "evaluate" in TODO:

    def evaluate_model(model, x_test, y_boxes_test, y_classes_test):
        
        print("\nEvaluating the model...")

        results = model.evaluate(
            x=x_test,
            y={
                "class": y_classes_test,
                "box": y_boxes_test
            },
            batch_size=BATCH_SIZE
        )

        loss, box_loss, class_loss, box_accuracy, class_accuracy = results
        print("Test loss: {:.5f}".format(loss))
        print("Test box loss: {:.5f}".format(box_loss))
        print("Test class loss: {:.5f}".format(class_loss))
        print("Test box accuracy: {:.2f}%".format(box_accuracy * 100))
        print("Test class accuracy: {:.2f}%".format(class_accuracy * 100))
        
    evaluate_model(model, x_test, y_boxes_test, y_classes_test)

####################################################################################################
###> Test model

if "test" in TODO:

    def test(id):

        print("\nTesting : " + str(id))

        ( ( img, box ), (d_class_id, d_vector), ( processed_img, processed_box ) ) = extractDataFromId(id)

        (img_w, img_h) = img.size[:2]
        img_array = tf.expand_dims(processed_img, 0)

        prediction_percentage_box, prediction_class_vector = model.predict(img_array)
        prediction_class_id = int(np.argmax(prediction_class_vector[0]))
        prediction_percentage_box = list(prediction_percentage_box[0])

        prediction_box = [ 
            prediction_percentage_box[0] * img_w, 
            prediction_percentage_box[1] * img_h, 
            prediction_percentage_box[2] * img_w, 
            prediction_percentage_box[3] * img_h
        ]
        class_name = CLASSES[d_class_id]["name"]
        prediction_class_name = CLASSES[prediction_class_id]["name"]

        print("Expected (class) : " + str(class_name))
        print("Predicted (class) : " + str(prediction_class_name))
        
        print("Expected (px) : " + str(box))
        print("Predicted (px) : " + str(prediction_box))

        print("Expected (%) : " + str(processed_box))
        print("Predicted (%) : " + str(prediction_percentage_box))

        answer_display_data = {
            "class": class_name,
            "box": processed_box
        }
        prediction_display_data = {
            "class": prediction_class_name,
            "box": prediction_percentage_box
        }

        displayImageWithTargets(img, img.size,   answer_display_data, prediction_display_data, show=False, figure_name="Original (" + str(id) + ")")
        displayImageWithTargets(img, IMAGE_SIZE, answer_display_data, prediction_display_data, show=False, figure_name="Resized (" + str(id) + ")")
        plt.show()

    # Add random images to test
    images_to_test = set()
    while len(images_to_test) != NUMBER_OF_IMAGES_TO_TEST:
        id = random.randint(MIN_ID_TO_TEST, MAX_ID_TO_TEST)
        if getPathsFromId(id) is not None:
            images_to_test.add(id)

    for image_id_to_test in images_to_test:
        test(image_id_to_test)
    
    plt.show()

####################################################################################################
###> Programm end message

print()
print("> Programm exited successfully!")