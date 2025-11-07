"""
Hand Gesture Recognition Pipeline:
1. Prepare LeapGestRecog dataset
2. Train RandomForest model (lightweight, no TensorFlow needed)
3. Run real-time webcam inference
"""
import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from PIL import Image
import shutil

# ----------------- USER CONFIG -----------------
DATASET_ZIP_OR_FOLDER = "./leapgestrecog"  # change if extracted elsewhere or zip file path
OUT_DIR = "./gesture_output"
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 8
EPOCHS_FINE = 12
FINE_TUNE_AT = 100  # layer number in base model to start fine-tuning
VAL_SPLIT = 0.2
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def discover_classes_and_images(root):
    """
    Many extracted Kaggle datasets put classes as subfolders under a folder named
    like 'leapGestRecog' or 'leapgestrecog'. This function tries to find images and class folders.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found; extract dataset first.")
    # find subfolders that contain images
    class_folders = [p for p in root.glob("*") if p.is_dir()]
    # If top-level contains many numbered folders, fall back to searching deeper
    if len(class_folders) == 0:
        raise RuntimeError("No class subfolders found in dataset root.")
    # return folder list
    return class_folders

def make_train_val_split(src_root, dest_root, val_split=0.2):
    """
    Create a tidy train/val structure under dest_root:
      dest_root/train/<class>/*.png
      dest_root/val/<class>/*.png
    Keeps stratified by folder.
    """
    src_root = Path(src_root)
    dest_root = Path(dest_root)
    train_root = dest_root / "train"
    val_root = dest_root / "val"
    if dest_root.exists():
        print(f"Removing previous dataset at {dest_root}")
        shutil.rmtree(dest_root)
    train_root.mkdir(parents=True)
    val_root.mkdir(parents=True)

    classes = [p for p in src_root.glob("*") if p.is_dir()]
    classes = sorted(classes)
    print(f"Found {len(classes)} classes.")
    for c in classes:
        imgs = [f for f in c.glob("*") if f.is_file()]
        random.shuffle(imgs)
        n_val = int(len(imgs) * val_split)
        train_imgs = imgs[n_val:]
        val_imgs = imgs[:n_val]
        (train_root / c.name).mkdir(parents=True, exist_ok=True)
        (val_root / c.name).mkdir(parents=True, exist_ok=True)
        for f in train_imgs:
            shutil.copy(f, train_root / c.name / f.name)
        for f in val_imgs:
            shutil.copy(f, val_root / c.name / f.name)
    return train_root, val_root

def build_generators(train_dir, val_dir, img_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.12,
        horizontal_flip=True,
        shear_range=0.05,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', seed=SEED
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen

def build_model(num_classes, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # train head first
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def fine_tune_model(model, base_model, fine_at, lr=1e-5):
    base_model.trainable = True
    # Freeze all layers before the `fine_at` layer
    for layer in base_model.layers[:fine_at]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def train_pipeline(dataset_root):
    # 1) prepare train/val folders
    train_dir, val_dir = make_train_val_split(dataset_root, OUT_DIR + "/data", VAL_SPLIT)
    # 2) generators
    train_gen, val_gen = build_generators(train_dir, val_dir)
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    print("Class names:", class_names)
    # 3) build model
    model, base_model = build_model(num_classes, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model.summary()
    cb = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    mc = callbacks.ModelCheckpoint(OUT_DIR + "/gesture_mobilenetv2_best.h5", save_best_only=True, monitor='val_accuracy')
    # 4) train head
    history_head = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD, callbacks=[cb, mc])
    # 5) fine-tune
    model = fine_tune_model(model, base_model, FINE_TUNE_AT, lr=1e-5)
    history_fine = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINE, callbacks=[cb, mc])
    # 6) evaluate
    print("Evaluating on validation set...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion(y_true, y_pred, class_names, OUT_DIR + "/confusion_matrix.png")
    # 7) save final model and class indices
    model.save(OUT_DIR + "/gesture_mobilenetv2.h5")
    np.save(OUT_DIR + "/class_indices.npy", val_gen.class_indices)
    print("Saved model and class indices to", OUT_DIR)
    return model, class_names

# ---------------- INFERENCE UTILITIES ----------------
def prepare_image_for_model(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255.0
    return img[np.newaxis, ...]

def inference_on_image(model, img_path, class_names):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    x = prepare_image_for_model(bgr)
    probs = model.predict(x)[0]
    idx = np.argmax(probs)
    return class_names[idx], probs[idx]

def inference_webcam(model, class_names, smoothing=5):
    """
    Realtime webcam prediction with simple temporal smoothing:
    keep last `smoothing` predictions and output the most common.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not available.")
        return
    last_preds = []
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = prepare_image_for_model(frame)
        probs = model.predict(x)[0]
        idx = int(np.argmax(probs))
        last_preds.append(idx)
        if len(last_preds) > smoothing:
            last_preds.pop(0)
        # smoothed prediction
        pred_idx = max(set(last_preds), key=last_preds.count)
        label = class_names[pred_idx]
        prob = probs[pred_idx]
        # overlay
        cv2.putText(frame, f"{label} {prob:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Gesture Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------- MAIN -----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_ZIP_OR_FOLDER, help="Path to extracted dataset root folder with class subfolders")
    p.add_argument("--train", action="store_true", help="Run training pipeline")
    p.add_argument("--infer_image", help="Path to single image to run inference")
    p.add_argument("--webcam", action="store_true", help="Run realtime webcam inference (requires trained model)")
    args = p.parse_args()

    if args.train:
        dataset_root = args.dataset
        model, class_names = train_pipeline(dataset_root)
    else:
        # load model and class indices if present
        model_path = OUT_DIR + "/gesture_mobilenetv2.h5"
        if not os.path.exists(model_path):
            raise RuntimeError("No trained model found. Run with --train first or place model at " + model_path)
        model = tf.keras.models.load_model(model_path)
        class_indices_path = OUT_DIR + "/class_indices.npy"
        class_indices = np.load(class_indices_path, allow_pickle=True).item() if os.path.exists(class_indices_path) else None
        if class_indices:
            # reorder class names by index
            class_names = sorted(class_indices, key=lambda k: class_indices[k])
        else:
            # fallback guess
            class_names = [str(i) for i in range(model.output_shape[-1])]

        if args.infer_image:
            label, prob = inference_on_image(model, args.infer_image, class_names)
            print("Prediction:", label, prob)
        if args.webcam:
            inference_webcam(model, class_names)
