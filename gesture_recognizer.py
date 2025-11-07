"""
Hand Gesture Recognition Pipeline:
1. Prepare LeapGestRecog dataset (using a manageable subset)
2. Train RandomForest model (lightweight, no TensorFlow needed)
3. Run real-time webcam inference
"""

# Consolidated imports and constants
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

# Constants
IMG_SIZE = (64, 64)  # Smaller size for faster processing
DATA_DIR = 'leapGestRecog'  # Raw dataset
PREPARED_DIR = 'data/prepared_gestures'  # Prepared dataset
SEED = 42
# --- MODIFIED THIS LINE ---
MAX_IMAGES_PER_CLASS = 1  # Limit images per class to manage memory
# --- END MODIFICATION ---

# Ensure reproducibility
np.random.seed(SEED)

def prepare_dataset(src_path=DATA_DIR, dst_path=PREPARED_DIR):
    """Prepare LeapGestRecog dataset into a clean format."""
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    
    dst.mkdir(parents=True, exist_ok=True)
    
    # Get all gesture folders
    gesture_folders = [d for d in src.iterdir() if d.is_dir()]
    print(f"Found {len(gesture_folders)} gesture folders")
    
    for gesture_dir in gesture_folders:
        gesture_name = gesture_dir.name
        dst_gesture = dst / gesture_name
        dst_gesture.mkdir(exist_ok=True)
        
        # Copy and resize all images from this gesture
        count = 0
        # Find all images recursively (to handle subject folders)
        for img_path in gesture_dir.rglob('*.png'):
                if count >= MAX_IMAGES_PER_CLASS:
                    break
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    dst_file = dst_gesture / f"{gesture_name}_{count:04d}.png"
                    img.save(dst_file)
                    count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {count} images for gesture: {gesture_name}")


def load_dataset(data_path=PREPARED_DIR):
    """Load prepared dataset into memory."""
    X = []  # Images (flattened)
    y = []  # Labels (numeric)
    labels = []  # Label names
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run prepare_dataset() first.")
    
    for idx, class_dir in enumerate(sorted(data_dir.iterdir())):
        if not class_dir.is_dir():
            continue
            
        labels.append(class_dir.name)
        class_files = list(class_dir.glob('*.png'))

        # Limit the number of images per class
        class_files = class_files[:MAX_IMAGES_PER_CLASS]
        print(f"Loading {len(class_files)} images (limited) from {class_dir.name}")

        for img_path in class_files:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img)
            X.append(img_array.flatten())  # Flatten to 1D for RandomForest
            y.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    return X, y, labels


def train_model(X, y, labels):
    """Train a RandomForest classifier."""
    
    # --- Warning for small dataset ---
    if len(np.unique(y)) <= 1 or len(y) < 10:
        print("\nWARNING: Dataset is too small (or has only one class) for a train/test split.")
        print("Training on the whole (tiny) dataset.")
        X_train, y_train = X, y
        X_test, y_test = X, y # Test on training data for a basic check
    else:
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    # --- End modification ---
    
    # Train model
    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    
    # Detailed evaluation
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    # Add zero_division=0 to handle cases where a class has no test/pred samples
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    
    return clf


def save_model(clf, labels, save_dir='models'):
    """Save the trained model and labels."""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'gesture_model.joblib')
    labels_path = os.path.join(save_dir, 'gesture_labels.txt')
    
    joblib.dump(clf, model_path)
    with open(labels_path, 'w') as f:
        f.write('\n'.join(labels))
    
    print(f"\nSaved model to {model_path}")
    print(f"Saved labels to {labels_path}")
    return model_path, labels_path


def run_webcam_demo(model_path='models/gesture_model.joblib', 
                    labels_path='models/gesture_labels.txt'):
    """Run real-time webcam demo."""
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("Model or labels not found. Train the model first.")
        return
    
    # Load model and labels
    clf = joblib.load(model_path)
    with open(labels_path, 'r') as f:
        labels = f.read().splitlines()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("\nRunning webcam demo (press 'q' to quit)...")
    print("Place your hand in the center of the frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a square ROI in the center
        h, w = frame.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        roi = frame[y:y+size, x:x+size]
        
        # Prepare ROI for prediction
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize(IMG_SIZE)
        img_array = np.array(img).flatten().reshape(1, -1)
        
        # Predict
        pred = clf.predict(img_array)[0]
        pred_label = labels[pred]
        
        # Draw ROI and prediction
        cv2.rectangle(frame, (x, y), (x+size, y+size), (0,250,0), 2)
        cv2.putText(frame, pred_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)
        
        # Show frame if GUI is available; otherwise print prediction to console
        try:
            cv2.imshow('Gesture Recognition Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # Fallback for environments without GUI support (headless)
            print(f"Prediction: {pred_label}")
            # slow down loop so console isn't flooded
            import time
            time.sleep(0.2)
            # allow user to stop with KeyboardInterrupt
            continue
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("\n=== Hand Gesture Recognition Demo ===")
    print("Using LeapGestRecog dataset")
    
    try:
        # 1. Prepare dataset
        print("\n=== Preparing Dataset ===")
        prepare_dataset()
        
        # 2. Load dataset
        print("\n=== Loading Dataset ===")
        X, y, labels = load_dataset()
        print(f"\nLoaded {len(X)} samples, {len(labels)} classes")
        
        if len(X) == 0:
            print("No data was loaded. Please check the dataset path and format.")
            return

        # 3. Train model
        print("\n=== Training Model ===")
        clf = train_model(X, y, labels)
        
        # 4. Save model
        print("\n=== Saving Model ===")
        model_path, labels_path = save_model(clf, labels)
        
        # 5. Run demo
        print("\n=== Running Demo ===")
        run_webcam_demo(model_path, labels_path)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == '__main__':
    main()