import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import uuid
from werkzeug.utils import secure_filename
import time
from pathlib import Path

# New imports for post-processing features
import exifread
from PIL.ExifTags import TAGS
import io
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize
import json
import datetime

# Set up Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = "static/results"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload size
app.config["MODEL_PATH"] = "deepfake_detector_model.pth"

# Ensure upload and result directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for model
model = None
transform = None
model_loaded = False


# Define the model architecture
def get_model(model_path="deepfake_detector_model.pth", num_classes=2):
    # Load the same architecture as used during training
    model = models.resnet50(weights=None)

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, num_classes))

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


# Define image transformation
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# Initialize model and transform
def init_model():
    global model, transform, model_loaded
    if model is None:
        print("Loading model...")
        try:
            if not os.path.exists(app.config["MODEL_PATH"]):
                print(f"Model file not found at {app.config['MODEL_PATH']}")
                return False

            model = get_model(app.config["MODEL_PATH"])
            transform = get_transform()
            model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
            model_loaded = False
            return False
    return model_loaded


# Check allowed file
def allowed_file(filename, type="image"):
    if type == "image":
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
        )
    elif type == "video":
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
        )
    return False


# New post-processing functions


def extract_metadata(image_path):
    """Extract metadata from the input image"""
    metadata = {}

    try:
        # Open the image file for EXIF data
        with open(image_path, "rb") as f:
            exif_tags = exifread.process_file(f, details=False)

            # Extract relevant EXIF data
            if exif_tags:
                for tag, value in exif_tags.items():
                    # Skip thumbnail data
                    if tag.startswith("Thumbnail"):
                        continue
                    metadata[tag] = str(value)

        # Get additional image information
        img = Image.open(image_path)
        metadata["ImageFormat"] = img.format
        metadata["ImageMode"] = img.mode
        metadata["ImageSize"] = f"{img.width}x{img.height} pixels"
        metadata["FileSize"] = f"{os.path.getsize(image_path) / 1024:.1f} KB"

        # Get file creation and modification dates
        stat = os.stat(image_path)
        metadata["ModificationDate"] = datetime.datetime.fromtimestamp(
            stat.st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Compression and format details
        try:
            if hasattr(img, "info"):
                if "compression" in img.info:
                    metadata["Compression"] = img.info["compression"]
                if "dpi" in img.info:
                    metadata["DPI"] = str(img.info["dpi"])
                if "quality" in img.info:
                    metadata["Quality"] = str(img.info["quality"])
        except:
            pass

        # Clean and organize metadata
        cleaned_metadata = {
            "Basic Information": {
                "Format": metadata.get("ImageFormat", "Unknown"),
                "Dimensions": metadata.get("ImageSize", "Unknown"),
                "File Size": metadata.get("FileSize", "Unknown"),
                "Mode": metadata.get("ImageMode", "Unknown"),
                "Last Modified": metadata.get("ModificationDate", "Unknown"),
            },
            "EXIF Data": {},
        }

        # Process EXIF data into categories
        exif_categories = {
            "Camera": ["Make", "Model", "LensModel", "LensInfo"],
            "Capture": [
                "DateTime",
                "DateTimeOriginal",
                "ExposureTime",
                "FNumber",
                "ISOSpeedRatings",
                "FocalLength",
            ],
            "Image": ["XResolution", "YResolution", "Software", "Artist", "Copyright"],
        }

        # Organize EXIF data into categories
        for tag, value in metadata.items():
            if tag in TAGS.values() or tag in exif_tags:
                for category, tags in exif_categories.items():
                    if any(t in tag for t in tags):
                        if category not in cleaned_metadata["EXIF Data"]:
                            cleaned_metadata["EXIF Data"][category] = {}
                        cleaned_metadata["EXIF Data"][category][tag] = value
                        break

        return cleaned_metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {"Basic Information": {"Error": f"Metadata extraction failed: {str(e)}"}}


def generate_explanation(is_fake, confidence, metadata):
    """Generate reasoning for the model's decision"""
    explanations = []

    # Basic explanation based on confidence
    if is_fake:
        if confidence > 90:
            explanations.append(
                "The image shows very strong indicators of manipulation."
            )
        elif confidence > 75:
            explanations.append(
                "The image contains several detectable signs of digital alteration."
            )
        elif confidence > 60:
            explanations.append(
                "The image shows some patterns consistent with digital manipulation."
            )
        else:
            explanations.append(
                "The image has subtle inconsistencies that suggest possible manipulation."
            )
    else:
        if confidence > 90:
            explanations.append(
                "The image shows characteristics highly consistent with authentic photographs."
            )
        elif confidence > 75:
            explanations.append(
                "The image contains natural patterns expected in genuine photographs."
            )
        elif confidence > 60:
            explanations.append(
                "The image appears to have properties consistent with real photographs."
            )
        else:
            explanations.append(
                "The image lacks strong indicators of manipulation, but confidence is moderate."
            )

    # Add metadata-based reasoning
    try:
        # Check for suspicious metadata patterns
        if "EXIF Data" in metadata:
            # Missing EXIF data in a high-quality image can be suspicious
            if len(metadata["EXIF Data"]) == 0:
                basic_info = metadata.get("Basic Information", {})
                dimensions = basic_info.get("Dimensions", "").split("x")[0]
                if dimensions and int(dimensions) > 1000:
                    explanations.append(
                        "High-resolution image lacks EXIF metadata, which is unusual for authentic camera photos."
                    )

            # Camera information consistency check
            camera_info = metadata.get("EXIF Data", {}).get("Camera", {})
            if camera_info and len(camera_info) >= 2:
                explanations.append(
                    f"Image contains consistent camera information which supports authenticity."
                )

            # Software processing indication
            for category, data in metadata.get("EXIF Data", {}).items():
                for tag, value in data.items():
                    if (
                        "Software" in tag
                        or "Photoshop" in str(value)
                        or "Adobe" in str(value)
                    ):
                        explanations.append(
                            f"Image contains tags indicating processing with {value}, which doesn't necessarily indicate manipulation but shows the image was edited."
                        )
    except:
        pass

    # Add technical analysis points
    if is_fake:
        explanations.append(
            "The neural network detected patterns consistent with GAN-generated or manipulated content."
        )
    else:
        explanations.append(
            "Image noise patterns and pixel relationships appear natural and consistent throughout the image."
        )

    # Return consolidated explanation
    return explanations


def generate_heatmap(image_path, model, target_layer_name="layer4"):
    """Generate Grad-CAM heatmap visualization showing regions that influenced the model's decision"""
    try:
        # Load the image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Resize image to model input size while preserving aspect ratio
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Get the target layer for GradCAM
        target_layer = None
        if hasattr(model, "layer4"):
            # For ResNet models
            target_layer = model.layer4[-1]
        else:
            # Fallback for other model architectures
            for name, module in model.named_modules():
                if target_layer_name in name:
                    target_layer = module
                    break

        if target_layer is None:
            print("Could not find target layer for GradCAM")
            return None

        # Create GradCAM object - fixed to remove use_cuda parameter
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Generate heatmap - remove target_category parameter
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]

        # Resize CAM to match original image dimensions
        grayscale_cam = resize(grayscale_cam, (img_array.shape[0], img_array.shape[1]))

        # Create visualization
        visualization = show_cam_on_image(
            img_array.astype(np.float32) / 255.0,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
        )

        # Save heatmap image
        heatmap_path = os.path.join(
            app.config["RESULT_FOLDER"], f"heatmap_{os.path.basename(image_path)}"
        )
        cv2.imwrite(heatmap_path, visualization)

        return heatmap_path
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None


# Process image and return result
def process_image(image_path):
    # Initialize model if not already done
    if not init_model():
        return {
            "error": "Model file not found. Please train or download the model first."
        }

    try:
        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_class = torch.max(outputs, 1)

        # Get prediction and confidence
        is_real = bool(predicted_class.item())
        confidence = probabilities[predicted_class.item()].item()

        # Save annotated image
        output_path = os.path.join(
            app.config["RESULT_FOLDER"], f"{os.path.basename(image_path)}"
        )

        # Add annotation to image
        draw = ImageDraw.Draw(image)
        result_text = f"{'REAL' if is_real else 'FAKE'}: {confidence:.2%}"
        text_color = (
            (0, 255, 0) if is_real else (255, 0, 0)
        )  # Green for real, red for fake

        try:
            # Try different system fonts
            font = None
            for font_name in [
                "arial.ttf",
                "Arial.ttf",
                "calibri.ttf",
                "Calibri.ttf",
                "segoeui.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_name, 20)
                    break
                except:
                    continue

            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Draw text with outline
        for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
            draw.text(
                (10 + offset[0], 10 + offset[1]), result_text, fill=(0, 0, 0), font=font
            )
        draw.text((10, 10), result_text, fill=text_color, font=font)

        # Save the annotated image
        image.save(output_path)

        # Post-processing features

        # 1. Extract metadata
        metadata = extract_metadata(image_path)

        # 2. Generate explanation for the decision
        explanation = generate_explanation(not is_real, confidence * 100, metadata)

        # 3. Generate heatmap visualization
        heatmap_path = generate_heatmap(image_path, model)
        heatmap_url = None
        if heatmap_path:
            heatmap_url = "/" + heatmap_path.replace("\\", "/").replace("//", "/")

        return {
            "is_real": is_real,
            "confidence": confidence * 100,  # Convert to percentage
            "result_path": output_path,
            "result_text": "REAL" if is_real else "FAKE",
            "metadata": metadata,
            "explanation": explanation,
            "heatmap_path": heatmap_path,
            "heatmap_url": heatmap_url,
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}


# Process video
def process_video(video_path, sample_rate=10):
    # Initialize model if not already done
    if not init_model():
        return {
            "error": "Model file not found. Please train or download the model first."
        }

    try:
        # Output video path
        output_filename = f"result_{os.path.basename(video_path)}"
        output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare output video
        if os.name == "nt":  # Windows
            fourcc = cv2.VideoWriter_fourcc(*"H264")  # or 'avc1' or 'mp4v'
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # If VideoWriter failed to initialize
        if not out.isOpened():
            print("Failed to open VideoWriter with first codec, trying fallback codec")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                print("Failed to open VideoWriter with fallback codec")
                return {"error": "Failed to initialize video encoder"}

        # Process frames
        results = []
        frame_count = 0

        print(f"Processing video with {total_frames} frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Only process every Nth frame
            if frame_count % sample_rate != 0:
                out.write(frame)  # Write original frame
                continue

            # Convert frame to PIL Image for model
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(pil_frame).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted_class = torch.max(outputs, 1)

            is_real = bool(predicted_class.item())
            confidence = probabilities[predicted_class.item()].item()

            # Add to results
            results.append(
                {"frame": frame_count, "is_real": is_real, "confidence": confidence}
            )

            # Annotate frame
            text = f"{'REAL' if is_real else 'FAKE'}: {confidence:.2%}"
            text_color = (
                (0, 255, 0) if is_real else (0, 0, 255)
            )  # BGR format: Green for real, Red for fake
            cv2.putText(
                frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4
            )  # Outline
            cv2.putText(
                frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
            )

            # Write annotated frame
            out.write(frame)

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

        # Release resources
        cap.release()
        out.release()

        # Calculate summary
        frames_analyzed = len(results)
        real_frames = sum(1 for r in results if r["is_real"])
        fake_frames = frames_analyzed - real_frames

        # Overall verdict based on majority
        is_real_video = fake_frames == 0

        # Confidence of verdict (how many frames agree)
        majority_pct = (
            max(real_frames, fake_frames) / frames_analyzed
            if frames_analyzed > 0
            else 0
        )

        return {
            "is_real": is_real_video,
            "confidence": majority_pct * 100,  # Convert to percentage
            "result_path": output_path,
            "result_text": "REAL" if is_real_video else "FAKE",
            "real_frames": real_frames,
            "fake_frames": fake_frames,
            "total_frames": frames_analyzed,
        }

    except Exception as e:
        print(f"Error processing video: {e}")
        return {"error": str(e)}


def cleanup_old_files(directory, max_age_days=1):
    """Remove files older than max_age_days from the directory"""
    now = time.time()
    max_age = max_age_days * 86400  # Convert days to seconds

    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age:
                    try:
                        os.remove(filepath)
                        print(f"Removed old file: {filepath}")
                    except Exception as e:
                        print(f"Error removing {filepath}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Routes
@app.route("/")
def index():
    # Clean up old files
    cleanup_old_files(app.config["UPLOAD_FOLDER"])
    cleanup_old_files(app.config["RESULT_FOLDER"])

    # Pass model loading status to the template
    has_model = os.path.exists(app.config["MODEL_PATH"])

    # Try to initialize the model if it exists but isn't loaded yet
    if has_model and not model_loaded:
        try:
            init_model()
        except Exception as e:
            print(f"Error initializing model on page load: {e}")

    return render_template("index.html", has_model=has_model)


@app.route("/upload", methods=["POST"])
def upload_file():
    # Check if model exists
    if not os.path.exists(app.config["MODEL_PATH"]):
        return jsonify(
            {"error": "Model file not found. Please train or download the model first."}
        )

    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    file_type = request.form.get("type", "image")

    if file and allowed_file(file.filename, file_type):
        # Create a unique filename
        filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process the file
        if file_type == "image":
            result = process_image(filepath)
        else:  # video
            sample_rate = int(request.form.get("sample_rate", 10))
            result = process_video(filepath, sample_rate)

        # Add paths for frontend
        if "result_path" in result:
            result["result_url"] = "/" + result["result_path"].replace(
                "\\", "/"
            ).replace("//", "/")

        return jsonify(result)

    return jsonify({"error": "File type not allowed"})


@app.route("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.route("/how-it-works.html")
def how_it_works():
    return send_from_directory("static", "how-it-works.html")


@app.route("/about.html")
def about():
    return send_from_directory("static", "about.html")


@app.route("/faq.html")
def faq():
    return send_from_directory("static", "faq.html")


@app.route("/contact.html")
def contact():
    return send_from_directory("static", "contact.html")


# Error handling
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    # Check for model file on startup and print warning if missing
    if not os.path.exists(app.config["MODEL_PATH"]):
        print(f"\nWARNING: Model file '{app.config['MODEL_PATH']}' not found.")
        print("You need to train the model first using deepfake_detector.py")
        print("or place a pre-trained model in the project directory.\n")

    init_model()  # Initialize model on startup
    app.run(debug=True)
