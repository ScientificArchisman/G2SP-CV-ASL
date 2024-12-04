import mediapipe as mp
import cv2
import numpy as np
import networkx as nx
import torch
from src.model import GCN
import yaml 
import gradio as gr
import sys 
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def translate_landmarks(landmarks, reference_point_index):
    """
    Translate the hand landmarks so that the reference point is at the origin.

    :param landmarks: NumPy array of shape (N, 3) representing hand landmarks.
    :param reference_point_index: Index of the landmark to use as the new origin.
    :return: Translated landmarks as a NumPy array.
    """
    # Get the reference point coordinates
    reference_point = landmarks[reference_point_index]

    # Translate all landmarks so that the reference point is at the origin
    translated_landmarks = landmarks - reference_point
    
    return translated_landmarks


def scale_landmarks(landmarks, desired_max_distance=1):
    """
    Scale the hand landmarks so that the maximum distance between any two landmarks
    is equal to desired_max_distance.

    :param landmarks: NumPy array of shape (N, 3) representing hand landmarks.
    :param desired_max_distance: The desired maximum distance between any two landmarks. Default value is 1.
    :return: Scaled landmarks as a NumPy array.
    """
    # Compute all pairwise distances
    distances = np.linalg.norm(landmarks[:, np.newaxis] - landmarks[np.newaxis, :], axis=2)

    # Find the maximum distance in the current set of landmarks
    current_max_distance = distances.max()

    # Calculate the scale factor
    scale_factor = desired_max_distance / current_max_distance

    # Scale the landmarks
    scaled_landmarks = landmarks * scale_factor
    
    return scaled_landmarks


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.6)

def extract_landmarks(image_path):
    """
    Extract hand landmarks from the given image.

    :param image_path: Path to the image file.
    :return: A NumPy array of landmarks.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Extract landmarks for the first hand detected
        landmarks = results.multi_hand_landmarks[0].landmark
        # Convert landmarks to a NumPy array
        landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        return landmarks_array
    return None


def preprocess_landmarks(landmarks, reference_point_index, desired_max_distance):
    """
    Preprocess the landmarks by translating and scaling.

    :param landmarks: NumPy array of landmarks.
    :param reference_point_index: Index of the landmark to use as the reference point for translation.
    :param desired_max_distance: The desired maximum distance between any two landmarks after scaling.
    :return: Preprocessed landmarks.
    """
    translated_landmarks = translate_landmarks(landmarks, reference_point_index)
    scaled_landmarks = scale_landmarks(translated_landmarks, desired_max_distance)
    return scaled_landmarks


# Define connections based on MediaPipe's hand model
connections = mp.solutions.hands.HAND_CONNECTIONS

def create_hand_graph(landmarks, connections=connections):
    """
    Create a graph from the extracted hand landmarks.

    :param landmarks: A NumPy array of landmarks.
    :return: A graph and its adjacency matrix.
    """
    G = nx.Graph()
    for idx, landmark in enumerate(landmarks):
        G.add_node(idx, pos=(landmark[0], landmark[1], landmark[2]))

    for connection in connections:
        G.add_edge(connection[0], connection[1])

    adjacency_matrix = nx.to_numpy_array(G)
    return G, adjacency_matrix


# MediaPipe hand landmark model indices for finger joints
def calculate_angle(point1, point2, point3):
    """
    Calculate the angle formed by three points.

    :param point1: The first point as a tuple (x, y, z).
    :param point2: The second point (the joint) as a tuple (x, y, z).
    :param point3: The third point as a tuple (x, y, z).
    :return: The angle in radians.
    """
    # Create vectors from the points
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)

    # Normalize the vectors
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)

    # Compute the dot product
    dot_product = np.dot(vector1_norm, vector2_norm)

    # Ensure the dot product is within the range [-1, 1] for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(dot_product)

    return angle_rad

finger_joints = {
        'thumb_cmc': (1, 2, 3),
        'thumb_mcp': (2, 3, 4),
        'index_finger_mcp': (5, 6, 7),
        'index_finger_pip': (6, 7, 8),
        'middle_finger_mcp': (9, 10, 11),
        'middle_finger_pip': (10, 11, 12),
        'ring_finger_mcp': (13, 14, 15),
        'ring_finger_pip': (14, 15, 16),
        'pinky_mcp': (17, 18, 19),
        'pinky_pip': (18, 19, 20)
    }

def calculate_all_finger_angles(landmarks, finger_joints = finger_joints):
    """
    Calculate angles for all finger joints using the landmarks.

    :param landmarks: A list of (x, y, z) tuples representing the hand landmarks.
    :return: A dictionary with joint names as keys and their angles in degrees as values.
    """

    angles = {}
    for joint, (p1, p2, p3) in finger_joints.items():
        angle = calculate_angle(landmarks[p1], landmarks[p2], landmarks[p3])
        angles[joint] = angle

    return angles


def process_angles(angles_dict:dict):
    """
    Process the angles to create a feature vector.

    :param angles: A dictionary of finger joint angles.
    :return: A NumPy array of the processed angles.
    """
    angles = np.zeros((21, 1))
    angle_joints_idx = (2, 3, 6, 7, 10, 11, 14, 15, 18, 19)
    for joint_idx, angle in zip(angle_joints_idx, angles_dict.values()):
        angles[joint_idx] = angle

    return angles


label_mapping = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "J": 9, 
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "O": 14,
            "P": 15,
            "Q": 16,
            "R": 17,
            "S": 18,
            "T": 19,
            "U": 20,
            "V": 21,
            "W": 22,
            "X": 23,
            "Y": 24,
            "Z": 25,
            "del": 26,
            "nothing": 27,
            "space": 28
        }

# Invert the label mapping
label_mapping = {v: k for k, v in label_mapping.items()}

def visualize_hand_graph(G):
    """
    Visualize the hand graph using matplotlib.

    :param G: A networkx graph of hand landmarks.
    :return: Matplotlib figure.
    """
    pos = nx.get_node_attributes(G, 'pos')

    # Since 'pos' contains (x, y, z), we need to convert it to 2D
    pos_2d = {node: (coords[0], -coords[1]) for node, coords in pos.items()}  # Flip y-axis for upright graph

    fig, ax = plt.subplots()
    nx.draw(G, pos=pos_2d, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, font_weight='bold', ax=ax)
    ax.set_title("Hand Landmarks Graph (Upright)")
    plt.close(fig)  # Close to prevent direct display

    return fig

def visualize_hand_on_image(image, landmarks, connections=connections):
    """
    Overlay hand graph structure on the hand image.

    :param image: PIL Image of the hand.
    :param landmarks: NumPy array of hand landmarks.
    :return: Image with graph overlay.
    """
    # Convert the image to OpenCV format for drawing
    image = np.array(image)
    if image.shape[-1] == 4:  # Handle RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    height, width, _ = image.shape

    # Draw connections (edges) between landmarks
    for connection in connections:
        start_idx, end_idx = connection
        start_point = tuple((landmarks[start_idx][:2] * [width, height]).astype(int))
        end_point = tuple((landmarks[end_idx][:2] * [width, height]).astype(int))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    # Draw nodes (landmarks)
    for idx, landmark in enumerate(landmarks):
        point = tuple((landmark[:2] * [width, height]).astype(int))
        cv2.circle(image, point, 5, (0, 0, 255), -1)
        # Optionally, draw the node indices
        cv2.putText(image, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Convert back to PIL Image for Gradio
    return Image.fromarray(image)

def pipeline(model, img_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Run the complete pipeline on the given landmarks.

    :param model: The trained model.
    :param img_path: The path to the image file.
    :param device: The device to run the model on. Default is 'cuda' if available else 'cpu'.
    :return: The predicted hand gesture, graph visualization, and features.
    """
    # Extract hand landmarks
    landmarks = extract_landmarks(img_path)
    if landmarks is None:
        return "No hand detected", None, None, None

    # Preprocess the landmarks
    preprocessed_landmarks = preprocess_landmarks(landmarks, reference_point_index=0, desired_max_distance=1)

    # Create a graph from the landmarks
    G, adjacency_matrix = create_hand_graph(preprocessed_landmarks)

    # Calculate angles for all finger joints
    finger_angles = calculate_all_finger_angles(preprocessed_landmarks)
    finger_angles = process_angles(finger_angles)

    # Combine features
    features = np.concatenate((preprocessed_landmarks, finger_angles), axis=1)

    # Convert the adjacency matrix to a PyTorch tensor
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)

    # Convert the features to a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Make a prediction using the model
    model.to(device)
    model.eval()
    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        adjacency_matrix_tensor = adjacency_matrix_tensor.to(device)

        output = model(features_tensor.unsqueeze(0), adjacency_matrix_tensor.unsqueeze(0))

    # Get the predicted class
    output = torch.argmax(output, dim=1)
    pred_class = label_mapping[output.item()]

    # Generate the graph visualization
    graph_fig = visualize_hand_graph(G)

    return pred_class, graph_fig, features, preprocessed_landmarks

def gradio_pipeline(image):
    """
    Run the pipeline and return the results for Gradio app.
    """
    # Save the input image to a temporary file
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    # Run the pipeline
    pred_class, graph_fig, features, preprocessed_landmarks = pipeline(model, temp_path)

    # Overlay graph on the hand image
    landmarks = extract_landmarks(temp_path)
    if landmarks is not None:
        overlaid_image = visualize_hand_on_image(image, landmarks)
    else:
        overlaid_image = image  # If no hand is detected, return the original image

    # Convert features to a DataFrame for display
    if features is not None:
        features_df = pd.DataFrame(features, columns=["x", "y", "z", "angles"])
    else:
        features_df = None

    # Format the predicted class with large font size
    formatted_pred_class = f"<div style='color:yellow; font-weight:bold; font-size:100px;'>{pred_class}</div>"

    return formatted_pred_class, graph_fig, features_df, overlaid_image

# Gradio app
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_file = "Models/20241130_211522.pth"
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Setup the model
    INPUT_DIM = config.get("model").get("input_dim", 4)
    HIDDEN_DIMS = config.get("model").get("hidden_dims", [64, 128, 32])
    NUM_CLASSES = config.get("model").get("num_classes", 32)
    model = GCN(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, num_classes=NUM_CLASSES, num_landmarks=21)
    model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))

    with gr.Blocks() as demo:
        gr.Markdown("## Hand Gesture Recognition with Graph Neural Network")

        # Pane for uploading the image and displaying the predicted class
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="pil")
                submit_button = gr.Button("Run Pipeline")
            with gr.Column(scale=1):
                class_output = gr.Markdown(value="", label="Predicted Class")

        # Pane for image with overlaid graph and separate graph visualization
        with gr.Row():
            with gr.Column(scale=1):
                uploaded_image = gr.Image(label="Hand with Graph Overlay")
            with gr.Column(scale=1):
                graph_output = gr.Plot(label="Hand Graph Structure")

        # Pane for features tensor
        with gr.Row():
            features_output = gr.Dataframe(label="Features Tensor (x, y, z, angles)", headers=["x", "y", "z", "angles"])

        # Action: Run the pipeline and update outputs
        submit_button.click(
            fn=gradio_pipeline,
            inputs=image_input,
            outputs=[class_output, graph_output, features_output, uploaded_image],
        )

    # Launch the app
    demo.launch()
