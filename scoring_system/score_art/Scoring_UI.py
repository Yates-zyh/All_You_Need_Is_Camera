import os
import sys
import importlib.util
# Critical fix: Set environment variables before importing streamlit
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_ENV'] = 'production'
os.environ['STREAMLIT_DISABLE_WATCHER'] = 'true'

# Import necessary libraries
import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import joblib
import librosa
from sklearn.preprocessing import StandardScaler
import tempfile
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# Import the original prediction function from art_score_predict.py
def import_original_predict():
    try:
        spec = importlib.util.spec_from_file_location("art_score_predict", "art_score_predict.py")
        art_score_predict = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(art_score_predict)
        return art_score_predict.predict_dance_level
    except Exception as e:
        st.error(f"Error importing original predict function: {e}")
        return None

# Skeleton keypoint mapping
JOINT_MAPPING = {
    0: 0,   # Nose
    1: 1,   # Left Eye
    2: 2,   # Right Eye
    3: 3,   # Left Ear
    4: 4,   # Right Ear
    5: 5,   # Left Shoulder
    6: 6,   # Right Shoulder
    7: 7,   # Left Elbow
    8: 8,   # Right Elbow
    9: 9,   # Left Wrist
    10: 10, # Right Wrist
    11: 11, # Left Hip
    12: 12, # Right Hip
    13: 13, # Left Knee
    14: 14, # Right Knee
    15: 15, # Left Ankle
    16: 16, # Right Ankle
}

# Dance feature extractor class (simplified, just for compatibility)
class DanceFeatureExtractor:
    pass

# Apply custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E88E5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .level-beginner {
        color: #FF9999;
        font-weight: bold;
    }
    .level-intermediate {
        color: #66B2FF;
        font-weight: bold;
    }
    .level-expert {
        color: #99FF99;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Page configuration
def setup_page():
    try:
        st.set_page_config(
            page_title="Dance Performance Analyzer",
            page_icon="💃",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.write(f"Page configuration error: {e}")

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLO model with caching to avoid reloading"""
    try:
        # Set environment variable to indicate we're running in Streamlit
        if 'STREAMLIT_RUNNING' not in os.environ:
            os.environ['STREAMLIT_RUNNING'] = 'True'
            
        model_path = 'yolo11x-pose.pt'
        if os.path.exists(model_path):
            return YOLO(model_path)
        else:
            st.warning(f"Model file {model_path} not found. Will attempt to download.")
            try:
                # This will trigger the download
                return YOLO(model_path)
            except Exception as download_error:
                st.error(f"Error downloading model: {download_error}")
                return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Modified prediction function that uses the original implementation
def predict_dance_level(model_yolo, video_path):
    """Uses the original predict_dance_level function from art_score_predict.py"""
    try:
        # Add progress indicators for UI
        progress = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting analysis...")
        
        # Get the original predict function
        original_predict = import_original_predict()
        if original_predict is None:
            st.error("Failed to import original prediction function")
            return generate_demo_results()
        
        # Update progress
        progress.progress(25)
        status_text.text("Processing video...")
        
        # Call the original predict function
        results = original_predict(video_path)
        
        # Update progress
        progress.progress(75)
        status_text.text("Finalizing analysis...")
        
        if results is None:
            st.error("Failed to analyze video")
            return None
        
        # Convert results format if needed
        # The original function returns results in this format:
        # {
        #   'overall': {
        #       'level': level_map[overall_prediction],
        #       'probabilities': {
        #           'Beginner': overall_probabilities[0] * 100,
        #           'Intermediate': overall_probabilities[1] * 100,
        #           'Expert': overall_probabilities[2] * 100
        #       }
        #   },
        #   'segments': {
        #       'predictions': [level_map[p] for p in segment_predictions],
        #       'total_duration': duration
        #   }
        # }
        
        # We need to adapt the format for our UI which expects:
        # {
        #   'overall': {
        #       'level': level_map[overall_prediction],
        #       'prediction': overall_prediction,  # This is missing in original
        #       'probabilities': {
        #           'Beginner': overall_probabilities[0] * 100,
        #           'Intermediate': overall_probabilities[1] * 100,
        #           'Expert': overall_probabilities[2] * 100
        #       }
        #   },
        #   'segments': {
        #       'predictions': [(p[0], p[1]) for p in segment_predictions],  # This format is different
        #       'segment_times': segment_times,  # This is missing in original
        #       'total_duration': duration
        #   }
        # }
        
        # Let's adapt the segments format to work with our UI
        # First, convert the segment predictions to our format
        level_to_idx = {'Beginner': 0, 'Intermediate': 1, 'Expert': 2}
        segment_times = []
        segment_predictions_with_probs = []
        
        duration = results['segments']['total_duration']
        for i, pred in enumerate(results['segments']['predictions']):
            start_time = i * 15
            end_time = min((i + 1) * 15, duration)
            segment_times.append((start_time, end_time))
            
            # Create a dummy probability array for the predicted level
            # We don't have actual probabilities from the original function,
            # so we'll create a high probability for the predicted level
            probs = np.zeros(3)
            probs[level_to_idx[pred]] = 0.9  # Give 90% probability to predicted level
            
            # Distribute remaining 10% to other levels
            remaining = 0.1
            if level_to_idx[pred] == 0:  # Beginner
                probs[1] = remaining * 0.7  # Intermediate
                probs[2] = remaining * 0.3  # Expert
            elif level_to_idx[pred] == 1:  # Intermediate
                probs[0] = remaining * 0.5  # Beginner
                probs[2] = remaining * 0.5  # Expert
            else:  # Expert
                probs[0] = remaining * 0.3  # Beginner
                probs[1] = remaining * 0.7  # Intermediate
            
            segment_predictions_with_probs.append((level_to_idx[pred], probs))
        
        # Also need to add the 'prediction' field
        max_prob_level = max(results['overall']['probabilities'].items(), key=lambda x: x[1])[0]
        overall_prediction = level_to_idx[max_prob_level]
        
        # Create the adapted results
        adapted_results = {
            'overall': {
                'level': results['overall']['level'],
                'prediction': overall_prediction,
                'probabilities': results['overall']['probabilities']
            },
            'segments': {
                'predictions': segment_predictions_with_probs,
                'segment_times': segment_times,
                'total_duration': results['segments']['total_duration']
            }
        }
        
        # Clear progress indicators
        progress.progress(100)
        status_text.empty()
        
        st.success("Analysis complete! View results in the 'Detailed Results' tab.")
        
        return adapted_results
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Generate demo results for UI testing
def generate_demo_results():
    """Generate demo results for UI testing when models are not available"""
    # Simulated overall probabilities (Beginner, Intermediate, Expert)
    overall_probs = [0.25, 0.65, 0.10]  # Example: mostly intermediate
    
    # Get the index of the highest probability
    overall_prediction = np.argmax(overall_probs)
    
    # Create segment predictions (5 segments of 15 seconds each)
    segment_predictions = []
    segment_times = []
    
    # Create varied segment predictions
    # Simulation pattern: starts weaker, improves in the middle, varies at the end
    segment_levels = [0, 1, 1, 2, 1]  # 0=Beginner, 1=Intermediate, 2=Expert
    
    for i, level in enumerate(segment_levels):
        # Create segment time (each 15 seconds)
        start_time = i * 15
        end_time = (i + 1) * 15
        segment_times.append((start_time, end_time))
        
        # Create segment probabilities
        if level == 0:  # Beginner
            probs = np.array([0.70, 0.25, 0.05])
        elif level == 1:  # Intermediate
            probs = np.array([0.20, 0.70, 0.10])
        else:  # Expert
            probs = np.array([0.10, 0.30, 0.60])
            
        # Add some randomness
        probs += np.random.uniform(-0.1, 0.1, size=3)
        probs = np.clip(probs, 0.05, 0.95)  # Ensure probabilities aren't too extreme
        probs = probs / probs.sum()  # Normalize to sum to 1
        
        segment_predictions.append((level, probs))
    
    # Map levels to readable names
    level_map = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
    
    return {
        'overall': {
            'level': level_map[overall_prediction],
            'prediction': overall_prediction,
            'probabilities': {
                'Beginner': overall_probs[0] * 100,
                'Intermediate': overall_probs[1] * 100,
                'Expert': overall_probs[2] * 100
            }
        },
        'segments': {
            'predictions': segment_predictions,
            'segment_times': segment_times,
            'total_duration': segment_times[-1][1]  # Last segment end time
        }
    }

# Create gauge chart using Plotly
def create_gauge_chart(probabilities):
    # Extract probabilities
    beginner_prob = probabilities['Beginner']
    intermediate_prob = probabilities['Intermediate']
    expert_prob = probabilities['Expert']
    
    # Create gauge chart
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(go.Pie(
        values=[beginner_prob, intermediate_prob, expert_prob],
        labels=['Beginner', 'Intermediate', 'Expert'],
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(colors=['#FF9999', '#66B2FF', '#99FF99']),
        hoverinfo='label+percent',
        hole=0.3,
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        title={
            'text': "Performance Level Distribution",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        }
    )
    
    return fig

# Create timeline chart with Plotly
def create_timeline_chart(segment_predictions, segment_times):
    """Create timeline chart for segment analysis"""
    # Create level mapping
    level_map = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
    
    # Prepare timeline data
    data = []
    
    for i, ((pred, _), (start_time, end_time)) in enumerate(zip(segment_predictions, segment_times)):
        level = level_map[pred]
        
        # Color mapping based on level
        color_map = {
            'Beginner': '#FF9999',  # Light Red
            'Intermediate': '#66B2FF',  # Light Blue
            'Expert': '#99FF99'   # Light Green
        }
        
        # Format time as MM:SS
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        
        time_str = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
        
        # Add segment to data
        data.append({
            'Segment': f"Segment {i+1}<br>{time_str}",
            'Start': start_time,
            'Finish': end_time,
            'Level': level,
            'Color': color_map[level]
        })
    
    # Create figure with segment bars
    fig = go.Figure()
    
    for level in ['Beginner', 'Intermediate', 'Expert']:
        level_data = [d for d in data if d['Level'] == level]
        
        if level_data:
            color = level_data[0]['Color']
            
            fig.add_trace(go.Bar(
                x=[d['Finish'] - d['Start'] for d in level_data],
                y=[d['Segment'] for d in level_data],
                orientation='h',
                name=level,
                marker_color=color,
                base=[d['Start'] for d in level_data],
                text=[d['Level'] for d in level_data],
                textposition='inside',
                insidetextanchor='middle',
                hoverinfo='text',
                hovertext=[f"{d['Level']}<br>{d['Segment']}" for d in level_data]
            ))
    
    # Update layout
    fig.update_layout(
        title='Performance Level by Segment',
        xaxis_title='Time (seconds)',
        yaxis_title='Video Segments',
        barmode='stack',
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=300,
        yaxis={'categoryorder': 'array', 
               'categoryarray': [d['Segment'] for d in data]},
    )
    
    # Format x-axis as MM:SS
    fig.update_xaxes(
        tickformat="%M:%S",
        tickmode='array',
        tickvals=[i*15 for i in range(int(data[-1]['Finish']//15)+2)],
    )
    
    return fig

# Create stacked bar chart for segment breakdown
def create_segment_breakdown(segment_predictions, segment_times):
    # Prepare data
    segments = []
    beginner_probs = []
    intermediate_probs = []
    expert_probs = []
    times = []
    
    for i, ((_, probs), (start_time, end_time)) in enumerate(zip(segment_predictions, segment_times)):
        segment_label = f"Segment {i+1}"
        segments.append(segment_label)
        beginner_probs.append(probs[0] * 100)
        intermediate_probs.append(probs[1] * 100)
        expert_probs.append(probs[2] * 100)
        
        # Format time as MM:SS - MM:SS
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        time_str = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
        times.append(time_str)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each level
    fig.add_trace(go.Bar(
        y=segments,
        x=beginner_probs,
        name='Beginner',
        orientation='h',
        marker=dict(color='#FF9999'),
        text=[f"{p:.1f}%" for p in beginner_probs],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"Beginner: {p:.1f}%<br>Time: {t}" for p, t in zip(beginner_probs, times)]
    ))
    
    fig.add_trace(go.Bar(
        y=segments,
        x=intermediate_probs,
        name='Intermediate',
        orientation='h',
        marker=dict(color='#66B2FF'),
        text=[f"{p:.1f}%" for p in intermediate_probs],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"Intermediate: {p:.1f}%<br>Time: {t}" for p, t in zip(intermediate_probs, times)]
    ))
    
    fig.add_trace(go.Bar(
        y=segments,
        x=expert_probs,
        name='Expert',
        orientation='h',
        marker=dict(color='#99FF99'),
        text=[f"{p:.1f}%" for p in expert_probs],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f"Expert: {p:.1f}%<br>Time: {t}" for p, t in zip(expert_probs, times)]
    ))
    
    # Update layout
    fig.update_layout(
        title="Segment Breakdown by Dance Level",
        barmode='stack',
        xaxis=dict(
            title="Probability (%)",
            range=[0, 100]
        ),
        yaxis=dict(
            title="Segments",
            autorange="reversed"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        height=350
    )
    
    # Add custom data to y-axis ticks
    fig.update_yaxes(
        ticktext=[f"{s}<br><sub>{t}</sub>" for s, t in zip(segments, times)],
        tickvals=segments
    )
    
    return fig

# Get performance insights based on level
def get_performance_insights(level):
    insights = ""
    
    if level == 'Beginner':
        insights = """
        ### Beginner Level Performance
        
        Your performance shows basic dance moves with limited coordination between music and movements. Here are some points to consider:
        
        - **Strengths**: Shows basic understanding of dance movements
        - **Areas for Improvement**: Timing, rhythm awareness, and body control
        
        ### Suggestions for Improvement:
        - Practice with slower songs to improve timing
        - Work on body isolation exercises
        - Focus on learning basic dance patterns
        - Practice maintaining rhythm throughout routines
        - Consider taking beginner dance classes to build foundational skills
        """
    elif level == 'Intermediate':
        insights = """
        ### Intermediate Level Performance
        
        Your performance demonstrates good coordination and rhythm awareness. There's clear understanding of basic techniques with some advanced elements.
        
        - **Strengths**: Good rhythm awareness, decent technical execution
        - **Areas for Improvement**: Consistency, expressiveness, and complex patterns
        
        ### Suggestions for Improvement:
        - Work on more complex dance patterns
        - Improve expressiveness and energy throughout performance
        - Focus on transitions between movements
        - Refine technical precision
        - Practice with varied music tempos to improve adaptability
        """
    else:  # Expert
        insights = """
        ### Expert Level Performance
        
        Your performance showcases excellent dance skills with exceptional coordination, rhythm, and expression. Movement quality is high with consistent energy throughout.
        
        - **Strengths**: Advanced technical skill, excellent rhythm, strong expressiveness
        - **Notable Features**: Consistent performance quality, complex movement patterns
        
        ### Suggestions for Further Development:
        - Continue refining personal style and artistic expression
        - Explore more challenging choreographies
        - Mentor others to strengthen your own abilities
        - Focus on subtle nuances in movement quality
        - Consider specialized training in advanced techniques
        """
    
    return insights

# Create video thumbnail
def create_video_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Read a frame from the middle of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create PIL Image
    img = Image.fromarray(frame_rgb)
    
    # Resize to reasonable dimensions
    max_width = 600
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    
    return img

def main():
    # Set up page
    setup_page()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Add title
    st.markdown("<h1 class='main-header'>Dance Performance Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("Analyze your dance performances and get professional feedback on your skill level.")
    
    # Check if the required model directories exist, if not create them
    model_dir = Path('./Trained_models')
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        st.warning("Created 'Trained_models' directory. Please make sure your models are placed here.")
    
    # Check if running with streamlit run
    if not os.environ.get('STREAMLIT_RUNNING', False):
        how_to_run = f"streamlit run {os.path.basename(__file__)}"
        st.warning(f"⚠️ For the best experience, run this app with: `{how_to_run}`")
    
    # Load YOLO model
    with st.spinner("Loading pose detection model..."):
        model = load_model()
        if model:
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model. Please check that 'yolo11x-pose.pt' is available.")
            
            # Option to download the model
            if st.button("Download YOLO model"):
                with st.spinner("Downloading YOLOv11-pose model..."):
                    try:
                        # This will trigger the download through ultralytics
                        YOLO('yolo11x-pose.pt')
                        st.success("Download complete! Please refresh the page.")
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("About the Tool")
        st.markdown("""
        This tool uses computer vision and machine learning to analyze dance performances from video files.
        It extracts movement features and audio characteristics to predict the dancer's skill level.
        
        - 💃 **Pose Detection**: Uses YOLO model to track body movements
        - 🎵 **Audio Analysis**: Analyzes rhythm and beats
        - 🧠 **ML Classification**: Classifies into Beginner, Intermediate, or Expert
        
        Upload your dance video to begin the analysis!
        """)
        
        st.divider()
        
        # Show system specs
        st.subheader("System Specifications")
        st.info(f"PyTorch Version: {torch.__version__}")
        st.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Detailed Results", "Video Preview"])
    
    # Upload tab
    with tab1:
        st.subheader("Upload Your Dance Video")
        
        # Simplified file uploader
        uploaded_file = st.file_uploader(
            "Select video file",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        # Analyze button with visual feedback
        if uploaded_file:
            # Display basic file info
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Save uploaded file to temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()
            
            # Get video thumbnail and basic info
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Get video thumbnail
                thumbnail = create_video_thumbnail(video_path)
                if thumbnail:
                    st.image(thumbnail, caption="Video Preview", use_column_width=True)
            
            with col2:
                # Display basic info about video
                st.subheader("Video Information")
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                    
                    # Display video info in a compact format
                    st.info(f"**Duration:** {int(duration//60):02d}:{int(duration%60):02d}")
                    st.info(f"**Resolution:** {width}x{height}")
                    st.info(f"**FPS:** {fps:.2f}")
                    st.info(f"**Size:** {file_size:.2f} MB")
                    
                    cap.release()
            
            # Simplified analyze button
            analyze_button = st.button("Analyze Performance", type="primary", use_container_width=True)
            
            if analyze_button:
                # Run analysis
                with st.spinner("Analyzing dance performance..."):
                    results = predict_dance_level(model, video_path)
                
                if results:
                    # Store results in session state for other tabs
                    st.session_state['analysis_results'] = results
                    st.session_state['video_path'] = video_path
                    
                    # Get highest probability level and format for display
                    overall_probs = results['overall']['probabilities']
                    max_prob_level, max_prob_value = max(overall_probs.items(), key=lambda x: x[1])
                    
                    # Display overall evaluation result in a format closer to the command line output
                    st.markdown("## Overall Evaluation:")
                    st.markdown(f"**Final Rating:** {max_prob_level} ({max_prob_value:.2f}%)")
                    
                    # Display probabilities for each level
                    st.markdown("### Probabilities for each level:")
                    for level, prob in overall_probs.items():
                        st.markdown(f"**{level}:** {prob:.2f}%")
                    
                    # Use columns for metric display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Beginner", f"{overall_probs['Beginner']:.1f}%")
                    with col2:
                        st.metric("Intermediate", f"{overall_probs['Intermediate']:.1f}%")
                    with col3:
                        st.metric("Expert", f"{overall_probs['Expert']:.1f}%")
                    
                    # Display segment-wise predictions
                    st.markdown("## Segment-wise predictions:")
                    for i, ((pred, _), (start_time, end_time)) in enumerate(zip(results['segments']['predictions'], results['segments']['segment_times'])):
                        level_map = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
                        level = level_map[pred]
                        
                        # Format time as MM:SS - MM:SS
                        start_min = int(start_time // 60)
                        start_sec = int(start_time % 60)
                        end_min = int(end_time // 60)
                        end_sec = int(end_time % 60)
                        time_str = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                        
                        st.markdown(f"**{time_str}:** {level}")
                    
                    # Prompt user to view detailed results
                    st.markdown("**Analysis complete!** Check the 'Detailed Results' tab for more information.")
                    
                else:
                    st.error("Analysis failed. Please try with a different video.")
    
    # Detailed results tab
    with tab2:
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            # Display overall results
            st.markdown("<h2 class='sub-header'>Overall Performance Analysis</h2>", unsafe_allow_html=True)
            
            # Gauge chart and insights side by side
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Gauge chart
                gauge_fig = create_gauge_chart(results['overall']['probabilities'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Performance insights based on level
                level = results['overall']['level']
                insights = get_performance_insights(level)
                st.markdown(insights)
            
            # Divider
            st.divider()
            
            # Timeline analysis
            st.markdown("<h2 class='sub-header'>Timeline Analysis</h2>", unsafe_allow_html=True)
            st.markdown("This chart shows how your performance level varies throughout the video.")
            
            # Create and display timeline chart
            segment_predictions = [(pred, probs) for pred, probs in results['segments']['predictions']]
            segment_times = results['segments']['segment_times']
            
            # Ensure there's data for the timeline
            if segment_predictions and segment_times:
                timeline_fig = create_timeline_chart(segment_predictions, segment_times)
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.warning("Not enough segment data to create timeline visualization.")
            
            # Segment breakdown
            st.markdown("<h2 class='sub-header'>Segment Breakdown</h2>", unsafe_allow_html=True)
            st.markdown("Detailed analysis of each segment of your performance.")
            
            # Create and display segment breakdown
            breakdown_fig = create_segment_breakdown(segment_predictions, segment_times)
            st.plotly_chart(breakdown_fig, use_container_width=True)
            
            # Detailed segment data
            st.markdown("<h2 class='sub-header'>Segment Details</h2>", unsafe_allow_html=True)
            
            # Prepare table data
            segment_data = []
            for i, ((pred, probs), (start_time, end_time)) in enumerate(zip(segment_predictions, segment_times)):
                # Format time as MM:SS - MM:SS
                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                end_min = int(end_time // 60)
                end_sec = int(end_time % 60)
                time_str = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                
                # Level mapping
                level_map = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
                
                # Add to segment data
                segment_data.append({
                    "Segment": f"Segment {i+1}",
                    "Time": time_str,
                    "Level": level_map[pred],
                    "Beginner %": f"{probs[0]*100:.1f}%",
                    "Intermediate %": f"{probs[1]*100:.1f}%",
                    "Expert %": f"{probs[2]*100:.1f}%"
                })
            
            # Display as dataframe
            if segment_data:
                df = pd.DataFrame(segment_data)
                st.dataframe(df, use_container_width=True)
                
                # Export options
                if st.button("Export Results as CSV"):
                    # Convert dataframe to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="dance_analysis_results.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Please upload and analyze a video on the 'Upload & Analyze' tab first.")
    
    # Video preview tab
    with tab3:
        if 'video_path' in st.session_state:
            video_path = st.session_state['video_path']
            
            st.markdown("<h2 class='sub-header'>Video Preview</h2>", unsafe_allow_html=True)
            
            # Display video
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            # Video metadata
            st.markdown("<h3>Video Details</h3>", unsafe_allow_html=True)
            
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                
                # Display details in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{int(duration//60):02d}:{int(duration%60):02d}")
                with col2:
                    st.metric("Resolution", f"{width}x{height}")
                with col3:
                    st.metric("Frame Rate", f"{fps:.2f} FPS")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Frames", f"{frame_count}")
                with col2:
                    st.metric("File Size", f"{file_size:.2f} MB")
                
                cap.release()
        else:
            st.info("Please upload a video on the 'Upload & Analyze' tab first.")

# Use direct launch method to avoid thread-related errors
if __name__ == "__main__":
    # Set Streamlit environment variables to help avoid warnings
    os.environ['STREAMLIT_RUNNING'] = 'True'
    os.environ['STREAMLIT_DISABLE_WATCHER'] = 'true'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_ENV'] = 'production'
    
    # Add command line argument detection for better error messages
    if len(sys.argv) == 1:  # No arguments provided
        print("\n⚠️  For the best experience, run this app with: streamlit run", sys.argv[0])
        print("    Running in compatibility mode...\n")
    
    # Launch the app
    main()