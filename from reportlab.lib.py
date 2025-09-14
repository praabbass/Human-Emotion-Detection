from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Create PDF document
pdf_path = "Audio_Emotion_Recognition_Report.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)

# Styles
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("Audio Emotion Recognition Project Report", styles['Title']))
story.append(Spacer(1, 12))

# Introduction
intro_text = """
This report covers the development of an Audio Emotion Recognition system. The system is designed to analyze audio inputs, 
either uploaded files or live recordings, to detect human emotions such as 'neutral', 'happy', 'angry', etc.
"""
story.append(Paragraph(intro_text, styles['BodyText']))
story.append(Spacer(1, 12))

# Dataset and Model
dataset_text = """
The model was trained using a labeled audio dataset where different emotions were represented by .wav files.
Feature extraction was done using Librosa, extracting MFCC, chroma, mel-spectrogram, contrast, and tonnetz features.
A Random Forest classifier was trained on these features achieving satisfactory accuracy.
"""
story.append(Paragraph(dataset_text, styles['BodyText']))
story.append(Spacer(1, 12))

# Streamlit App Screenshot
story.append(Paragraph("Below is a screenshot of the Streamlit application interface:", styles['BodyText']))
story.append(Spacer(1, 12))
image_path = "Screenshot (57).png"  # Make sure this image is in the same folder as the script
img = Image(image_path)
img.drawHeight = 4 * inch
img.drawWidth = 6 * inch
story.append(img)
story.append(Spacer(1, 12))

# Conclusion
conclusion_text = """
This project demonstrates an efficient approach for audio-based emotion detection with real-time usability.
The model can analyze various audio sources and accurately predict emotions.
Future improvements may include expanding the dataset and enhancing feature extraction techniques.
"""
story.append(Paragraph(conclusion_text, styles['BodyText']))
story.append(Spacer(1, 12))

# Build PDF
doc.build(story)

print(f"PDF report generated at {pdf_path}")
