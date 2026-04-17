🏥 AI-Based MRI-LINAC Patient Positioning and Image Registration

🔍 Overview

This project explores the use of artificial intelligence techniques for improving patient positioning and image registration in MRI-guided radiotherapy (MRI-LINAC systems). Accurate alignment between imaging and treatment delivery is critical for ensuring precise dose delivery to tumours while minimising exposure to surrounding healthy tissue.

🏥 Clinical Motivation

MRI-LINAC systems integrate high-resolution imaging with radiation delivery, enabling adaptive radiotherapy. However, patient positioning errors and image misalignment can lead to:

Reduced treatment accuracy
Increased radiation exposure to healthy tissues
Variability in treatment outcomes

Automated and AI-assisted positioning and registration methods can enhance precision, reduce setup time, and improve overall treatment efficiency.

⚙️ Methodology

Image Preprocessing
MRI data normalisation
Noise reduction and filtering
Standardisation of image orientation
Image Registration
Rigid and/or deformable registration techniques
Alignment of planning and treatment images
Evaluation of spatial transformation accuracy
AI-Based Approach
Feature-based or learning-based alignment methods
Exploration of similarity metrics for optimisation
Potential use of machine learning for improving registration performance

🔄 Pipeline

MRI Acquisition → Preprocessing → Feature Extraction → Image Registration → Alignment Evaluation → Position Correction

📊 Results

Demonstrated feasibility of automated alignment between MRI datasets
Improved consistency in positioning across image sets
Quantitative evaluation using similarity metrics (e.g. mutual information, MSE)

⚠️ Limitations

Limited dataset availability
Simplified registration framework compared to clinical systems
Lack of integration with real-time radiotherapy workflows

🔐 Clinical and Safety Considerations

Accurate registration is critical for patient safety in radiotherapy
Errors in positioning may lead to incorrect dose delivery
AI models must undergo rigorous validation before clinical implementation

🔮 Future Work

Integration with real-time MRI-LINAC workflows
Use of deep learning for deformable image registration
Validation using clinical radiotherapy datasets
Incorporation of motion management strategies

🧾 Technical Stack

Python
NumPy, OpenCV
SimpleITK
Scikit-learn / Deep learning frameworks 
