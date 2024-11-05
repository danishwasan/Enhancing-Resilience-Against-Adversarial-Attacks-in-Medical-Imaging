# Enhancing Resilience Against Adversarial Attacks in Medical Imaging

# Project Overview

This project introduces a robust defense mechanism for medical imaging systems against adversarial attacks, focusing on preserving the accuracy and security of Convolutional Neural Networks (CNNs) used in critical healthcare applications. Leveraging advanced feature transformation and transfer learning on ResNet152V2, the proposed approach mitigates the impact of adversarial examples, maintaining high performance even under attack. This project demonstrates the applicability of the method on Chest X-ray datasets, achieving a performance retention rate above 90% against imperceptible adversarial perturbations.

# Key Features

	•	Advanced Feature Transformation: Uses transfer learning with ResNet152V2, tailored specifically to enhance resilience in medical imaging.
	•	Efficient Adversarial Training: Implements adversarial training on transformed features rather than directly augmenting adversarial samples, resulting in a low computational cost and effective defense.
	•	Dimensionality Reduction with PCA: Principal Component Analysis (PCA) reduces the dimensionality of feature vectors, optimizing the adversarial training process and improving model robustness.
	•	Evaluation on Medical Imaging Data: Tested on Chest X-ray datasets (normal vs. pneumonia cases), demonstrating high resistance to adversarial attacks while maintaining diagnostic accuracy.

# Project Structure

	•	Fine-tuned ResNet152V2: Custom model fine-tuned on Chest X-ray images to differentiate between normal and pneumonia cases.
	•	Feature Transformation: Transforms input images into robust feature representations for training and defense.
	•	Adversarial Training: Efficient adversarial training using the transformed features, achieving resilience without high computational costs.
	•	Evaluation Metrics: Includes precision, recall, accuracy, and ROC curves to assess the model’s performance across various adversarial perturbations.

# Technical Requirements

	•	Python Libraries: TensorFlow, Scikit-learn, XGBoost, and PCA tools for model training, feature extraction, and dimensionality reduction.
	•	Adversarial Robustness Toolbox (ART): Library for implementing and testing adversarial attack methods (e.g., FGSM, PGD) to evaluate model robustness.

# How It Works

	1.	Data Preparation: Prepares and preprocesses Chest X-ray images, standardizing them for input into the CNN.
	2.	Model Fine-tuning: Adapts a ResNet152V2 model with medical imaging-specific parameters, fine-tuning it on X-ray data to optimize for accuracy.
	3.	Feature Transformation and PCA: Extracts high-dimensional features from images and applies PCA to reduce redundancy, creating a transformed feature dataset.
	4.	Adversarial Training: Trains an XGBoost classifier on the transformed feature dataset, enhancing robustness against adversarial attacks without excessive computational demands.
	5.	Evaluation: Assesses the model’s accuracy across original and adversarial datasets using key metrics and visualizations.

# Results and Findings

	•	High Resilience: The model retains over 90% performance accuracy on Chest X-ray data under adversarial conditions, showing robustness across various perturbations.
	•	Low Computational Cost: Achieves efficient training, with adversarial training reduced from hundreds of hours to a few minutes using feature transformation.
	•	Comprehensive Defense: Demonstrates effective defense against Fast Gradient Sign Method (FGSM) and other common adversarial attacks, with ROC and precision-recall curves highlighting resilience.

# Conclusion

The proposed defense mechanism significantly enhances the robustness of CNNs in medical imaging against adversarial attacks. By employing advanced feature transformation and efficient adversarial training, this project provides a reliable and computationally feasible solution for securing critical healthcare AI applications.

# Citation

Danish Vasan, Mohammad Hammoudeh,
Enhancing Resilience Against Adversarial Attacks in Medical Imaging Using Advanced Feature Transformation Training,
Current Opinion in Biomedical Engineering,
2024,
100561,
ISSN 2468-4511,
https://doi.org/10.1016/j.cobme.2024.100561.
(https://www.sciencedirect.com/science/article/pii/S2468451124000412)
Abstract: This study presents a machine learning-driven defense mechanism against adversarial attacks, specifically tailored for medical imaging applications. This mechanism utilizes feature transformation through transfer learning, leveraging a fine-tuned ResNet152V2 network trained on original medical images. To enhance the model’s robustness, we apply efficient adversarial training on transformed features extracted from both original and adversarial images. Additionally, we integrate Principal Component Analysis (PCA) to reduce feature dimensionality, optimizing the adversarial training process. When evaluated on Chest X-ray datasets, focusing on pneumonia and normal cases, the proposed mechanism demonstrated strong resilience against imperceptible attacks while maintaining a performance retention rate above 90%. These results show the potential of the proposed mechanism to enhance the reliability and security of CNN-based medical imaging systems in practical, real-world settings.
Keywords: Adversarial Attacks; Fast Gradient Sign Method (FGSM); Projected Gradient Descent (PGD); Medical CNN Fine-tuning; Advanced Feature Transformation; Principal Component Analysis; Medical Imaging System
