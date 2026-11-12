# Auralis — Multimodal Wearable Physiological Monitoring System

Auralis is a research-oriented wearable technology project exploring the integration of physiological sensors, embedded systems, and edge artificial intelligence to identify patterns associated with physiological stress and anxiety.

The project aims to develop an affordable, modular platform that combines multiple physiological signals with contextual information to support research into physiological arousal and stress-related patterns.

> **Research disclaimer:** Auralis is a prototype under development. It is not a clinically validated diagnostic device and must not be used to diagnose panic attacks, anxiety disorders, or other medical conditions.

## Project Objectives

- Develop a wearable platform for multimodal physiological data acquisition.
- Integrate physiological sensors with an ESP32-based embedded system.
- Explore preprocessing and feature extraction for physiological signals.
- Investigate machine-learning methods for classifying stress-related patterns.
- Explore edge inference and efficient data processing.
- Develop supporting software for displaying and analyzing acquired signals.
- Evaluate sensor performance and system reliability through appropriate testing.

## System Components

The following are planned or candidate components and may change during development:

- **PPG:** MAX30102 planned for optical pulse sensing and heart-rate estimation (the bench prototype currently uses a MAX30100 board).
- **Electrodermal activity:** A suitable EDA/GSR sensing circuit and electrode configuration.
- **Temperature:** A suitable temperature sensor for the intended measurement location.
- **Motion:** MPU6050-class accelerometer and gyroscope for activity context (the bench GY-521 board carries an MPU6500).
- **Embedded platform:** ESP32 for sensor interfacing and data acquisition.
- **Machine learning:** Python-based preprocessing, feature extraction, and model development.
- **Software interface:** A dashboard or application for displaying and analyzing data.

## Machine-Learning Approach

Initial development will investigate public physiological datasets, including WESAD, using the labels actually provided by the dataset.

Potential approaches include:
- Feature-based classification using models such as Random Forest and XGBoost.
- Signal preprocessing and feature extraction.
- Activity-aware interpretation of physiological changes.
- Exploration of lightweight models suitable for embedded deployment.

WESAD provides stress-related experimental labels rather than clinical panic-attack diagnoses. Any clinical anxiety-related claims would require appropriately labelled data, ethics approval, and independent validation.

## Repository Structure

- `hardware/` — CAD, PCB, sensor integration, FEA, and bill of materials.
- `firmware/` — ESP32 firmware and sensor interfacing.
- `software/` — Data processing, machine learning, backend, dashboard, and application code.
- `research/` — Literature review, methodology, references, and approved research materials.
- `testing/` — Hardware, software, and validation tests.
- `documentation/` — Architecture, setup instructions, and progress reports.
- `competitions/` — Competition-specific deliverables and supporting evidence.
- `assets/` — Project diagrams and non-confidential images.

## Development Status

Auralis is under active development. Hardware integration, detailed mechanical design, PCB development, machine-learning workflows, and study documentation may be at different stages of completion.

Only verified implementation results and completed tests will be reported as finished. Planned features and preliminary analyses will be clearly identified as such.

## Research Ethics and Data Protection

Human-participant studies will only proceed after obtaining the necessary institutional ethical approval and following the approved protocol.

Personal or identifiable participant data, confidential clinical information, credentials, and private institutional documents must not be stored in this public-facing project documentation.

## Project Applications

This repository supports the development lifecycle of Auralis, including engineering development, research documentation, prototype testing, and relevant innovation or competition submissions.

## Contributions

Team members should use meaningful commit messages, document major design changes, and preserve relevant native engineering files and test evidence.

## Disclaimer

This project is intended for research and engineering development. Its outputs are experimental and are not a substitute for professional medical assessment or diagnosis.
