# E-Commerce Product Details Extraction using Tesseract-OCR

## Overview

This project addresses an e-commerce problem where product images are present, but crucial details such as the product name, description, and other relevant information are missing. To solve this, we use **Tesseract-OCR** to automatically extract text from product images, filling in the missing details. The project was developed as part of the **Amazon ML Challenge**, where the goal was to automate the extraction process, streamlining product cataloging.

## Problem Statement

Many e-commerce platforms face the issue where products are uploaded with only images, lacking essential textual details. This project provides a solution by:
- Extracting text directly from images using **Tesseract-OCR**.
- Ensuring that the text from images is extracted with the same format and structure as it appears visually.
- Helping automate product cataloging and improving data integrity for online product listings.

## Key Features

- **Image Text Extraction:** Automatically extracts text from images using Tesseract-OCR.
- **Structured Output:** Preserves the structure and format of text in the output as it appears in the image.
- **Handles Mixed Data:** Extracts both numeric and alphabetic characters, ensuring no data is left out.
- **Easy Integration:** Can be integrated into e-commerce systems to automate the extraction of product details.

## Technologies Used

- **Python 3.8+**
- **Tesseract-OCR**
- **Pillow (Python Imaging Library)**
- **Google Colab** (for cloud-based model testing and processing)




