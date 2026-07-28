# 🌉 CV19 Bridge Deck Crack Classifier

GET 324 (AI & Machine Learning) Mini-Project — Laboratory Exercise 10
**Task:** Binary image classification — Concrete Bridge Deck Crack Detection (Cracked vs Non-Cracked)

## Live App
https://YOUR-STREAMLIT-URL-HERE.streamlit.app

> **Note:** This app is hosted on Streamlit Community Cloud's free tier, which puts apps to sleep after a period of inactivity. If the link shows a "Zzzz... this app has gone to sleep" screen, simply click **"Yes, get this app back up!"** and wait 30–60 seconds for it to restart.

## About
This application was developed to classify concrete bridge deck surface images as Cracked or Non-cracked, using a MobileNetV3 transfer-learning model trained on the SDNET2018 dataset from Kaggle. The training data showed a significant class imbalance, with 2,025 Cracked images against 11,595 Non-cracked images, which was addressed using class weighting during training. The model achieved an overall test accuracy of 87.4%, with a recall of 63% on the Cracked class, meaning some actual cracks were still missed — an area that could be improved further. Users can upload a surface photo and get an instant prediction with confidence scores. One major challenge was the slow speed of copying files to Google Drive while organizing the dataset; this was resolved by processing locally on Colab's disk first. Going forward, the model could be improved by collecting more Cracked images to reduce the imbalance, or fine-tuning deeper layers for better minority-class recall.

## How to Use
1. Open the live app link above
2. Upload a photo of a concrete bridge deck surface (JPG or PNG)
3. View the prediction and confidence scores for Cracked vs Non-Cracked

## Group Members (CV19)
| Name | Registration Number | GitHub Username |
|------|---------------------|------------------|
| Enweme Emmanuel Okon | 22/EG/CV/1439 | |
| Umana, Abasiekeme Godwin | 22/EG/CV/1409 | |
| Ndem, Timabasi Aniefiok | 22/EG/CV/1519 | |
| Ekong, Richard Martin | 22/EG/CV/1449 | |
| Jacob, Godstime Thursday | 22/EG/CV/1479 | |
| Asukwo, Martin Ukpa | 22/EG/CV/1419 | |
| Okon, Martin Victor | 22/EG/CV/1429 | |

## Tech Stack
- TensorFlow / Keras (MobileNetV3Small transfer learning)
- Streamlit (web interface)
- Google Colab (training environment)
- GitHub + Streamlit Community Cloud (version control & deployment)
