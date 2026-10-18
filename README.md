# Early Identification of At-Risk Students in Distance Education (OULAD)

Binary classification of whether a student will **Fail/Withdraw** (at-risk) or **Pass/Distinction**, using only
information available up to day 60 of the course (early-warning setting). Compares traditional ML (scikit-learn)
with deep learning (TensorFlow/Keras, `tf.data`).

## Dataset
Open University Learning Analytics Dataset (OULAD), Kuzilek et al., *Scientific Data* 4:170171 (2017), CC-BY 4.0.
Download: https://analyse.kmi.open.ac.uk/open-dataset (also listed on the UCI repository).
Raw data is **not** stored in this repo. Place `assessments.csv`, `studentAssessment.csv`, `studentInfo.csv`,
`studentRegistration.csv`, `studentVle.csv` in `./data` (or set `OULAD_DATA_DIR`).

## Run
```
pip install -r requirements.txt
jupyter notebook OULAD_Early_Warning_Final_Project.ipynb
```
In Google Colab, upload the CSVs to `MyDrive/OULAD_data` and run all cells. Set `FAST_MODE = True` for a quick test.

## Structure of the notebook
Setup/seeds -> data loading -> EDA -> feature engineering -> grouped split -> preprocessing -> 7 traditional ML
experiments -> ablation and cutoff experiments -> 6 deep learning experiments -> results table -> learning curves,
confusion matrices, ROC/PR -> error analysis -> model saving.

## Links
- Google Drive (data zip + saved models): `[PASTE LINK]`
- Video: `[PASTE LINK]`
