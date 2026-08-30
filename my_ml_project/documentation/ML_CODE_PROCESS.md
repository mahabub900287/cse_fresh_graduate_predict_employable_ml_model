# 📜 কোড অ্যানালাইসিস ও ভাইভা ব্যাখ্যা গাইড (Code Breakdown)

এই নথিতে প্রজেক্টের বর্তমান (আজকের আপডেট করা) পাইথন কোডের প্রতিটি অংশের কাজ, ব্যবহৃত প্রযুক্তি এবং ভাইভাতে উত্তর দেওয়ার কৌশল বাংলা ভাষায় সহজভাবে উপস্থাপন করা হয়েছে। কোড এখন চারটি ফাইলে ভাগ করা: `train_model.py` (মূল ট্রেইনিং পাইপলাইন), `compare_algorithms.py` (৪টি অ্যালগরিদম তুলনা), `main.py` (FastAPI সার্ভিং), এবং `webview.html` (লোকাল ওয়েব ইন্টারফেস)।

---

## 📌 অংশ ১: train_model.py — Setup & Data Loading (প্রস্তুতি ও ডাটা লোড)

* **কাজের বিবরণ:** প্রয়োজনীয় লাইব্রেরি ইমপোর্ট করা এবং raw CSV ডাটাসেট লোড করা। এখন আর Google Colab/Drive ব্যবহার হয় না — পুরো পাইপলাইনটি লোকাল মেশিনে (venv-এর মধ্যে) চলে।
* **কী কী ব্যবহার হয়েছে:**
  * `pandas` & `numpy`: ডাটা প্রসেসিং ও গাণিতিক হিসেব।
  * `joblib`: ট্রেইন করা মডেল সেভ (`.pkl`) এবং লোড করা।
  * `RANDOM_STATE = 42`: প্রতিবার রান করলে যেন ফলাফল হুবহু reproducible থাকে (split, RandomizedSearchCV, XGBoost — সবখানে একই seed)।
  * `sys.argv[1]`: dataset-এর path কমান্ড-লাইন আর্গুমেন্ট হিসেবে দেওয়া যায় (default: `student_career_success_dataset.csv`)।

---

## 📌 অংশ ২: Data Cleaning & Label Normalisation (ডাটা ক্লিন ও টার্গেট ঠিক করা)

* **কাজের বিবরণ:** raw Kaগগল ডাটাসেট ক্লিন করে থিসিসে বর্ণিত ফরম্যাটে আনা।
* **কী কী ব্যবহার হয়েছে:**
  * `df.drop_duplicates()`: ডুপ্লিকেট রেকর্ড মুছে ফেলা।
  * `str.strip()`: টেক্সট কলামের আগে-পরে থাকা স্পেস মুছে ফেলা।
  * **Target normalisation:** raw ডাটাসেটে টার্গেট কলামের মান থাকে `"Placed"` / `"Not Placed"` — কোড স্বয়ংক্রিয়ভাবে এগুলোকে `"Employable"` / `"Not Employable"`-এ ম্যাপ করে দেয়, যাতে থিসিসের বর্ণনার সাথে মিলে।
  * **Leakage কলাম বাদ দেওয়া:** `Student_ID`, `Employability_Score`, `Company_Tier`, `Career_Field`, `Placement_Mode`, `Starting_Salary_USD` — এই কলামগুলো প্লেসমেন্ট আউটকামের *পরে* তৈরি হয় (post-outcome leakage), তাই ট্রেইনিং থেকে বাদ।

---

## 📌 অংশ ৩: Feature Engineering (নতুন ফিচার তৈরি)

* **কাজের বিবরণ:** RAW ডাটা থেকে মডেলের বুদ্ধিমত্তা বাড়ানোর জন্য ২টি নতুন কাস্টম ফিচার বানানো (অপরিবর্তিত)।
* **কী কী ব্যবহার হয়েছে:**
  * `Overall_Preparedness_Index = Interview_Score × 0.7 + Internships × 0.3`
  * `Skills_per_Project = Programming_Skill / (Projects_Completed + 1)`

---

## 📌 অংশ ৪: Feature Exclusion & Split X/y (বাদ দেওয়া ও ইনপুট-আউটপুট আলাদা করা)

* **কাজের বিবরণ:** ৬টি demographic/contextual অ্যাট্রিবিউট বাদ দিয়ে বাকি ১৮টি ফিচার দিয়ে মডেল ট্রেইন করা।
* **কী কী ব্যবহার হয়েছে:**
  * **বাদ দেওয়া কলাম (`EXCLUDED_COLUMNS`):** `Age`, `Gender`, `University_Year`, `Major`, `Attendance_Percentage`, `LinkedIn_Profile` — ছাত্রের নিয়ন্ত্রণের বাইরে থাকা ডেমোগ্রাফিক তথ্য, bias এড়ানোর জন্য বাদ।
  * `LabelEncoder()`: টার্গেট (`Employable`/`Not Employable`) সংখ্যায় রূপান্তর করা — alphabetical ক্রমে, তাই `Employable = 0`, `Not Employable = 1`।
  * `select_dtypes`: নিউমেরিক্যাল (১৪টি) ও ক্যাটাগরিক্যাল (৪টি) কলাম আলাদা করা।

---

## 📌 অংশ ৫: Train-Test Split (ডাটা ভাগ করা)

* **কাজের বিবরণ:** ৫০,০০০ রেকর্ডকে ট্রেনিং ও টেস্টিং সেটে ভাগ করা।
* **কী কী ব্যবহার হয়েছে:**
  * `train_test_split(test_size=0.2, random_state=42, stratify=y)`: ৮০% (৪০,০০০) ট্রেইনিং, ২০% (১০,০০০) টেস্ট।
  * `stratify=y`: উভয় সেটে Employable/Not Employable-এর অনুপাত (৭৮:২২) অবিকৃত রাখা।

---

## 📌 অংশ ৬: Preprocessing Pipeline (অটোমেটিক ডাটা প্রসেসিং)

* **কাজের বিবরণ:** সংখ্যা ও টেক্সট ডাটাকে মডেলের উপযোগী করা — পাইপলাইনের ভেতরেই fit করা হয়, যাতে data leakage না হয়।
* **কী কী ব্যবহার হয়েছে:**
  * `SimpleImputer(strategy="median")`: নিউমেরিক কলামের ফাঁকা মান পূরণ (median — outlier-এর প্রভাব কম)।
  * `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`: ক্যাটাগরিক্যাল কলামের ফাঁকা মান পূরণ ও বাইনারি এনকোডিং।
  * `StandardScaler()`: নিউমেরিক ফিচার স্ট্যান্ডার্ডাইজ করা।
  * `ColumnTransformer`: সংখ্যা ও টেক্সট প্রসেসিং একসাথে যুক্ত করা।

---

## 📌 অংশ ৭: Model Training — Tuned XGBoost, **No SMOTE** ⚠️ (পরিবর্তিত অংশ)

* **কাজের বিবরণ:** আগের ভার্সনে SMOTE + `scale_pos_weight` + কাস্টম থ্রেশহোল্ড (০.৩৫) ব্যবহার হতো। **এখন এই তিনটি সম্পূর্ণ বাদ** — কারণ থিসিসের Section 5.6/5.7-এ দেখানো হয়েছে, এই তিনটি একসাথে ব্যবহার করলে overall accuracy ৭৮.০৮% majority-class baseline-এর নিচে (৬৩.৮৪%) নেমে যায়।
* **কী কী ব্যবহার হয়েছে:**
  * `XGBClassifier`: কোনো SMOTE বা class-weight ছাড়া সরাসরি ট্রেইন।
  * `RandomizedSearchCV(n_iter=15, cv=3, scoring="accuracy")`: হাইপারপ্যারামিটার (n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_child_weight) খোঁজার জন্য cross-validated সার্চ।
  * **সেরা হাইপারপ্যারামিটার (এই ডেটাসেটে):** `n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=1`।
  * Full training সময় (৪০,০০০ রেকর্ডে): প্রায় ০.৭৭ সেকেন্ড।

---

## 📌 অংশ ৮: Evaluation — Standard 0.50 Threshold ⚠️ (পরিবর্তিত অংশ)

* **কাজের বিবরণ:** আগে `custom_threshold = 0.35` ব্যবহার হতো (recall বাড়ানোর জন্য)। **এখন স্ট্যান্ডার্ড `0.50` থ্রেশহোল্ড** ব্যবহার হয়, কারণ SMOTE বাদ দেওয়ার পর ০.৩৫ থ্রেশহোল্ড আর প্রয়োজন/উপযুক্ত না।
* **কী কী ব্যবহার হয়েছে:**
  * `predict_proba`: "Not Employable" ক্লাসের সম্ভাবনা বের করা (label encoding-এ এটাই positive/risk ক্লাস)।
  * `classification_report`, `confusion_matrix`, `roc_auc_score`: পারফরম্যান্স মাপা।
  * **সর্বশেষ ফলাফল:** Accuracy = 0.8170, ROC-AUC = 0.8068 (৭৮.০৮% majority-baseline-এর উপরে)।

---

## 📌 অংশ ৯: Feature Importance — Native XGBoost (পরিবর্তিত অংশ)

* **কাজের বিবরণ:** আগে `permutation_importance` ব্যবহার হতো। **এখন XGBoost-এর নিজস্ব (`native`) `feature_importances_`** অ্যাট্রিবিউট ব্যবহার করা হয় — দ্রুত এবং থিসিসের Section 5.10-এর সাথে সামঞ্জস্যপূর্ণ।
* **কী কী ব্যবহার হয়েছে:**
  * ক্যাটাগরিক্যাল ফিচারের one-hot এনকোডেড কলামগুলোর importance যোগ করে মূল অ্যাট্রিবিউটে ম্যাপ করা হয় (যেমন `Academic_Performance_Good` + `Academic_Performance_Poor` + ... = `Academic_Performance`)।
  * **সর্বশেষ Top-৪ ফিচার:** Resume_Score (৩৩.৩%), Overall_Preparedness_Index (২০.৯%), Programming_Skill (১০.৬%), Internships (৯.৯%) — মোট ৭৪.৬% সিদ্ধান্তের ওজন।
  * `joblib.dump`: মডেল পাইপলাইন (`best_employability_pipeline.pkl`) এবং লেবেল এনকোডার (`label_encoder.pkl`) সেভ করা।

---

## 📌 অংশ ১০: compare_algorithms.py — অ্যালগরিদম তুলনা (নতুন ফাইল) 🆕

* **কাজের বিবরণ:** থিসিসের Table 5.6-এর জন্য Logistic Regression, Decision Tree, Random Forest এবং XGBoost — একই split, একই ১৮-ফিচার, একই ০.৫০ থ্রেশহোল্ডে তুলনা করা।
* **কী কী ব্যবহার হয়েছে:**
  * একই preprocessing pipeline ব্যবহার করা হয় সব মডেলের জন্য (fair comparison)।
  * `Decision Tree(max_depth=6)`: থিসিসে বর্ণিত কনফিগারেশন মিলিয়ে।
  * `precision_score`, `recall_score` (minority class-এর জন্য `pos_label`): শুধু accuracy না, minority-class recall/precision-ও রিপোর্ট করা হয়।
  * **ফলাফল:** Logistic Regression সর্বোচ্চ accuracy (০.৮১৯৬) পেয়েছে, XGBoost-এর চেয়ে সামান্য বেশি — কিন্তু XGBoost native feature-importance ও ভবিষ্যতে non-linear feature engineering যোগ করার সুবিধার জন্য বেছে নেওয়া হয়েছে।
  * SVM (Support Vector Machine) বাদ দেওয়া হয়েছে — n=৫০,০০০-এ kernel training ব্যয়বহুল।

---

## 📌 অংশ ১১: main.py — FastAPI প্রেডিকশন সার্ভিস (আপডেট করা)

* **কাজের বিবরণ:** ট্রেইন করা মডেল লোড করে HTTP API-এর মাধ্যমে প্রেডিকশন দেওয়া।
* **কী কী ব্যবহার হয়েছে ও কী ঠিক করা হয়েছে:**
  * **বাগ ফিক্স:** আগে কোড `best_employability_pipeline_v3.pkl` নামে একটা ফাইল খুঁজত যেটা আসলে ছিলই না — সার্ভার স্টার্টআপেই ক্র্যাশ করত। এখন সঠিক ফাইলনেম (`best_employability_pipeline.pkl`, `label_encoder.pkl`) ব্যবহার করা হয়।
  * **CORS মিডলওয়্যার যোগ:** `CORSMiddleware` যোগ করা হয়েছে যাতে `webview.html` (যেটা ব্রাউজারে `file://` প্রোটোকলে খোলা হয়) সরাসরি এই API-কে কল করতে পারে।
  * `predict_proba` + threshold ০.৫০: "Not Employable" ক্লাসের সম্ভাবনার উপর ভিত্তি করে সিদ্ধান্ত।
  * **Gap Analysis (rule-based):** যদি প্রেডিকশন "Not Employable" হয়, তাহলে fixed thresholds চেক করে (Interview_Score < ৮৫, Internships < ২, Projects < ৫, ইত্যাদি) কী কী উন্নতি দরকার তার লিস্ট রিটার্ন করা হয়। এটা SHAP-based attribution না — সহজ rule-based চেক।

---

## 📌 অংশ ১২: webview.html — লোকাল ওয়েব ইন্টারফেস (সম্পূর্ণ নতুন) 🆕

* **কাজের বিবরণ:** থিসিস/API টেস্ট করার জন্য একটা সেলফ-কন্টেইনড HTML পেজ, যেটা ব্রাউজারে সরাসরি খুলে (`file://`) FastAPI সার্ভারের সাথে কথা বলে।
* **কী কী আছে:**
  * একটা ফর্ম (Academic, Technical & Portfolio, Experience & Readiness, Soft Skills বিভাগে সাজানো) যেটা `/predict` endpoint-এ প্রোফাইল পাঠায়।
  * ফলাফল প্যানেল: verdict badge (Employable/Not Employable), probability gauge (animated), এবং gap-analysis লিস্ট।
  * `fetch()` ব্যবহার করে সরাসরি `http://127.0.0.1:8000/predict`-এ কল করা হয় — কোনো ব্যাকএন্ড টেমপ্লেটিং লাগে না।
  * সার্ভার চালু আছে কিনা তা দেখানোর জন্য একটা লাইভ স্ট্যাটাস চিপ।

---

## 📌 সারাংশ: আগের কোডের সাথে মূল পার্থক্য

| বিষয় | আগে (পুরনো Colab নোটবুক) | এখন (train_model.py) |
|---|---|---|
| পরিবেশ | Google Colab + Drive mount | লোকাল Python venv |
| Imbalance handling | SMOTE + `scale_pos_weight` | কোনোটাই না (হাইপারপ্যারামিটার টিউনিং-ই যথেষ্ট) |
| Decision threshold | ০.৩৫ (কাস্টম) | ০.৫০ (স্ট্যান্ডার্ড) |
| Feature importance | `permutation_importance` | Native `feature_importances_` |
| Accuracy | ৬৩.৮৪% (majority baseline-এর নিচে) | ৮১.৭০% (baseline-এর উপরে) |
| অ্যালগরিদম তুলনা | ছিল না | `compare_algorithms.py`-তে ৪টি অ্যালগরিদম |
| ইউজার ইন্টারফেস | ছিল না (শুধু API) | `webview.html` — লোকাল ব্রাউজার ইন্টারফেস |
| main.py bug | `_v3.pkl` ফাইল খুঁজত (crash) | সঠিক ফাইলনেম + CORS ফিক্স করা |
