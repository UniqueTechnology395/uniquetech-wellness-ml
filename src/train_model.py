#----------------------------------------------------------------------------
# * Project: Unique Tech Wellness ML (Python)
#* Author: Unique Tech Team
#* License: MIT License
#* * Copyright (c) 2026 Unique Tech
#* * Permission is hereby granted, free of charge, to any person obtaining a copy
#* of this software and associated documentation files (the "Software"), to deal
#* in the Software without restriction, including without limitation the rights
#* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#* copies of the Software, and to permit persons to whom the Software is
#* furnished to do so, subject to the following conditions:
#* * The above copyright notice and this permission notice shall be included in all
#* copies or substantial portions of the Software.
#----------------------------------------------------------------------------

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import io

# 1. Δημιουργία του Διευρυμένου Dataset (Unique Tech Wellness Data)
# co2: ppm, pm25: μg/m3, hr_boost: αύξηση παλμών, wellness_label: 0=Critical, 1=Warning, 2=Ideal
csv_data = """co2,pm25,hr_boost,wellness_label
420,5,0,2
440,4,1,2
480,5,1,2
600,8,1,2
700,10,2,2
750,11,3,1
800,12,3,1
850,12,3,1
900,10,2,1
950,16,5,1
1000,17,6,0
1050,18,6,0
1100,20,7,0
1200,22,8,0
1400,25,9,0
800,45,4,0
700,50,5,0
1100,8,2,1
1200,10,3,1"""

# Μετατροπή σε DataFrame
df = pd.read_csv(io.StringIO(csv_data))

# Διαχωρισμός χαρακτηριστικών (X) και στόχου (y)
X = df[['co2', 'pm25', 'hr_boost']]
y = df['wellness_label']

# 2. Εκπαίδευση του Decision Tree
# Χρησιμοποιούμε max_depth=3 για να παραμείνει το μοντέλο Εξηγήσιμο (XAI)
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
clf.fit(X, y)

# 3. Εξαγωγή Κανόνων σε μορφή κειμένου (export_text)
# Αυτοί οι κανόνες βοηθούν στη μεταφορά της λογικής στο Arduino
feature_names = ['CO2_Level', 'PM2.5_Level', 'HR_Boost']
tree_rules = export_text(clf, feature_names=feature_names)

print("=== Unique Tech: AI Decision Tree Rules ===")
print(tree_rules)

# 4. Οπτικοποίηση του Δέντρου Απόφασης
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=feature_names, 
          class_names=['Critical', 'Warning', 'Ideal'], 
          filled=True, 
          rounded=True, 
          fontsize=10)

plt.title("Unique Tech - Wellness Prediction Decision Tree")
plt.savefig('decision_tree_visualization.png')
print("\n[Success] Το διάγραμμα του δέντρου αποθηκεύτηκε ως 'decision_tree_visualization.png'")

# 5. Παράδειγμα Πρόβλεψης για δοκιμή
sample_data = [[1100, 15, 6]] # 1100ppm CO2, 15 PM2.5, +6 παλμοί
prediction = clf.predict(sample_data)
labels = {0: "Critical", 1: "Warning", 2: "Ideal"}
print(f"\nΔοκιμαστική Μέτρηση {sample_data}: Πρόβλεψη AI -> {labels[prediction[0]]}")
