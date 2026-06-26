HIGGS Makine Ogrenmesi Final Projesi

Bu projede UCI HIGGS veri seti uzerinde makine ogrenmesi pipeline'i uygulanmistir.

Icerik:
- data/higgs_100k.csv: HIGGS veri setinden rastgele secilmis 100.000 ornek
- data/selected_features.csv: ANOVA F-score ile secilen en iyi 15 ozellik
- data/model_results.csv: Nested cross-validation fold sonuclari
- data/model_summary.csv: Modellerin ortalama metrik sonuclari
- data/roc_data.csv: ROC grafigi icin tahmin skorlari
- figures/: Ozellik secimi, performans, ROC ve flowchart grafikleri
- report/HIGGS_Final_Raporu.docx: Kisa final raporu
- src/: Projede kullanilan Python kodlari

Calistirma sirasi:
1. python src/rastgele_orneklem.py
2. python src/main.py
3. python src/modelleme.py
4. python src/grafikler.py
5. python src/roc_grafigi.py
6. python src/roc_ova.py
7. python src/flowchart_olustur.py
8. python src/rapor_olustur.py

Not:
Orijinal HIGGS.csv.gz dosyasi cok buyuk oldugu icin teslim paketine eklenmemistir. Veri seti UCI Machine Learning Repository uzerinden indirilebilir:
https://archive.ics.uci.edu/ml/datasets/HIGGS
