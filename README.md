## Plant Project (Local)

Bu klasör yapısı ile:
- `notebooks/`: Jupyter notebook'lar (`.ipynb`)
- `src/`: Python kodu (tekrar kullanılabilir modüller)
- `data/`: veri (ham/işlenmiş)
- `models/`: kaydedilen modeller (`.keras`, `.h5`)
- `outputs/`: metrikler, grafikler, confusion matrix, history JSON vb.
- `logs/`: eğitim logları
- `class_names/`: sınıf isimleri JSON vb.

Önerilen akış:
1) Notebook'ları `notebooks/` altında tut.
2) Tekrar kullanılan kodu `src/` altına taşı.
3) Eğitim sırasında `models/` ve `outputs/` altına mutlaka kayıt al (checkpoint + history + görseller).

### Dataset indirme (Kaggle)

Projeyi klonlayan birinin veri hazırlaması için aşağıdaki komutlar yeterlidir (Kaggle API kurulu ve `kaggle.json` ayarlı olmalıdır):

```bash
# Leafsnap
kaggle datasets download -d xhlulu/leafsnap-dataset -p data --unzip

# PlantNet-300K
kaggle datasets download -d noahbadoa/plantnet-300k-images -p data --unzip
```

İndirilen zip'ler `data/` klasörü altında açılır; notebook içindeki kodlar bu yapıyı kullanarak `leafsnap_subset_50`, `leafsnap_split` ve `combined_split` klasörlerini üretir.

