# yolo_comparaison

# ATTENTION

Le modèle de Colin entraîné sur 6500 images est un modèle de détection d'objet, on ne peut donc pas faire d'inférence pour obtenir des polygones car il n'y a pas de masques. On ne peut qu'afficher des bounding boxes : 

```bash 
yolo detect predict model=/Users/ayoub/Documents/GitHub/yolo_comparaison/models/detection_6500_images_Colin.pt source='/Users/ayoub/Documents/GitHub/yolo_comparaison/images_Steles/18028-21.JPG' 
```

# Clone le repo : 

```bash
git clone https://github.com/AY2018/yolo_comparaison.git
```


# Télcharger sur whatsapp et Dézipper models.zip qui contient les modèles 

# Créer un nouvel environement Conda si nécessaire + Installer les librairies 

```bash
pip install -r requirements.txt
```

# 3 dossiers d'images 

images_Colin = images de Colin 

images_Steles = images de Stèles (full Image)

images_Steles_INTEXT = images de la zone intext. 


# 3 modèles 

seg_160_images_Colin = Dernier modèle de SEGMENTATION entraîné au labo sur 160 images, 120 epochs. => Sur les images de Colin 

seg_steles_70_images_Ayoub = Modèle de SEGMENTATION entraîné sur 70 images, 200 epochs. -> Sur des images de la zone Intexte des Stèles 

detect_6500_imafes_Colin.pt = Modèle de DÉTECTION entraîné sur 65000 images, 200 epochs. -> Images de Colin

