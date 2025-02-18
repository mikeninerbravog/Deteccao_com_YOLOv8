import json
import os
import glob

# Defina os diretórios
json_dir = './json'  # Pasta com arquivos JSON do LabelMe
output_dir = './labels'  # Pasta para salvar os arquivos TXT no formato YOLO
image_dir = './images'  # Pasta com as imagens

# Cria a pasta de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Lista todos os arquivos JSON
json_files = glob.glob(os.path.join(json_dir, '*.json'))

# Classes definidas (ajuste conforme suas classes)
class_names = ["tank", "warship", "soldier"]

def convert_to_yolo(size, points):
    # Converte polígono (LabelMe) para bounding box (YOLO)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calcula YOLO format: x_center, y_center, width, height
    x_center = (x_min + x_max) / 2 / size[0]
    y_center = (y_min + y_max) / 2 / size[1]
    width = (x_max - x_min) / size[0]
    height = (y_max - y_min) / size[1]
    
    return x_center, y_center, width, height

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    image_name = os.path.splitext(os.path.basename(data['imagePath']))[0]
    image_path = os.path.join(image_dir, data['imagePath'])

    if not os.path.exists(image_path):
        print(f"[Aviso] Imagem não encontrada: {image_path}")
        continue

    # Obtém tamanho da imagem
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    label_file = os.path.join(output_dir, f"{image_name}.txt")
    
    with open(label_file, 'w') as out_file:
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_names:
                print(f"[Aviso] Classe desconhecida: {label}")
                continue

            class_id = class_names.index(label)
            points = shape['points']
            x_center, y_center, width, height = convert_to_yolo((img_width, img_height), points)

            out_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"[✔] Convertido: {json_file} -> {label_file}")
