# Prova de Conceito (POC) - Detecção com YOLOv8

## Informações do Projeto
- Aluno: Marcello S. Bastos  
- Curso: BairesDev - Machine Learning Practitioner  
- Mês/Ano: Fevereiro 2025  

## Descrição do Projeto
Este projeto é a Prova de Conceito (POC) para a conclusão do curso, demonstrando a implementação do modelo YOLOv8 (You Only Look Once) para a detecção de três classes militares:
- tank (tanque)  
- soldier (soldado)  
- warship (navio de guerra)  

## Ferramentas Utilizadas
- YOLOv8 (Ultralytics): Modelo de detecção de objetos.
- Labelme: Ferramenta de anotação de imagens (instalada via `pip`).
- `conv_json_text.py`: Script para conversão de arquivos JSON (gerados pelo Labelme) em arquivos TXT compatíveis com YOLO.

## Estrutura do Dataset (`MyDrive/yolo_war_dataset`)
```
MyDrive/
└── yolo_war_dataset/
    ├── images/
    │   ├── train/   # Imagens de treinamento
    │   └── val/     # Imagens de validação
    ├── labels/
    │   ├── train/   # Labels de treinamento (convertidas do formato JSON)
    │   └── val/     # Labels de validação
    └── probe/       # Imagens para teste final
```

## Instalação e Configuração
```bash
# Conectar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Ajustar codificação para UTF-8
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Instalar o YOLOv8 e o Labelme
!pip install ultralytics --upgrade
!pip install labelme
```

## Conversão de Anotações (Labelme para YOLO)
Os arquivos `.json` gerados pelo Labelme foram convertidos para `.txt` com o script `conv_json_text.py`:
```bash
# Executar conversão de JSON para TXT
python conv_json_text.py --input /path/to/json --output /path/to/txt
```

## Treinamento do Modelo
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data='/content/drive/MyDrive/yolo_war_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolo_war_detection',
    project='/content/drive/MyDrive/yolo_war_dataset/',
    device=0
)
```

## Teste com Imagens da Pasta `probe`
```python
import glob
from PIL import Image
from ultralytics import YOLO
import IPython.display as display

model = YOLO('/content/drive/MyDrive/yolo_war_dataset/yolo_war_detection/weights/best.pt')
probe_images = glob.glob('/content/drive/MyDrive/yolo_war_dataset/probe/*.jpg')
for img_path in probe_images:
    results = model.predict(img_path, save=True, conf=0.4)
    display.display(Image.open(img_path))
```

## Conclusão
- Detecção de 3 classes: `tank`, `soldier`, `warship`.
- Uso do Labelme: Criação de anotações.
- Conversão com `conv_json_text.py`: Compatibilidade com YOLO.
- Execução completa via Google Colab com Google Drive.

Desenvolvido por: Marcello S. Bastos  
Curso: BairesDev - Machine Learning Practitioner (Fevereiro 2025)
