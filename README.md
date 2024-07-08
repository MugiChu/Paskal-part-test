# Иерархическая семантическая сегментация изображений

Этот проект реализует иерархическую семантическую сегментацию изображений с использованием нейронной сети U-Net. Модель обучается на датасете Pascal-Part и способна сегментировать изображения на следующие классы:

* background
* body
    * upper_body
        * low_hand
        * up_hand
        * torso
        * head
    * lower_body
        * low_leg
        * up_leg

## Структура проекта
```
Paskal-part-test/
│
├── data/
│   ├── JPEGImages/
│   └── gt_masks/
│
├── models/
│   ├── dataset.py
│   ├── model.py
│   └── transforms.py
│
├── notebooks/
│   ├── train.ipynb
│   └── visualize.ipynb
│
├── utils/
│   └── transforms.py
│ 
├── .gitignore
├── README.md
└── requirements.txt

```
## Среда
Для создания среды conda с необходимыми пакетами используйте следующую команду:
```bash
conda create --name <env> --file <this file>
```


## Запуск проекта

Чтобы запустить проект, вы можете выполнить следующую команду в терминале:
```bash
jupyter notebook notebooks/UNet_experiment.ipynb
```

Эта команда запустит Jupyter Notebook и откроет файл UNet_experiment.ipynb, содержащий код проекта. В этом файле вы можете найти подробные инструкции по запуску обучения модели и ее тестированию.

## Для ознакомления с данными вы можете выполнить следующую команду в терминале:
```bash
jupyter notebook/Preview_images_and_masks.ipynb
```
