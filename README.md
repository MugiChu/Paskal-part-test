# Иерархическая семантическая сегментация изображений

## структура проекта:

project/
├── data/
│   ├── JPEGImages/
│   ├── gt_masks/
│   ├── train.txt
│   └── val.txt
├── models/
│   ├── dataset.py
│   └── model.py
│   └── unet_model.pth
├── notebooks/
│   └── UNet_experiment.ipynb
│   └── Previev_images_and_masks.ipynb
├── utils/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md

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
