# Astro Dataton 2024

Este repositorio contiene el código y los datos necesarios para llevar a cabo las submissions para la predicción de la distribución de masa en cúmulos de galaxias.

## Estructura de Carpetas

- Para ejecutar el código, es necesario que exista una carpeta llamada "dataset" que contenga las 8 subcarpetas con los datos de entrenamiento descomprimidos. Cada una de estas subcarpetas debe incluir las carpetas "EPSILON" y "KAPPA".
- Además, dentro de la carpeta "dataset", deben encontrarse las carpetas "test_public" y "test_private".
- En la carpeta "experimentos/unet_mae_magandorgraderr_lr_totalpartition", es necesario crear dos subcarpetas: "models" y "submissions".


## Instalación

Para instalar las librerías necesarias, utiliza el archivo `environment.yml`. Puedes crear un entorno con el siguiente comando:

```bash
conda env create -f environment.yml
```

Luego, activa el entorno:

```bash
conda activate nombre_del_entorno
```

## Generación de tableros

Para generar los tableros, ejecuta el Jupyter Notebook experimentos/unet_mae_magandorgraderr_lr_totalpartition/purrfect_predictors_dataton.ipynb. Este notebook generará automáticamente las submissions y guardará los archivos de salida en la carpeta submissions. Los archivos generados serán:

- best_model_partition_10_v2
- private_partition_14


