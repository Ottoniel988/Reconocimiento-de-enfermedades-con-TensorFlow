# Reconocimiento-de-enfermedades-con-TensorFlow
Modelo para reconocer enfermedades de la piel como herpes, vitiligo, tiña

Proyecto creado con el fin de utilizar las nuevas tecnologías en el ambito de la inteligencia artificial en esta practica se utilizo un modelo ya preentrenado que nos proporciona TensorFlow asi mismo utilizamos una API denominada resnet101_coco solo necesitas presentar una imagen o fotografia de una parte del cuerpo donde tengas alguna imperfección y el modelo te da como respuesta un porcentaje de que puede ser herpes, vitiligo o tiña.


![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/descarga-_14_.jpeg)


![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/aaaa.jpeg)


![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/tina_28_.jpeg)

![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/tina_32_.jpeg)

![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/tina_40_.jpeg)

![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/vitiligo_53_.jpeg)


![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/vitiligo_66_.jpeg)

![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/vitiligo_57_.jpeg)

![Screenshot](https://github.com/Ottoniel988/Reconocimiento-de-enfermedades-con-TensorFlow/blob/master/output/img_pruebas/vitiligo-_100_.jpeg)






# TECNOLOGÍAS UTILIZADAS:
1. [React js](https://es.reactjs.org/)

2. [TensorFlow](https://www.tensorflow.org/)

3. [Python](https://www.python.org/)

4. [Visual studio code](https://code.visualstudio.com/)

5. [Anaconda](https://www.anaconda.com/)

6. [Pandas](https://pandas.pydata.org/)

7. [Numpy](https://numpy.org/)

8. [Matplotlib](https://matplotlib.org/)


# Versiones

A continuación se muestra en la Tabla I el detalle de cada versión especificando el commit y su descripción de la funcionalidad incluida.

| No. | Commit | Descripción |
| ------ | ------ | ------ |
| 1 | a0d75c6beafe0cd9fa31188a5e0e103275164be4  | Se cargan las imagenes y se crean sus XML con Labellmg |
| 2 | 24183893cf4839e87b5ac4a9c7dafc3fd1f7baf0  | Se configura el modelo resnet_coco_101 |
| 3 | 1f5fe62747d30048d442289f1495869eac1034b7  | No se subio debido al peso de los modelos esta el link en drive  |




# Como ejecutar esta aplicación 
1. Clona el repositorio en tu maquina local.

2. Descarga el modelo, modelo_congelado y el train donde estan los step ya preentrenados solo para ejecutar.

3. Copia y pega estas 3 carpetas dentro del repositorio que clonastes.

4. Ingresa la imagen que desees verificar en la carpeta img_pruebas, formato (jpeg).

5. Borra las imagenes que estan en la ruta output/img_pruebas, en esta ruta es donde te almacenara el resultado.

6. Ejecuta el comando python object_detection/object_detection_runner.py

7. Listo ya abras generado tu predicción.




# LINK de descarga=> https://drive.google.com/file/d/1em1du1_QoAzqU40MYM6KCstEnn0HUlhJ/view?usp=sharing



# Se listan errores que se pueden presentar en el proceso de entrenamiento.
# Proyecto hecho en Windows.

1.Al momento de ejecutar el comando export is not recognized : export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim genera error ya que es para Linux y estamos en windows.

Solución; En Windows se debe realizar lo siguiente:

En lugar de ejecutar el comando export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
Ir desde el cmd (command prompt) a la carpeta descargada del repositorio y ejecutar los siguientes comandos (uno a la vez):

python setup.py build

python setup.py install


Luego ir a la carpeta slim con el comando cd (estando en la deteccion-de-objetos) y ejecutar el siguiente comando:

pip install -e .
 

2.Error de undefined carpeta al ejecutar los siguientes comandos.

python xml_a_csv.py --inputs=img_test --output=test

python xml_a_csv.py --inputs=img_entrenamiento --output=entrenamiento

python csv_a_tf.py --csv_input=CSV/test.csv --output_path=TFRecords/test.record --images=images

python csv_a_tf.py --csv_input=CSV/entrenamiento.csv --output_path=TFRecords/entrenamiento.record --images=images

Solucíon crea dos carpetas de forma tradicional una llamada CSV y la otra TFRecords, y ejecuta nuevamente los 4 comandos anteriores

3.Al ejecutar el comando de entrenamiento python object_detection/object_detection_runner.py genera el error siguiente:

File "object_detection/train.py", line 51, in

from object_detection import trainer

File "C:\Users\51999\Anaconda3\lib\site-packages\object_detection-0.1-py3.7.egg\object_detection\trainer.py", line 33, in

from deployment import model_deploy

ModuleNotFoundError: No module named 'deployment'

Solución: tienes que irte a la carpeta de slim que está dentro del repositorio que clonastes. Una vez ahí, copias los folders deployment y nets y lo pegas en Librery y Lib que están ubicadas en el directorio donde se instaló Anaconda.

Para acceder al dirrectorio, puedes escribir en el buscador de Windows la aplicación Anaconda Navigator. Luego, haz click derecho en ella y presiona "Abrir ubicación del archivo". Si tienes suerte, te redireccionará automáticamente a la carpeta en donde se instaló dicho programa. Sino, te redireccionará al acceso directo del mismo. Bueno, de ser el último caso, es más que suficiente hacer click derecho en el accseo directo y volver a presionar "Abrir ubicación del archivo". Una vez ahí, busca las carpetas de Lib y Library para pegar las que al principio copeaste.

# Para realizar la prediccion tienes que tener las herramientas que se detallan a continuación.











# Requerimientos
Para poder llevar a cabo este tutorial en tu totalidad es importante tener lo siguiente en nuestra computadora.

 

1. Python 3.6

2. Tensorflow 1.14

3. Numpy 1.16.4

4. Pandas

5. Matplotlib

Tarjeta de gráficos (Recomendada para poder hacer un entrenamiento de manera rápida, aunque es posible hacerlo sin GPU, puede llegar a tardar horas o días sin una tarjeta de gráficos NVIDIA)
Pasos a seguir
Estos son los pasos que seguiremos en este tutorial, no te preocupes si algo no queda claro, más adelante lo veremos a detalle.


# Si quieres entrenar tu propio modelo sigue el siguiente tutorial

# Preparación de la data
Preparar nuestros datos para entrenamiento, es decir marcar en un set de entrenamiento las coordenadas donde están los objetos que querremos que detecte. Usaremos un programa que nos ayuda a hacer esto de manera fácil y nos da un archivo XML con la información que requerimos.
Convertir los datos de XML a TFRecord. TFrecord es el formato de imagen que necesita nuestro algoritmo para poder entrenarse.
Entrenamiento
Elegir el modelo que entrenaremos.
Preparar los archivos para entrenamiento
Configuración de modelo
Etiquetas de entrenamiento
Gráfica computacional del modelo
Entrenar el modelo
Congelar el modelo entrenado para poderlo ejecutar después
 

# Prediccion
Dar imágenes a nuestro programa para que detecte los objetos dentro de la misma
 

# Preparación de la data
Antes de empezar necesitamos preparar los datos con los que entrenaremos a nuestro programa. Para esto necesitaremos tener un monton de imagenes, como mínimo recomendaria tener 200 imágenes por cada uno de los objetos que queremos detectar (si en una imagen tenemos 3 de los objetos que queremos aprender a detectar, esto podria contar como tres imágenes).

Al etiquetar nuestras imágenes le diremos a nuestro programa en que coordenadas de nuestras imágenes puede encontrar cada uno de los objetos que queremos que nuestro programa pueda detectar, esta puede ser una tarea algo tediosa, pero usaremos una herramienta que nos facilitara hacerlo y aparte la recompensa al final será grande.

Es importante que tengamos un set de datos variado, es decir que tengamos los objetos que queremos detectar desde varios ángulos, tipos de iluminación, posiciones etc. Al igual, también es importante que nuestras imágenes no sean de gran tamaño ya que pueden llegar a ser mucho para nuestra computadora, por lo cual recomiendo que se haga un a modificación en tamaño para que una imagen no pese más de 0.5 MB

# IMPORTANTE:

Sobre la misma imagen podemos seleccionar distintos elementos que queremos que nuestro programa detecte, por ejemplo, en la misma imagen marcar autos, semáforos, pasos peatonales, camionetas, letreros etc.

 
# Etiquetado con labelImg
 

Estaremos usando un programa llamado labelImg el cual nos facilitara el etiquetado de nuestras imágenes, para descargarlo e instalarlo les recomiento entrar a su Github (https://github.com/tzutalin/labelImg)  tambien pueden descargarlos para Windows o Linux desde esta liga (https://tzutalin.github.io/labelImg)
Básicamente lo que hacemos con este programa es abrir una imagen, seleccionar un recuadro para marcar el objeto que queremos que nuestro programa aprenda a detectar y salvar un XML con la información de las coordenadas. Para ver un poco como funciona pueden ver en este video 
Estas imágenes (Junto con el archivo XML) que se genera las debemos de guardar en la carpeta de ‘imagenes’ dentro del proyecto de GitHub.
https://github.com/tzutalin/labelImg

# Conversion de las imagenes a TFRecord
Ya que tenemos todas la imágenes con sus respectivos XML marcando sus coordenadas, tendremos que convertirlas a un formato llamado TFRecord, este tipo de archivo es especial para que nuestra red neuronal en Tensor Flow pueda ser entrenada, el TFRecord contendra la informacion de todas las imágenes y las coordenadas que marcamos en un solo archivo. Para poderlos llevar a TFRecord, primero convertiremos TODOS los XMLs en un solo archivo tipo CSV, después ya convertiremos estos CSVs al formato final.
Antes de empezar, vamos a duplicar nuestras imágenes, haremos dos carpetas llamadas ‘img_test’ e ‘img_entrenamiento’ en la primera pondremos alrededor del 10% de nuestras imagenes con sus respectivos XMLs y en la segunda el 90% restante. (DUPLICAREMOS LAS IMÁGENES, ES DECIR QUE EN LA CARPETA DE IMÁGENES SEGUIREMOS TENIENDO EL 100% DE LAS IMÁGENES)
Ya que tenemos las imágenes en esta estructura ahora en nuestra terminal (cmd en windows o bash en ubuntu) nos posicionamos en la carpeta ‘deteccion-de-objetos/object_detection’ y ejecutaremos el siguiente comando. (si no corremos este comando al correr los siguientes comandos nos encontraremos con el error que nuestro python no encuentra la paqueteria ‘object_detection’

### export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
 
Después nos regresamos a la carpeta raíz del repositorio que clonamos de GitHub (deteccion-de-objetos) y ejecutamos los siguientes cuatros comandos.

### python xml_a_csv.py --inputs=img_test --output=test

### python xml_a_csv.py --inputs=img_entrenamiento --output=entrenamiento

### python csv_a_tf.py --csv_input=CSV/test.csv --output_path=TFRecords/test.record --images=images

### python csv_a_tf.py --csv_input=CSV/entrenamiento.csv --output_path=TFRecords/entrenamiento.record --images=images


Suponiendo que los scripts corrieron sin problema debemos tener ahora una carpeta llamada TFRecords en la cual tendremos dos archivos, entrenamiento.record y test.record  Estos dos archivos ya contienen la información de todas las imágenes y de las coordenadas de los objetos que marcamos.

Ya con esto listo pasaremos a preparar los archivos necesarios para nuestro entrenamiento y al entrenamiento del modelo que deseemos.

 

# Entrenamiento
Elegir modelo a entrenar
Antes de empezar, debemos decidir que modelo es el que querremos entrenar, algunos nos ofrecen detecciones más veloces, sacrificando certeza o viceversa. Para ver todos los modelos podemos ingresar a esta liga. En este tutorial usaremos el modelo faster_rcnn_resnet101_coco (dar clic para descargar) el cual nos brinda predicciones más veloces.  A su vez tambien descargaremos un archivo tipo config que coincida con el modelo que vamos a entrenar (faster_rcnn_resnet101_coco.config), desde esta liga: (https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

Al descargarlo encontraremos varios archivos, los que nos interesan para entrenar un modelo desde cero son:

### faster_rcnn_resnet101_coco.config (configuración sobre cómo entrenaremos el modelo)
### Model.ckpt.index
### Model.ckpt.meta
### Model.ckpt.data-000000-of-00001
 

Tomemos los archivos con el formato ‘xxxx.ckpt.yyyy’ al igual que el XXXX.config y pasarlos a la carpeta llamada ‘modelo’ dentro de nuestro proyecto.

 

Si no queremos entrenar un programa nosotros y queremos usar un algoritmo ya pre-entrenado podemos usarlo con los archivos mencionados anteriormente y aparte con el ‘frozen_inference_graph.pb’ y ‘saved_model.pb’ el cual tiene toda la información sobre un modelo ya pre-entrenado

 

# Preparar archivos para entrenamiento
 

Ya que elegimos qué modelo vamos a entrenar, ahora es momento de preparar unos archivos de configuración, estos archivos le dirán a nuestro script de entrenamiento, donde encontrar el modelo que encontramos, donde encontrar las imagenes para entrenar, cuales son las etiquetas que usaremos (los objetos que queremos entrenar) entre otros parámetros más.

Estos archivos los podemos en la carpeta llamada ‘configuracion’

Etiquetas (label_map.pbtxt)
En este archivo (configuracion/label_map.pbtxt) le dirá a nuestro algoritmo cuales son las etiquetas sobre el cual lo entrenaremos. El nombre que pongamos en las etiquetas debe ser el mismo que usamos en la herramienta labelImg (incluyendo mayúsculas y espacios). Básicamente este archivo tiene una serie de elementos ‘item’ con su respectivo identificador ‘id’ y nombre de clase ‘name’.

 

He aquí un ejemplo, esto cambia segun el numero de elementos que quieras aprender a detectar.

 

item {
  id: 1
  name: 'Auto'
}
item {
  id: 2
  name: ‘Semaforo’
}

item {
  id: 3
  name: 'Paso Peatonal'
}
 

Labels.txt
Este archivo es similar al pasado pero mucho mas sencillo, solo es una lista de los elementos que queremos detectar, siendo el primer elemento (en el primer renglon) siempre el valor null. He aqui un ejemplo.

null
Motocicleta
Payaso
Sandalia
 

# Configuración de entrenamiento (faster_rcnn_resnet101_coco.config)

Todos los archivos que hemos editado tienen un grado de importancia, pero si fuera a elegir uno como favorito seria este. Si hemos seguido el tutorial este es el archivo que debemos tener en la carpeta de ‘modelo’

Este archivo es el que nuestro script para entrenamiento va a leer para saber parámetros sumamente importantes, tales como:

Donde obtener los tfrecords
Donde obtener el archivos de etiquetas label_map.pbtxt
Donde encontrar los archivos requeridos de nuestro modelo (los checkpoints que aparecen como xxxxx.ckpt.yyyyy)
El número de pasos a entrenar
Batch_size (Número de imágenes que entrenaremos en cada iteración, podemos empezar con un número bajo como 1 e irlo subiendo si vemos que nuestra computadora lo soporta)
 

Para cambiar estos parámetros, debemos de abrir el archivo pipeline.config y cambiar todos los campos que digan ‘PATH_TO_BE_CONFIGURED’.  Los cambios que tenemos que hacer son los siguientes:

 

model {
  faster_rcnn {
    num_classes: 13 (Aqui ponemos el número de objetos a detectar)
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }

...




train_config: {
  batch_size: 1
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "modelo/model.ckpt"
  from_detection_checkpoint: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "TFRecords/entrenamiento.record"
  }
  label_map_path: "configuracion/label_map.pbtxt"
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "TFRecords/test.record"
  }
  label_map_path: "configuracion/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  num_epochs: 1
}
 

# Entrenar
Excelente, ya estamos ahora en la parte más interesante, vamos a empezar a entrenar nuestro algoritmo, teniendo ya todo esto preparado el entrenamiento será sencillo, solo debemos correr el siguiente comando en nuestra terminal.

 

### python object_detection/train.py --logtostderr --train_dir=train --pipeline_config_path=modelo/faster_rcnn_resnet101_coco.config
 

Si todo ha salido bien veremos en nuestra terminal algo similar a esto:

 

INFO:tensorflow:global step 11: loss = 0.6935 (0.648 sec/step)
INFO:tensorflow:global step 12: loss = 0.7426 (0.885 sec/step)
INFO:tensorflow:global step 13: loss = 0.7700 (3.551 sec/step)
INFO:tensorflow:global step 14: loss = 0.8026 (0.664 sec/step)
INFO:tensorflow:global step 15: loss = 0.9608 (0.646 sec/step)
 

Lo que estamos buscando es llegar a un ‘loss’ muy bajo, como mínimo que esté por debajo de 0.9, ya que llegue a este número, podemos terminar el entrenamiento tecleando CTRL -C desde terminal/

# Congelar el modelo entrenado
Ahora que hemos terminado nuestro entrenamiento, tendremos una carpeta llamada ‘train’ en la cual tendremos varios checkpoints (los cuales nos sirven en el futuro por si queremos re-entrenar sobre lo que ya hemos hecho) y un graph.pbtxt , estos archivos son los que contienen la información necesaria para poder hacer predicciones en el futuro, pero antes de esto debemos de ‘congelar’ nuestro modelo, es decir, vamos a convertir nuestros ckeckpoints a un modelo final.

 

Para esto, solo debemos correr un comando, la parte STEP_NUMBER debemos de cambiarla por el ultimo checkpoint que tengamos generado, es decir el de valor mas alto.

 

### python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path  modelo/faster_rcnn_resnet101_coco.config  --trained_checkpoint_prefix train/model.ckpt-684 --output_directory modelo_congelado


Después de haber corrido esto con éxito tendremos un archivo en una carpeta llamada ‘modelo_congelado’, este ya es nuestro archivo listo para generar predicciones.

# Prediccion
 

Listo, hemos llegado al final, espero que todos hayan llegado hasta aquí sin problemas. Ahora es momento de generar predicciones. Para esto solo tenemos que poner las imágenes en las que queremos generar detección de objetos en la carpeta llamada ‘img_pruebas’ y correremos el siguiente comando, el resultado lo obtendremos en una nueva carpeta llamada output.

 

### python object_detection/object_detection_runner.py
 

LISTO! Ya tenemos nuestras imágenes con predicciones.

