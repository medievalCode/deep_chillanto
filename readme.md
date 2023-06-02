# Classificación de la base de datos Chillanto

Los paquetes necesarios para correr el código incluido en este projecto
están en `requirements.txt`. La mejor opción para instalar estos paquetes
es crear un nuevo ambiente virtual e instalar los paquetes dentro de ese
ambiente;

```commandline
pip install -r requirements.txt
```

Antes de correr este código es necesario tener un folder llamado `chillanto`
afuera de esta carpeta, 

![img.png](images/img.png)

La carpeta `chillanto` debe contener todas las muestras de llanto. El clasificador
principal está programado para clasificar 5 clases: 

* asphyxia
* deaf
* normal
* hunger
* pain

Entonces dentro de esta carpeta deben estar otras carpetas con las muestras de llanto
separadas por clase en cada una de las carpetas. 

![img.png](images/img_2.png)

Incluyo el archivo `chillanto_metadata.csv` para resolver cualquier duda de la organizacion
de las carpetas.

Cuando las muestras estén listas, simplemente se debe ejecutar el codigo `main.py`;

```commandline
python main.py
```

Después de que el modelo termina de entrenar, se guardarán dos versiones del modelo en
la carpeta `saved_models`. Uno en formato `hdf5` y el otro como archivo `json` y `h5`.
Para cargar el modelo entrenado y probarlo para generar una matriz de confusión, es 
necesario ejecutar el codigo `test_model.py`

```commandline
python test_model.py
```