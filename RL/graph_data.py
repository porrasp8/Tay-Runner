import numpy as np
import matplotlib.pyplot as plt

def leer_lista_desde_archivo(nombre_archivo):
    return np.load(nombre_archivo)

def graficar_lista(datos):
    plt.plot(datos, marker='o', color='blue')
    plt.title('Graficar Datos desde Archivo')
    plt.xlabel('Iteración')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.show()

# Ejemplo de uso
nombre_archivo = './rewards.npy'
datos_leidos = leer_lista_desde_archivo(nombre_archivo)
graficar_lista(datos_leidos)