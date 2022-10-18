from mlxtend.frequent_patterns import association_rules, apriori as ap, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import mlxtend
import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
import pylab
import calendar
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings


def Init():
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # %matplotlib inline
    sn.set_palette(palette="OrRd")

    # Importamos los datos de las ventas
    salesCsv = pd.read_csv("ventasDataset.csv")

    timeStamp_column = pd.to_datetime(salesCsv['createdAt'], unit='ms', utc=True).map(
        lambda x: x.tz_convert('Africa/Ceuta'))
    salesCsv['createdAt'] = timeStamp_column
    salesCsv['hora'] = timeStamp_column.dt.hour
    salesCsv['diaSemana'] = timeStamp_column.dt.day_of_week

    salesNoNa = salesCsv.dropna(subset=['categoria'])

    # print("Tamaño del Dataset inicial:", len(salesCsv.index))

    rows_nan = salesCsv.isna().any(axis=1).sum()
    # print("Filas con al menos un NA:", rows_nan)

    # print("Tamaño final sin las filas con NA en categoria:", len(salesNoNa.index))
    # print("Número de filas eliminadas: ",
    #       (len(salesCsv.index) - len(salesNoNa.index)))

    # display(salesCsv.head(10))

    ventas = salesNoNa.groupby(["ventaID"])[
        ["nombreProd", "ean", "cantidadVendida", "categoria"]].agg(lambda x: list(x)).reset_index()

    # Obtenemos una lista de todos los productos por EAN
    productos_dict = salesNoNa[['ean']].drop_duplicates()
    productos_dict = {name: np.array(value)
                      for name, value in productos_dict.items()}
    productos = productos_dict["ean"]

    # Obtenemos una lista de todos los productos por nombre
    productos_nombre_dict = salesNoNa[['nombreProd']].drop_duplicates()
    productos_nombre_dict = {name: np.array(
        value) for name, value in productos_nombre_dict.items()}
    productos_nombre = productos_nombre_dict["nombreProd"]

    # Obtenemos una lista de todas las categorias
    categorias_dict = salesNoNa[['categoria']].drop_duplicates()
    categorias_dict = {name: np.array(value)
                       for name, value in categorias_dict.items()}
    categorias = categorias_dict["categoria"]

    datasetEan = list(ventas["ean"])
    datasetCategoria = list(ventas["categoria"])

    dfCategorias = []
    for lista in datasetCategoria:
        mylist = list(dict.fromkeys(lista))
        dfCategorias.append(mylist)

    return dfCategorias


def CalcularFP(dataset, min_supp):
    con = TransactionEncoder()
    con_arr = con.fit(dataset).transform(dataset)
    df = pd.DataFrame(con_arr, columns=con.columns_)

    # Ejecución del algoritmo
    resFP = fpgrowth(df, min_support=min_supp, use_colnames=True)
    association_resFP = association_rules(
        resFP, metric="lift", min_threshold=1.2)

    association_resFP["antecedents_len"] = association_resFP["antecedents"].apply(
        lambda x: len(x))

    # display(association_resFP[ (association_resFP['antecedents_len'] <= 1) &
    #        (association_resFP['confidence'] > 0.2) &
    #        (association_resFP['lift'] > 1.2) ])

    # display(association_resFP)
    return association_resFP


def Recomendar(association_result: pd.DataFrame, categorias_carrito, confidence_min: float, lift_min: float):
    recomendador: pd.DataFrame = association_result[(association_result['antecedents_len'] <= 1) &
                                                    (association_result['confidence'] >= confidence_min) &
                                                    (association_result['lift'] >= lift_min)]

    indice = -1
    indiceRecomendador = -1

    for _, recomendacion in recomendador.iterrows():
        indiceRecomendador += 1
        confianza = -1

        for _, categoria in enumerate(categorias_carrito):
            if categoria in recomendacion["antecedents"]:
                if confianza < recomendacion["confidence"]:
                    confianza = recomendacion["confidence"]
                    indice = indiceRecomendador

    if indice >= 0:
        recomendacionFinal = recomendador.iloc[[indice]]
        res = list(list(recomendacionFinal["consequents"])[0])[0]
        return res

    return None


# Ejecución del modelo
# respuesta = Recomendar(association_result=association_results_categorias, categorias_carrito=[
#     "NEVERA", "DROGUERIA", "BEBIDAS"], confidence_min=0.2, lift_min=1.2)

# print(respuesta)
