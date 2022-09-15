from flask import Flask, request, jsonify

from train import CalcularFP, Init, Recomendar

app = Flask(__name__)


association_results_categorias = CalcularFP(
    dataset=Init(), min_supp=0.02)


@app.route("/", methods=['GET', 'POST'])
def get_recommendation():
    data = request.json
    print(data)

    # dataJson = jsonify(data)

    respuesta = Recomendar(association_result=association_results_categorias, categorias_carrito=[
        "NEVERA", "DROGUERIA", "BEBIDAS"], confidence_min=0.2, lift_min=1.2)
    return respuesta
