from flask import Flask, request, jsonify

from train import CalcularFP, Init, Recomendar

app = Flask(__name__)

association_results_categorias = CalcularFP(
    dataset=Init(), min_supp=0.02)


@app.route("/", methods=['GET', 'POST'])
def get_recommendation():
    try:
        data = request.json
        respuesta = Recomendar(association_result=association_results_categorias,
                               categorias_carrito=data["categoriasEnCarrito"], confidence_min=0.2, lift_min=1.2)

        if respuesta == None:
            return jsonify({"data": "INVALID"})
        else:
            return jsonify({"data": respuesta})

    except:
        return jsonify({"data": "INVALID"})
