import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model
model = pickle.load(open("best_model_decision_tree.pkl", "rb"))

# Define the mappings(For preprocessing)
warehouse_block_mapping = {'D': 0, 'F': 1, 'A': 2, 'B': 3, 'C': 4}
shipment_mapping = {'Flight': 0, 'Ship': 1, 'Road': 2}
product_importance_mapping = {'low': 0, 'medium': 1, 'high': 2}
gender_mapping = {'Female': 0, 'Male': 1}

@app.route('/')
def input():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:    
        warehouse_block = request.form["Warehouse_block"]
        mode_of_shipment = request.form["Mode_of_shipment"]
        customer_care_calls = int(request.form["Customer_care_calls"])
        customer_rating = int(request.form["Customer_rating"])
        cost_of_the_product = float(request.form["Cost_of_the_product"])
        prior_purchases = int(request.form["Prior_purchases"])  
        product_importance = request.form["Product_importance"]
        gender = request.form["Gender"]
        discount_offered = float(request.form["Discount_offered"])
        weight_in_gms = float(request.form["Weight_in_grams"])

        # Apply mappings(Preprocessing)
        warehouse_block = warehouse_block_mapping.get(warehouse_block, warehouse_block)
        mode_of_shipment = shipment_mapping.get(mode_of_shipment, mode_of_shipment)
        product_importance = product_importance_mapping.get(product_importance, product_importance)
        gender = gender_mapping.get(gender, gender)

        preds = [[warehouse_block, mode_of_shipment, customer_care_calls, customer_rating,
                  cost_of_the_product, prior_purchases, product_importance, gender,
                  discount_offered, weight_in_gms]]
        print("Form Data:", request.form)
        prob = model.predict(preds)
        reach = prob[0]

        return f"There is a {reach*100:.2f}% chance that your product will reach in time."

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return error_message

if __name__ == '__main__':
    app.run(debug=True, port=4000)
