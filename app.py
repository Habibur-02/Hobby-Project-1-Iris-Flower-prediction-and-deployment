import torch
from torch import nn
from flask import Flask, render_template, request

class LoadIrisv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 10)
        self.layer2 = nn.Linear(10, 20)
        self.layer3 = nn.Linear(20, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 2. Create an instance of your model
model = LoadIrisv1()

model.load_state_dict(torch.load("03_pytorch_classification_model_2.pth"))
model.eval()
classes = ["Setosa", "Versicolor", "Virginica"]

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get inputs
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width']),
            ]
            print(features)
            # Predict
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
                prediction = classes[predicted_class]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)