from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
AND = {
    "x1" : [0,0,1,1],
    "x2" : [0,1,0,1],
    "y" : [0,0,0,1],
}

df= pd.DataFrame(AND)

x,y = prepare_data(df)

eta = 0.1

epochs=10

model = perceptron(eta=eta,epochs=epochs)
model.fit(x,y)

_ = model.totalloss()

save_model(model,filename="and.model")
save_plot(df,"and.png",model)