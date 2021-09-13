from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
OR = {
    "x1" : [0,0,1,1],
    "x2" : [0,1,0,1],
    "y" : [0,1,1,1],
}

df= pd.DataFrame(OR)

x,y = prepare_data(df)

eta = 0.1

epochs=10

model = perceptron(eta=eta,epochs=epochs)
model.fit(x,y)

_ = model.totalloss()

save_model(model,filename="or.model")
save_plot(df,"or.png",model)