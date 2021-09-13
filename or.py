from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
def main(data,eta,epochs,filename):

    df= pd.DataFrame(data)

    x,y = prepare_data(df)

    eta = 0.1

    epochs=10

    model = perceptron(eta=eta,epochs=epochs)
    model.fit(x,y)

    _ = model.totalloss()

    save_model(model,filename)
    save_plot(df,"or.png",model)

if __name__ == '__main__':
    OR = {
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
        "y" : [0,1,1,1],
    }
    eta = 0.1
    epochs=10
    main(data=OR,eta=eta,epochs=epochs,filename="or.model")