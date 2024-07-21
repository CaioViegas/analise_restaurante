import pandas as pd
from otimizacao import modelo_otimizado

if __name__ == '__main__':
    df = pd.read_csv("../data/restaurant_customer_satisfaction.csv")
    modelo_otimizado(df, "HighSatisfaction")