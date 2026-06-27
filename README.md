dataset (put in the "data" folder): https://catalog.data.gov/dataset/crime-data-from-2020-to-present
train the model: python train_models.py
launch a web application: python -m streamlit run app.py

1) A bar chart with the 10 most common types of crimes and their number
2) Crime type prediction (by location, time, and victim characteristics)
3) Neighborhood crime risk prediction (the likelihood of a crime being committed in a specific neighborhood at a specific time)
4) A list of all 20 neighborhoods by their crime rate
5) Crime Hotspot Prediction (prediction of crime “hot spots” by coordinates)
