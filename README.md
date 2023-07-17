## Pneumonia Classification Web Application

### Open the app with the link:
https://pneumonia-classifier.streamlit.app/

### This application was built in several stages: 
1. collect chest xrays data
2. divide the images into 2 groups: test and validation sets
3. build a model using the tensorflow pipeline
4. train the model with test data, when epochs = 10, the data is overtrained, so epochs are reduced to 3
5. test the model on validation set
6. save the model
7. create a streamlit application with a UI part and a model to determine the presence or absence of pneumonia
8. deploy the application
