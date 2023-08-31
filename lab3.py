#Importing Libraries
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as pt

def build_model(learning_rate):
    
    # Add sequential model from keras library into model
    model = tf.keras.models.Sequential()
    
    
    # add one layer of one neuron to this model
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),activation='relu',use_bias=True))
    
    
    # compile the model using RMSprop compiler
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    
    return model


def train_model(model, features,labels, epochs, batch_size):

    #Feed the feature values and the label values to the model
    history = model.fit(x = features, y = labels, batch_size = batch_size, epochs = epochs)
    
    
    # Gather the trained model's weight and bias
    trained_weight = model.get_weights()[0]
    print("trained weight:", trained_weight)
    trained_bias = model.get_weights()[1]
    print("trained bias:", trained_bias)
    
    
    #Store the list of epochs from the trained model separately in a variable called epochs
    epochs = history.epoch
    print("epochs:", epochs)
    
    # Write the history of each epoch using dataFrame
    hist = pd.DataFrame(history.history)
    print("hist", hist)
    
    # get "root_mean_squared_error" from the history
    rmse = hist["root_mean_squared_error"]
    print("rmse:", rmse)
    
    return trained_weight, trained_bias, epochs, rmse

def plot_model(trained_weight, trained_bias, feature, label):
    
    
    # set xLabel as feature and yLabel as label on plot using pyplot
    pt.xlabel("feature")
    pt.ylabel("lable")
    
    # Plot the features values vs. label values
    pt.scatter(feature, label)  # scatter plot anzeigen
    
    # plot a line
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    pt.plot([x0,x1],[y0,y1],c='r') 
    pt.show() # Plot Anzeigen
    
def plot_loss_curve (epochs, rmse):
    #set xLabel as Epoch and yLabel as "Root Mean Squared Error"
    pt.figure()
    pt.xlabel("Epoch")
    pt.ylabel("Root Mean Squared Error")
    pt.plot(epochs, rmse, label = "Loss")
    pt.legend()
    pt.ylim([rmse.min()*0.97, rmse.max()])
    
    #pt.plot(epochs, rmse)
    pt.show()
    
my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
my_lable = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

# declare a new variable learning_rate = 100
# declare new variable epochs = 500
# declare new variable batch_size = 12
# build model by calling build_model function. Use learning_rate as input parameter. Save the return value in a new variable my_model
# train model by calling train_model function. Use my_model, my_feature, my_label, epochs, batch_size as input parameters. Save return values (trained_weight, trained_bias, epochs, rmse) in new variables
# plot train model using plot_model function. 
# plot loss curve using plot_loss_curve function.
# see the result. Based on result, update learning_rate and epoch, to get correct linear regression line and loss curve approaching zero.

learning_rate = 0.1 #dritte Test, funktioniert am besten

epochs = 50 # dritte Test, funktioniert am besten

batch_size = 12

#create model by calling build_model
model = build_model(learning_rate = learning_rate)


#train model by calling train_model
trained_weight, trained_bias, epochs, rmse = train_model(model, my_feature, my_lable, epochs, batch_size)


print("rmse:", rmse)

#Output diagram
plot_model(trained_weight, trained_bias, my_feature, my_lable)

#plot loss curve
plot_loss_curve (epochs, rmse)