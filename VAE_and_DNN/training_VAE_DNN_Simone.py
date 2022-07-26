from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import sys
import numpy
import pandas as pd
import tensorflow as tf
from keras import layers
import keras


#taking the model
#from VAE_model_extended_moreDKL import *
from VAE_testDK_Reco_Loss import *
import matplotlib.pyplot as plt

import ROOT
# ROOT.ROOT.EnableImplicitMT()
#RDataFrame = ROOT.RDF.Experimental.Distributed.Spark.RDataFrame
 


#
# variable from the nutple
#
#pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
#       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
#       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
#kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
kinematicFilter = "ptj1 > 30 && abs(etaj1-etaj2) > 2. && ptj2 >30 && mjj>200"
ntuple_location = "../../ntuples4Momentum/"
dfSM = ROOT.RDataFrame("SSWW_SM",ntuple_location+"ntuple_SSWW_SM.root")
dfSM = dfSM.Filter(kinematicFilter)
dfBSM = ROOT.RDataFrame("SSWW_cW_QU",ntuple_location+"ntuple_SSWW_cW_QU.root")
dfBSM = dfBSM.Filter(kinematicFilter)

np_SM = dfSM.AsNumpy()
wSM = dfSM.AsNumpy(["w"])
npd = pd.DataFrame.from_dict(np_SM)
print(npd.head(5))
npd.drop(['w','phil1','phil2',"phij1","phij2"],axis='columns', inplace=True)
wpdSM = pd.DataFrame.from_dict(wSM)
npd.info()

# Take all columns og dfBSM because of using RDateFrame.AsNumpy() without any filter
np_BSM = dfBSM.AsNumpy()
wBSM = dfBSM.AsNumpy(["w"])
npd_BSM = pd.DataFrame.from_dict(np_BSM)
npd_BSM.drop(['w','phil1','phil2',"phij1","phij2"],axis='columns', inplace=True)
wpdBSM = pd.DataFrame.from_dict(wBSM)

nEntries = 3000000
npd = npd.head(nEntries)
npd_BSM = npd_BSM.head(nEntries)
wpdSM = wpdSM.head(nEntries)
wpdBSM = wpdBSM.head(nEntries)
#to be done for all the pt and mass and met variables
for var in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"Ej1","Ej2","El1","El2"]:
    npd[var] = np.log10(npd[var])
    npd_BSM[var] = np.log10(npd_BSM[var])

Y_true = np.full(npd.shape[0],0)
Y_true_BSM = np.full(npd_BSM.shape[0],1)
#concatenating SM and BSM
samples_df = pd.concat([npd,npd_BSM])
labels_arr = np.concatenate((Y_true,Y_true_BSM))
samples_df['labels'] = labels_arr

# Shuffle the dataset in order to shuffle the order of BSM & SM data
samples_df = samples_df.sample(frac=1)

shuffled_labels_df = samples_df['labels']
shuffled_samples_df = samples_df.drop(['labels'], axis='columns')


X_train, X_rem, y_train, y_rem = train_test_split(shuffled_samples_df,
                                                  shuffled_labels_df,
                                                  train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,
                                                    y_rem, 
                                                    test_size=0.5)
SM_train,SM_test,_,_ = train_test_split(npd, npd, test_size=0.2, random_state=1)
BSM_train,BSM_test,_,_ = train_test_split(npd_BSM, npd_BSM, test_size=0.2, random_state=1)
#wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)

#BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
#wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
#print wx_train,X_train
#wx = wx_train["w"].to_numpy()
#wxtest = wx_test["w"].to_numpy()
#wBSM = wBSM_train["w"].to_numpy()
#wBSMtest = wBSM_test["w"].to_numpy()
# scale data

scaler = MinMaxScaler()
#t = StandardScaler()
scaler.fit(shuffled_samples_df)
scaled_x_train = scaler.transform(X_train)
scaled_x_valid = scaler.transform(X_valid)
scaled_x_test = scaler.transform(X_test)

n_inputs = npd.shape[1]
original_dim = n_inputs

intermediate_dim = 50 #50 by default
input_dim = 10 #was 20 in default
half_input = 7 #was 20 in the newTest
latent_dim = 10 #tried 7 as well  #was 3 for optimal performance
epochs = 20 #80
batch_size=64 #32
nameExtenstion = str(intermediate_dim) + "_" +\
                 str(input_dim) + "_" +\
                 str(half_input) + "_" +\
                 str(latent_dim) + "_" +\
                 str(epochs) + "_" +\
                 str(batch_size)

vae = VariationalAutoEncoder(original_dim, 
                             intermediate_dim, 
                             input_dim, 
                             half_input, 
                             latent_dim,
                             )  
#vae.compile(
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
#               loss=tf.keras.losses.MeanSquaredError()
#           )
#vae.compile(
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#               run_eagerly=True, loss="binary_crossentropy",
#               metrics = [tf.keras.metrics.BinaryAccuracy()]
#           )
#vae.compile(
#               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
#               loss_weights=[0.1],loss="binary_crossentropy",
#               metrics = [tf.keras.metrics.BinaryAccuracy()]
#           )
vae.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss="binary_crossentropy",
                metrics = [tf.keras.metrics.BinaryAccuracy()]
            )
hist = vae.fit(scaled_x_train,
               y_train,
               validation_data=(scaled_x_valid,y_valid), 
               epochs=epochs, 
               batch_size = batch_size) 
encoderDecoder =  EncoderDecoder(original_dim,
                                 intermediate_dim,
                                 input_dim,
                                 half_input,
                                 latent_dim)
reco = encoderDecoder.predict(scaled_x_test)
#encoder = LatentSpace(intermediate_dim,input_dim,half_input,latent_dim)
#z = encoder.predict(X_train)

keras.models.save_model(encoderDecoder,'encoderDecoder_newModelUsingKL_Reco_Loss_newWayToAddUpSamples_'+nameExtenstion)
keras.models.save_model(vae,'vae_newModelUsingKL_Reco_Loss_newWayToAddUpSamples_'+nameExtenstion)
#numpy.savetxt("lossVAE_test_newModelDimenstions_MinMaxScaler_"+nameExtenstion+".csv",hist.history["loss"],delimiter=",")
#vae=tf.keras.models.load_model('vae_test_newModelUsingLatentSpace_'+nameExtenstion)


output_SM = vae.predict(SM_test)
output_BSM = vae.predict(BSM_test)

#print output_SM
#print output_BSM
bins=100
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(output_SM,
        bins=bins, 
        density=1,
        range=[0.,1.],
        histtype="step",
        color="red",
        alpha=0.6,
        linewidth=2,
        label="SM Output",)                        
ax.hist(output_BSM,
        bins=bins, 
        density=1,
        range=[0.,1.],
        histtype="step",
        color="blue",
        alpha=0.6,
        linewidth=2,
        label="BSM Output",) 
plt.rc('legend',fontsize='small')    
plt.show()


x_bins = range(len(hist.history["loss"]))
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(x_bins,hist.history["loss"],color="blue",label="training losss")                        
plt.rc('legend',fontsize='small')    
plt.show()
