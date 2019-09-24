import h5py
import json
import sqlite3
from os.path import *
from os import listdir, makedirs
import numpy as np
import sys
from pylab import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT_FOLDER = dirname(realpath(__file__))
OUTPUT_FOLDER = join(ROOT_FOLDER, "save/v9_test/2018-12-12_16-54") if len(sys.argv)<2 else sys.argv[1]
Dver = dirname(OUTPUT_FOLDER).split("/")[-2]
NEST_OUTPUT = join(OUTPUT_FOLDER, "nest_output")
HIGHD_FOLDER = join(dirname(ROOT_FOLDER), "HighD")
sys.path.append(join(dirname(ROOT_FOLDER), "genBrain"))
from extract_regions import *


def makeLegendFig(figname, labels, colors, numColumns = 1):
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(3)]
    labelsMAXLEN = np.array([len(str1) for str1 in labels])
    labelsSPLIT = np.split(labels, np.array(range(0,1+len(labels),int(np.ceil(float(len(labels))/float(numColumns)))))[1:] )
    colorsSPLIT = np.split(colors, np.array(range(0,1+len(labels),int(np.ceil(float(len(labels))/float(numColumns)))))[1:] )
    colLength = (0.5+0.13*np.max(labelsMAXLEN))
    fig = plt.figure(figsize=( colLength*float(numColumns), 0.25*float(len(colors))/float(numColumns)))
    for iC in range(numColumns):
        patches = [mpatches.Patch(color=color, label=label) for label, color in zip(labelsSPLIT[iC], colorsSPLIT[iC])]
        fig.legend(patches, labelsSPLIT[iC], loc=(0.0+float(iC)*(1.0/float(numColumns)),0.0), frameon=False)
    plt.savefig(figname)
    close('all')
    plt.gcf()


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return np.array([float(int(value[i:i + lv // 3], 16)) for i in range(0, lv, lv // 3)])


t_trial = 250
dt = 0.1
timeInterval = 1
timeFrom = 200.0
steps = 10

file_ = join(HIGHD_FOLDER, 'PointNeuronModels', Dver, 'ptneu_brain_' + Dver + '.h5')
seed = 75599
stim = 2000.0

# np.random.seed(seed)
# standalone_input = stim * np.fromfunction(lambda x: (x>=200) & (x % 200 <= 20), (int(t_trial / dt),))

databases_file = HIGHD_FOLDER + "/elecDB.db"
specie = 'mouse'
excluded_regions = ["spinal cord", "spinal white matter"] # excluded regions of the database for the whole brain generation
included_regions = [] # Specific regions of the database targeted
 
conn = sqlite3.connect(databases_file)
cursor = conn.cursor()
get_regions(cursor, excluded_regions, included_regions, specie)
maxRegionID = 0
for jn in region_dictionary_to_id_ALLNAME.keys():
    if region_dictionary_to_id_ALLNAME[jn]>maxRegionID:
        maxRegionID = region_dictionary_to_id_ALLNAME[jn]
conn.close()

print "Loading h5 file"
h5file = h5py.File(file_, "r")
xs      = h5file["x"][:]
ys      = h5file["y"][:]
zs      = h5file["z"][:]
Larea   = h5file["Larea"][:]
etype   = h5file["eTypes"][:]
ETTN    = h5file["eTypesToName"][:]
mtype   = h5file["cellTypes"][:]
CTTN    = h5file["cellTypesToName"][:]
cellTypeToColor = [ 0.5*np.ones((len(CTTN)+1,3)), 0.5*np.ones((len(ETTN)+1,3)) ] # last id is grey for unknown
for itypeArray,typeArray in enumerate([CTTN, ETTN]):
    for iTTN,TTN_ in enumerate(typeArray):
        np.random.seed( int(np.sum(np.array([ord(ch) for ch in TTN_]))) + 1 )
        clrRAND = np.random.rand(3)
        if np.sum(clrRAND)<1.5: clrRAND *= 1.5/np.sum(clrRAND)
        clrRAND[clrRAND>1.0] = 1.0
        cellTypeToColor[itypeArray][iTTN,:] = clrRAND[:]
del clrRAND
h5file.close()
makeLegendFig( OUTPUT_FOLDER+"/legend_mtype.png", labels=CTTN, colors= cellTypeToColor[ 0 ], numColumns=3 )
makeLegendFig( OUTPUT_FOLDER+"/legend_etype.png", labels=ETTN, colors= cellTypeToColor[ 1 ] )
del CTTN, ETTN

json_cont = json.loads(open(OUTPUT_FOLDER + "/conversion.json", "r").read())
json_convert = {}
for k, v in json_cont.items():
    json_convert[str(v)] = int(k)
del json_cont

print "Loading masks"
mask_axial    = np.zeros((528,528), np.int16)
mask_coronal  = np.zeros((528,528), np.int16)
mask_sagittal = np.zeros((528,528), np.int16)
mask_axial[   264-264:264+264,264-228:264+228] = np.int16(np.load("mask_axial.npy"))
mask_coronal[ 264-160:264+160,264-228:264+228] = np.int16(np.load("mask_coronal.npy"))
mask_sagittal[264-160:264+160,264-264:264+264] = np.int16(np.load("mask_sagittal.npy")).T

print "Extract regions colors"
Larea_STR = np.array(id_to_region_dictionary_ALLNAME.values())
Larea2clr = np.zeros((43000, 3), np.float32)
itcS_keys = np.array(id_to_color.keys())
itcS_keys = itcS_keys[itcS_keys<43000] # exclude spinal cord and weird retinal id numbers
for ikk,kk in enumerate(itcS_keys):
    color = hex_to_rgb(id_to_color[kk])
    Larea2clr[kk] = color /255.
del itcS_keys

time_diff = float(t_trial - timeFrom)
tau_spike = 2.0 # ms

for time_step in range(steps): 
    timeFromStep = timeFrom + time_diff * time_step/steps 
    t_trialStep = timeFrom +time_diff * (time_step+1)/steps
    print("Loading spikes: [" + str(timeFromStep) + ", " + str(t_trialStep) + "]") 
    spikes = {"senders":[], "times":[]}
    for f in listdir(NEST_OUTPUT):
    # f = listdir(NEST_OUTPUT)[0]
        print(f)
        file_ = join(NEST_OUTPUT, f)
        if isfile(file_):
            with open(file_, 'r') as f:
                for line in f:
                    splitLine = line.split()
                    if "spike_detector-" in file_:
                        if float(splitLine[1]) >= timeFromStep - tau_spike and float(splitLine[1]) <= float(t_trialStep):
                            # spikes["senders"].append(int(splitLine[0])-1) # correct if neurons are the first to be created and nest ids starts at 1
                            spikes["senders"].append(json_convert[splitLine[0]])
                            spikes["times"].append(float(splitLine[1]))
                        elif float(splitLine[1]) > float(t_trialStep):
                            break;

    spikes['senders'] = np.array(spikes['senders'])
    spikes['times'] = np.array(spikes['times'])

    if len(spikes['senders']) <=0:
        print("No spikes for the interval selected.")
    else:
        for pst in ["region", "mtype", "etype"]:
            print pst
            sender_colors = []
            if pst=="region":
                sender_colors = Larea2clr[ Larea[ spikes["senders"] ] ]
            elif pst=="mtype" : 
                sender_colors = cellTypeToColor[0][ mtype[ spikes["senders"] ],: ] # mtype
            elif pst=="etype" : 
                sender_colors = cellTypeToColor[1][ etype[ spikes["senders"] ],: ] # etype

            print "Creating spatial rasters"
            if not exists(OUTPUT_FOLDER+"/pos_axial_rasters/"+pst):
                makedirs(OUTPUT_FOLDER+"/pos_axial_rasters/"+pst)
                makedirs(OUTPUT_FOLDER+"/pos_coronal_rasters/"+pst)
                makedirs(OUTPUT_FOLDER+"/pos_sagittal_rasters/"+pst)

            for it_ in range(int(timeFromStep), int(t_trialStep), timeInterval):
                spikes_in_dT = np.where( (spikes["times"]<float(it_))*(spikes["times"]>float(it_)-2.0*tau_spike) )[0]
                if spikes_in_dT.shape[0]>0:
                    sendersTMP = spikes["senders"][spikes_in_dT]
                    alphasTMP  = np.exp( (spikes["times"][spikes_in_dT] - float(it_)) / tau_spike )
                    clrsTMP    = np.zeros((spikes_in_dT.shape[0],4))
                    clrsTMP[:,:3] = np.array(sender_colors)[ spikes_in_dT,: ]
                    clrsTMP[:, 3] = alphasTMP
                    print("Spikes in "+ str(it_), len(sendersTMP))
                
                extent_ = [-528*25.0*0.5,528*25.0*0.5,-528*25.0*0.5,528*25.0*0.5]
                figure( figsize=(8,8) )
                if spikes_in_dT.shape[0]>0: scatter( -zs[ sendersTMP ], -xs[ sendersTMP ], marker=".", s=20, edgecolor="none", c = clrsTMP )
                imshow( 255-20*mask_axial, extent=extent_, cmap="gray", vmin=0, vmax=255 )
                axis(extent_); axis('off')
                savefig(OUTPUT_FOLDER+"/pos_axial_rasters/"+pst+"/pos_"+str(it_)+"ms.png")
                plt.gcf()

                figure( figsize=(8,8) )
                if spikes_in_dT.shape[0]>0: scatter( -zs[ sendersTMP ],  ys[ sendersTMP ], marker=".", s=20, edgecolor="none", c = clrsTMP )
                imshow( 255-20*mask_coronal, extent=extent_, cmap="gray", vmin=0, vmax=255 )
                axis(extent_); axis('off')
                savefig(OUTPUT_FOLDER+"/pos_coronal_rasters/"+pst+"/pos_"+str(it_)+"ms.png")
                plt.gcf()
                                
                figure( figsize=(8,8) )
                if spikes_in_dT.shape[0]>0: scatter(  xs[ sendersTMP ],  ys[ sendersTMP ], marker=".", s=20, edgecolor="none", c = clrsTMP )
                imshow( 255-20*mask_sagittal, extent=extent_, cmap="gray", vmin=0, vmax=255 )
                axis(extent_); axis('off')
                savefig(OUTPUT_FOLDER+"/pos_sagittal_rasters/"+pst+"/pos_"+str(it_)+"ms.png")
                close('all')
                plt.gcf()
