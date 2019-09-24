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
PREFIX = "neuron_"
NEST_OUTPUT = join(OUTPUT_FOLDER, PREFIX + "output")
convert = True if PREFIX == "nest_" else False
HIGHD_FOLDER = ROOT_FOLDER
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


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return np.array([float(int(value[i:i + lv // 3], 16)) for i in range(0, lv, lv // 3)])


t_trial = 250
dt = 0.1
timeInterval = 1
timeFrom = 200.0
file_ = join(HIGHD_FOLDER, 'ptneu_brain_' + Dver + '_Nest.h5')
seed = 75599
stim = 2000.0
rasterSize = 20

doRasters = True
doFrHistograms = True
do_ISI = True

# np.random.seed(seed)
# standalone_input = stim * np.fromfunction(lambda x: (x>=200) & (x % 200 <= 20), (int(t_trial / dt),))

databases_file = HIGHD_FOLDER + "/elecDB.db"
specie = 'mouse'
excluded_regions = ["spinal cord", "spinal white matter"] # excluded regions of the database for the whole brain generation
included_regions = [] # Specific regions of the database targeted
regions = ["Primary somatosensory area, lower limb", 'Ventral posteromedial nucleus of the thalamus']
regions_labels = ['SSCtx:HL', 'VPN'] 
conn = sqlite3.connect(databases_file)
cursor = conn.cursor()
get_regions(cursor, excluded_regions, included_regions, specie)
maxRegionID = 0
for jn in region_dictionary_to_id_ALLNAME.keys():
    if region_dictionary_to_id_ALLNAME[jn]>maxRegionID:
        maxRegionID = region_dictionary_to_id_ALLNAME[jn]
request = "select id from regions where is_leaf=1 AND (full_name like '%"+("%' OR full_name like '%").join(regions) + "%')"
cursor.execute(request)
regions_ids = np.array(cursor.fetchall())[:,0]
rev_regions_ids = {}
for id_, regions_id in enumerate(regions_ids):
    rev_regions_ids[regions_id] = id_

print "Loading h5 file"
h5file = h5py.File(file_, "r")
Larea   = h5file['neurons']["Larea"][:]
etype   = h5file['neurons']["eTypes"][:]
request = "select name from etypes"
cursor.execute(request)
ETTN    = np.array(cursor.fetchall())[:,0]
mtype   = h5file['neurons']["cellTypes"][:]
request = "select name from mtypes"
cursor.execute(request)
CTTN    = np.array(cursor.fetchall())[:,0]
Umtype = np.unique(mtype)
Uetype = np.unique(etype)
h5file.close()
conn.close()

print("Loading spikes") 
json_cont = json.loads(open(OUTPUT_FOLDER + "/conversion.json", "r").read())
json_convert = {}
for k, v in json_cont.items():
    json_convert[str(v)] = int(k)
del json_cont

spikes = {"senders":[], "times":[], "regions":[]}
for f in listdir(NEST_OUTPUT):
# f = listdir(NEST_OUTPUT)[0]
    print(f)
    file_ = join(NEST_OUTPUT, f)
    if isfile(file_):
        with open(file_, 'r') as f:
            for line in f:
                splitLine = line.split()
                if "spike_detector-" in file_:
                    if convert:
                        gid = json_convert[splitLine[0]]
                    else:
                        gid = int(splitLine[0])
                    larea = Larea[gid]
                    if float(splitLine[1]) >= timeFrom and float(splitLine[1]) <= float(t_trial) and larea in regions_ids:
                        # spikes["senders"].append(int(splitLine[0])-1) # correct if neurons are the first to be created and nest ids starts at 1
                        spikes["senders"].append(gid)
                        spikes["times"].append(float(splitLine[1]))
                        spikes["regions"].append(rev_regions_ids[larea])
                    elif float(splitLine[1]) > float(t_trial):
                        break;

spikes['senders'] = np.array(spikes['senders'])
spikes['times'] = np.array(spikes['times'])
spikes['regions'] = np.array(spikes['regions'])
if len(spikes['senders']) <=0:
    print("No spikes for the interval selected.")
else:
    (IDsUnique, Indices, Counts_) = np.unique(spikes["senders"], return_index=True, return_counts=True)
    Counts_ = np.float64(Counts_) / (1e-3*float(t_trial-timeFrom)) # to get firing rates in Hz


    print("Loading colors")
    cellTypeToColor = [ 0.5*np.ones((len(CTTN)+1,3)), 0.5*np.ones((len(ETTN)+1,3)) ] # last id is grey for unknown
    for itypeArray,typeArray in enumerate([CTTN, ETTN]):
        for iTTN,TTN_ in enumerate(typeArray):
            np.random.seed( int(np.sum(np.array([ord(ch) for ch in TTN_]))) + 1 )
            clrRAND = np.random.rand(3)
            if np.sum(clrRAND)<1.5: clrRAND *= 1.5/np.sum(clrRAND)
            clrRAND[clrRAND>1.0] = 1.0
            cellTypeToColor[itypeArray][iTTN,:] = clrRAND[:]
    del clrRAND

    Larea2clr = np.zeros((regions_ids.size, 3), np.float32)
    Larea_STR = ['']*regions_ids.size
    Larea_name = {}
    for ikk,kk in enumerate(regions_ids):
        color = hex_to_rgb(id_to_color[kk])
        Larea2clr[ikk] = color /255.
        Larea_STR[ikk] = id_to_region_dictionary_ALLNAME[kk]
        Larea_name[ikk] = id_to_region_dictionary[kk]
        for i in range(len(regions)):
            if regions[i] in Larea_name[ikk]:
                Larea_name[ikk] = Larea_name[ikk].replace(regions[i], regions_labels[i])
    Larea_STR = np.array(Larea_STR, dtype=np.str)
    makeLegendFig( OUTPUT_FOLDER+"/legend_mtype.png", labels=CTTN, colors= cellTypeToColor[ 0 ], numColumns=3 )
    makeLegendFig( OUTPUT_FOLDER+"/legend_etype.png", labels=ETTN, colors= cellTypeToColor[ 1 ] )
    for pst in ["region", "mtype", "etype"]:
        print pst
        sender_colors = []
        sortarrN = []
        if pst=="region": 
            sortarrN = np.argsort(np.argsort(Larea_STR))[spikes["regions"]] # double argsort creates an array that we can send neuron ids to
            sender_colors = Larea2clr[spikes["regions"]]
        elif pst=="mtype" : 
            sortarrN = np.argsort(np.argsort(mtype))[spikes["senders"]]
            sender_colors = cellTypeToColor[0][ mtype[ spikes["senders"] ],: ] # mtype
        elif pst=="etype" : 
            sortarrN = np.argsort(np.argsort(etype))[spikes["senders"]]
            sender_colors = cellTypeToColor[1][ etype[ spikes["senders"] ],: ] # etype


        if doRasters:
            print "Creating rasters"
            if not exists(OUTPUT_FOLDER+"/rasters"):
                makedirs(OUTPUT_FOLDER+"/rasters")
            for isSorted in [0,1]:
                figure( figsize=(14,12) )
                if isSorted: 
                    scatter(spikes["times"]-timeFrom, sortarrN, marker=".", s=rasterSize, edgecolor="none", c=sender_colors)
                else:
                    scatter(spikes["times"]-timeFrom, spikes["senders"]          , marker=".", s=rasterSize, edgecolor="none", c=sender_colors)
                xlabel("t [ms]"); ylabel("Neuron ID");
                xlim([0.0,t_trial-timeFrom]); 
                # ylim([np.min(spikes["senders"]), np.max(spikes["senders"])]); 
                tight_layout()
                savefig(OUTPUT_FOLDER+"/rasters/"+PREFIX+("sorted_"*bool(isSorted))+"raster_"+pst+"_"+str(int(timeFrom))+"-"+str(int(t_trial))+".png")
                close('all')
                plt.gcf()

        if doFrHistograms:
            print "Creating firing rate histograms"
            if not exists(OUTPUT_FOLDER+"/histograms"):
                makedirs(OUTPUT_FOLDER+"/histograms")

                # adding neurons with zero firing rate: (usually not a good idea)
                #~ diffsInISs = np.setdiff1d( np.array(range(Larea.shape[0])), IDsUnique, assume_unique=True )
                #~ IDsUnique = np.append( IDsUnique, diffsInISs )
                #~ Counts_   = np.append( Counts_, np.array([0.0]*len(diffsInISs)) )
                #~ print IDsUnique.shape
                #~ print Counts_.shape
                
            if pst=="region":
                popsUnique = np.arange(regions_ids.size) 
                IDsUniquePoptype = spikes["regions"][Indices]
                clrsTMP = Larea2clr 
                label_dict = Larea_name 
            if pst=="mtype":
                popsUnique = Umtype
                IDsUniquePoptype = mtype[IDsUnique]
                clrsTMP = cellTypeToColor[0][ :,: ]
                label_dict = CTTN
            if pst=="etype":
                popsUnique = Uetype
                IDsUniquePoptype = etype[IDsUnique]
                clrsTMP = cellTypeToColor[1][ :,: ]
                label_dict = ETTN
            curr_i = 0
            for pop in popsUnique:
                IDsLocal = np.where(IDsUniquePoptype==pop)[0]
                if len(IDsLocal)>0: curr_i +=1 
            f, axarr = subplots(curr_i, sharex=True, figsize=(12,14))
            curr_i = 0
            for pop in popsUnique:
                IDsLocal = np.where(IDsUniquePoptype==pop)[0]
                FiringRatesLocal = Counts_[ IDsLocal ]
                numBins = np.unique(FiringRatesLocal).shape[0]
                if numBins>0:
                    axarr[curr_i].hist(FiringRatesLocal , edgecolor="none", bins=numBins, density=True, color=[clrsTMP[pop,0],clrsTMP[pop,1],clrsTMP[pop,2],1.0])
                    #~ axarr[curr_i].set(ylabel=ETTN[pop])
                    axarr[curr_i].set_ylabel(label_dict[pop], rotation=45, ha="right")
                    #~ h = plt.ylabel('y')
                    #~ h.set_rotation(0)
                    axarr[curr_i].set_yticks([])
                    axarr[curr_i].set_xlim([0.0,np.max(Counts_)])
                    curr_i +=1
            f.subplots_adjust(hspace=0)
            xlabel("Firing rates [Hz]")
            savefig(OUTPUT_FOLDER+"/histograms/"+PREFIX+"histFr_"+pst+".png")
            close('all')
            plt.gcf()
        if do_ISI:
            print "Creating Inter-spike intervals histograms"
            if not exists(OUTPUT_FOLDER+"/ISIs"):
                makedirs(OUTPUT_FOLDER+"/ISIs")
            if pst=="region":
                popsUnique = np.arange(regions_ids.size) 
                IDsUniquePoptype = spikes["regions"][Indices]
                clrsTMP = Larea2clr 
                label_dict = Larea_name
            if pst=="mtype":
                popsUnique = Umtype
                IDsUniquePoptype = mtype[IDsUnique]
                clrsTMP = cellTypeToColor[0][ :,: ]
                label_dict = CTTN
            if pst=="etype":
                popsUnique = Uetype
                IDsUniquePoptype = etype[IDsUnique]
                clrsTMP = cellTypeToColor[1][ :,: ]
                label_dict = ETTN
            curr_i = 0
            for pop in popsUnique:
                IDsLocal = np.where(IDsUniquePoptype==pop)[0]
                if len(IDsLocal)>0: curr_i +=1 
            f, axarr = subplots(curr_i, sharex=True, figsize=(12,14))
            curr_i = 0
            for pop in popsUnique:
                IDsLocal = np.where(IDsUniquePoptype==pop)[0]
                if len(IDsLocal)>0:
                    ISI = []
                    for id_ in IDsLocal:
                        timeLocal = spikes['times'][np.where(spikes['senders']==IDsUnique[id_])]
                        ISI.extend(timeLocal[1:] - timeLocal[:-1])
                    axarr[curr_i].hist(ISI , edgecolor="none", density=True, color=[clrsTMP[pop,0],clrsTMP[pop,1],clrsTMP[pop,2],1.0])
                    #~ axarr[curr_i].set(ylabel=ETTN[pop])
                    axarr[curr_i].set_ylabel(label_dict[pop], rotation=45, ha="right")
                    #~ h = plt.ylabel('y')
                    #~ h.set_rotation(0)
                    axarr[curr_i].set_yticks([])
                    # axarr[curr_i].set_xlim([0.0,np.max(ISI)])
                    curr_i +=1
            f.subplots_adjust(hspace=0)
            xlabel("Inter spikes intervals [ms]")
            savefig(OUTPUT_FOLDER+"/ISIs/"+PREFIX+"isi_"+pst+".png")
            close('all')
            plt.gcf()