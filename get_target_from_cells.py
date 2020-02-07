import h5py
cells = ['a' + cell for cell in h5py.File('whole_brain_model_SONATA.h5', 'r')["/nodes/default/node_id"]]
print cells

