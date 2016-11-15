# STORN implementation for keras

The implementation can be found in greenarm/models/STORN.py!

### Required packages
 - keras >= 1.0.6
 - theano >= 0.8.2
 - numpy >= 1.11.0
 - scipy >= 0.17.1
 - h5py = 2.6.0
 - hdf5 = 1.8.16
 - [hualos](https://github.com/fchollet/hualos) (optional monitoring)

## Please replace this code snippet in the `keras` library!
Keras does not currently have proper support for Masking and merged layer outputs.

Changes for Masked Layer Merge building on top of keras 1.0.6
file: keras/engine/topology.py, method: compute_mask, line: 1349

 ```{py}
    def compute_mask(self, inputs, mask=None):
        if mask is None or all([m is None for m in mask]):
            return None

        assert hasattr(mask, '__len__') and len(mask) == len(inputs)

        if self.mode in ['sum', 'mul', 'ave']:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)
        elif self.mode == 'concat':
             # Make a list of masks while making sure the dimensionality of each mask
             # is the same as the corresponding input.
             masks = []
             for input_i, mask_i in zip(inputs, mask):
                 if mask_i is None:
                     # Input is unmasked. Append all 1s to masks
                     masks.append(K.ones_like(input_i))
                 elif K.ndim(mask_i) < K.ndim(input_i):
                     # Mask is smaller than the input, expand it
                     masks.append(K.expand_dims(mask_i))
                 else:
                     masks.append(mask)
             concatenated = K.concatenate(masks, axis=self.concat_axis)
             return K.all(concatenated, axis=-1, keepdims=False)
        elif self.mode in ['cos', 'dot']:
            return None
        elif hasattr(self.mode, '__call__'):
            if hasattr(self._output_mask, '__call__'):
                return self._output_mask(mask)
            else:
                return self._output_mask
        else:
            # this should have been caught earlier
            raise Exception('Invalid merge mode: {}'.format(self.mode))
 ```
