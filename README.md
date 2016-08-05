### Baxter Collision Data - Group Greenarm

Required packages:
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

### Running all experiments
* Place the 3 folders `anomal`, `normal1` and `normal2` directly under `data` to let our model pickup the Baxter collision dataset.
* In the `main.py` file, there are a number of callable functions (commented out in the end) which are there for running different experiments:
    * `run_TS_evaluator()` will run the simple Timeseries Predictor, producing plots of the fit and the loss value in the `plots` directory.
    * `run_STORN_evaluator(use_anomalies=True)` will do the same for STORN. Set `use_anomalies` to `False` to run STORN on normal data, `True` makes it run on anomalous data. Both calls will print how many sequences were detected as anomalous.
    * `max_detection_STORN()` will run the Max Logistic Regression anomaly detection classifier on the STORN loss. Prints a confusion matrix.
    * `max_detection_tsp()` does the same for the Timeseries Predictor.
    * `conv_detection_storn()` will run the ConvNet anomaly detection classifier on the STORN loss. Also prints a confusion matrix.
    * `create_coarse_ROC_plot()` will create a ROC diagram for the above 3 experiments. The diagram is saved under `plots`.
    * The two `run_fine_grained_evaluation(...)` calls run the fine-grained detection experiments with the two heuristic intervals as described in our report.

Please be patient when running the functions in `main.py`, as the STORN model is always recompiled in between calls.
Currently the functions in main use already trained model weights that we have saved for STORN and the Timeseries predictor. If you want to train e.g. STORN again before running an experiment, just pass an argument `train=True` to any of the above functions. Keep in mind STORN needs very long time to train (hours).