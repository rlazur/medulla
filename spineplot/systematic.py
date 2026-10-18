import numpy as np
import pandas as pd
import uproot

from variable import Variable

class Systematic:
    """
    A class designed to encapsulate the systematic uncertainty on a Sample
    due to a single source of uncertainty. A few types of systematic
    uncertainties are supported:
    - Statistical uncertainty: The uncertainty due to the finite size of
        the sample. This is calculated using a simple sqrt(N)/N formula.
    - Multisim uncertainty: The uncertainty due to the variation of
        parameters in the simulation. This is calculated by generating
        a set of universes and calculating the covariance matrix for
        the systematic parameter. The weights for the universes are
        stored in a TTree by code external to this tool.
    - Multisigma uncertainty: The uncertainty due to the variation of
        parameters in the simulation. This is calculated by generating
        a set of universes and calculating the covariance matrix for
        the systematic parameter. The weights are stored in a TTree by
        code external to this tool as "n-sigma" weights. This class
        interpolates the weights at the desired sigma level to generate
        a set of universe weights.

    The Systematic object is designed to be used in conjunction with the
    Sample object. The Sample object contains the data for the analysis
    and critically the exposure information. All systematic
    uncertainties must be rescaled to the target exposure before being
    used in an analysis. This class provides a method to rescale the
    covariance matrices to ensure that the systematic uncertainties are
    properly treated.

    Attributes
    ----------
    _name : str
        The name of the systematic uncertainty.
    _label : str
        The label of the systematic uncertainty. Intended to be used in
        plots for identifying the systematic parameter.
    _handle : uproot.models.TBranch.Model_TBranchElement
        The handle to the branch containing the weights for the
        systematic parameter.
    _variables : dict
        The Variable objects to be used for the calculation of the
        impact of the systematic uncertainty. The keys are the names of
        the variables and the values are the Variable objects
        themselves.
    _universe_weights : numpy.ndarray
        An array of weights for the systematic uncertainty. This has
        shape (nevents, nuniv) where nevents is the number of events in
        the sample and nuniv is the number of universes generated. This
        is not meant to be kept around for long periods of time due to
        memory concerns and is cleared internally by the `clear`
        method.
    _covariances : dict
        A dictionary containing the covariance matrices for each of the
        variables. The keys are the names of the variables and the values
        are numpy.ndarray objects representing the covariance matrices.
    _std : float
        The one-bin fractional uncertainty for the systematic parameter.
        This intuitively can be thought of as the uncertainty on a simple
        count of selected candidates due to the influence of the
        systematic parameter.

    Methods
    -------
    register_variable(variable)
        Register a Variable object with the Systematic object.
    process(sample, nuniv=1000)
        Processes the systematic uncertainty for the given sample for
        all configured Variables.
    get_covariance(variable)
        Retrieve the covariance matrix for the given variable.
    set_weight(weight)
        Rescales the covariance matrices using the given weight.
    combine(systematics, name, label)
        Combine a list of Systematic objects into a single Systematic
        object.
    """
    def __init__(self, name, handle, label=None):
        """
        Initializes the Systematic object with the given name and Variable.

        Parameters
        ----------
        name : str
            The name of the systematic uncertainty.
        handle : uproot.models.TBranch.Model_TBranchElement
            The handle to the branch containing the weights for the
            systematic parameter.
        """
        self._name = name
        self._label = label
        self._handle = handle
        self._variables = dict()

    def register_variable(self, variable):
        """
        Register a Variable object with the Systematic object.

        Parameters
        ----------
        variable : Variable
            The Variable object to register.
        """
        self._variables[variable._key] = variable

    def process(self, sample, mask, nuniv=1000, step=1000) -> np.ndarray:
        """
        Processes the systematic uncertainty for the given sample for
        all configured Variables.

        Parameters
        ----------
        sample : Sample
            The parent Sample object containing the dataset.
        mask : pd.Series
            A mask to apply to the sample data. This reflects the
            `presel` mask applied to the sample.
        nuniv : int, optional
            The number of universes to generate. The default is 1000.
        step : int, optional
            The number of events whose weights are read and binned at
            a time. The default is 1000.

        Returns
        -------
        numpy.ndarray
            An array of weights for the systematic uncertainty.
        """
        # Check that the handle is valid. This is the "usual" case
        # where the systematic weights are stored in a TTree.
        if self._handle is not None:
            # The bin of each selected event for each of the Variables.
            # Each Variable has an associated binning and field name,
            # which are used to bin the events for each set of universe
            # weights.
            bins = dict()
            for name, variable in self._variables.items():
                bin_edges = list(variable._bin_edges.values())[0]
                bin_indices = np.digitize(sample._data[name].to_numpy(), bin_edges) - 1
                valid_indices = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
                bins[name] = (bin_indices, valid_indices, len(bin_edges) - 1)

            # The weights are read from the TTree `step` events at a
            # time and binned chunk by chunk, so memory use does not
            # grow with the size of the sample: the weights of a full
            # 1e20 POT SBND sample are ~19 GB, more than a gpvm has.
            mask = np.asarray(mask, dtype=bool)
            histograms = dict()
            interpolation = None
            row = 0
            for start in range(0, len(mask), step):
                chunk_mask = mask[start:start + step]
                # array_cache=None: uproot's array cache sizes a jagged
                # branch read with library='np' by its object array of
                # per-event pointers, not the weights they hold, so its
                # 100 MB limit never triggers and every chunk would stay
                # cached (several GB over a full sample).
                weights_array = np.stack(self._handle.array(
                    entry_start=start, entry_stop=start + len(chunk_mask), library='np',
                    array_cache=None,
                ))[chunk_mask, :]

                # Some assumptions are made here about the shape and
                # structure of the weights array. If the shape indicates
                # that these are "n-sigma" weights, the appropriate
                # interpolation is performed to generate the universe
                # weights. Otherwise, the weights array is assumed to be
                # the universe weights directly. The "general" GENIE
                # multisigma weights have shape (N,6) corresponding to
                # -1, +1, -2, +2, -3, +3 sigma variations, whereas we tend
                # to store detector systematic multisigma weights with
                # shape (N,7) corresponding to -3, -2, -1, 0, +1, +2, +3
                # sigma variations.
                if weights_array.shape[1] in (6, 7):
                    if interpolation is None:
                        # Case for GENIE multisigma weights
                        if weights_array.shape[1] == 6:
                            sigma_levels_raw = np.array(
                                [-1, 1, -2, 2, -3, 3],
                                dtype=float
                            )
                            order = np.argsort(sigma_levels_raw)
                            sigma_levels = sigma_levels_raw[order]
                        # Case for detector multisigma weights
                        else:
                            order = np.arange(7)
                            sigma_levels = np.linspace(-3, 3, 7)

                        # A set of `nuniv` random values is drawn from a
                        # normal distribution with mean 0 and standard
                        # deviation 1, shared by every event. The weights
                        # are interpolated at these values to generate
                        # the universe weights. Interpolation is linear
                        # in the weights, so it is a fixed matrix
                        # (sigma levels x universes) applied to every
                        # event: row k is the interpolation of a weight
                        # vector that is 1 at level k and 0 elsewhere.
                        random_sigmas = np.random.normal(0, 1, nuniv)
                        interpolation = np.array([
                            np.interp(random_sigmas, sigma_levels, level)
                            for level in np.eye(len(sigma_levels))
                        ])
                    self._universe_weights = np.asarray(weights_array, dtype=float)[:, order] @ interpolation
                else:
                    self._universe_weights = weights_array

                # Universes
                for name, (bin_indices, valid_indices, nbins) in bins.items():
                    rows = slice(row, row + len(self._universe_weights))
                    valid = valid_indices[rows]
                    histogram = histograms.setdefault(name, np.zeros((nbins, self._universe_weights.shape[1])))
                    np.add.at(histogram, bin_indices[rows][valid], self._universe_weights[valid, :])
                row += len(self._universe_weights)
            n_universes = self._universe_weights.shape[1]

            # The universe histograms are used to characterize the
            # systematic uncertainty for each of the Variables by
            # method of covariance matrix.
            self._covariances = dict()
            for name, (bin_indices, valid_indices, nbins) in bins.items():
                histogram = histograms[name]

                # Central value
                cv_histogram = np.zeros(nbins)
                np.add.at(cv_histogram, bin_indices[valid_indices], 1)

                # Covariance matrix calculated with respect to the central
                # value.
                diff = histogram - cv_histogram[:, np.newaxis]
                self._covariances[f'{self._name}_{name}'] = (diff @ diff.T) / n_universes

                # One-bin uncertainty
                diff = np.sum(diff, axis=0)
                self._std = np.sqrt((diff @ diff.T) / n_universes)
                self._std /= row

            self._universe_weights = None
        # The handle is None, which is taken to be the case where we
        # calculate the statistical uncertainty.
        else:
            self._covariances = dict()
            for name, variable in self._variables.items():
                data = sample._data[name].to_numpy()
                bin_edges = list(variable._bin_edges.values())[0]
                bin_indices = np.digitize(data, bin_edges) - 1
                valid_indices = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
                bin_indices = bin_indices[valid_indices]
                
                histogram = np.zeros(len(bin_edges) - 1)
                np.add.at(histogram, bin_indices, 1)

                self._covariances[f'{self._name}_{name}'] = np.diag(histogram)
                self._std = np.sqrt(histogram.sum())
                self._std /= histogram.sum()

    def get_covariance(self, variable) -> np.ndarray:
        """
        Retrieve the covariance matrix for the given variable.

        Parameters
        ----------
        variable : str
            The name of the variable to retrieve the covariance matrix
            for.

        Returns
        -------
        numpy.ndarray
            The covariance matrix for the given variable.
        """
        return self._covariances[f'{self._name}_{variable}']

    def set_weight(self, weight):
        """
        Rescales the covariance matrices using the given weight. This
        is necessary for cases where the analysis is comprised of an
        ensemble of samples with different exposures. The proper
        treatment in this case is to "divide out" the exposure carried
        by the parent Sample, then rescale the covariance matrices by
        the exposure of the ordinate sample. This ratio is exactly the
        `weight` parameter. Because these are covariance matrices, the
        applied rescaling is the square of the weight.

        Parameters
        ----------
        weight : float
            The weight for the systematic uncertainty.

        Returns
        -------
        None.
        """
        for kvar, vvar in self._variables.items():
            self._covariances[f'{self._name}_{kvar}'] *= weight**2

    @staticmethod
    def combine(systematics, name, label) -> 'Systematic':
        """
        Combine a list of Systematic objects into a single Systematic
        object. This is done by adding the covariance matrices of the
        Systematic objects. The underlying assumption is that the
        systematic parameters encapsulated by the Systematic objects are
        totally uncorrelated.

        Parameters
        ----------
        systematics : list
            A list of Systematic objects to combine.
        name : str
            The name of the new Systematic object.
        label : str
            The label of the new Systematic object.

        Returns
        -------
        Systematic
            A new Systematic object with the covariance matrices added.
        """
        new_systematic = Systematic(name, None, label)
        new_systematic._variables = systematics[0]._variables
        new_systematic._covariances = dict()
        for kvar, vvar in new_systematic._variables.items():
            if any([kvar not in sys._variables.keys() for sys in systematics]):
                msg = f'Variable {kvar} not found in all Systematic objects.'
                raise ValueError(msg)
            new_systematic._covariances[f'{name}_{kvar}'] = np.sum(
                [sys._covariances[f'{sys._name}_{kvar}'] for sys in systematics],
                axis=0
            )
            new_systematic._std = np.sqrt(np.sum([sys._std**2 for sys in systematics]))
        
        return new_systematic

    @staticmethod
    def transform_as(cov, param):
        """
        Apply a scale correction to the covariance matrix. This is
        intended to be used when the covariance matrix is meant to be
        rescaled to a different exposure, or for the case where the
        spectrum is being area normalized. The former case is realized
        by a "simple" scale correction (i.e. the covariance matrix is
        multiplied by the square of the scale). The latter case is
        realized by a "normalized" scale correction (i.e. the
        covariance matrix is properly transformed to account for how
        the area normalization impacts the covariance matrix).

        Parameters
        ----------
        cov : numpy.ndarray
            The covariance matrix to transform.
        param : float or np.ndarray
            The parameter to use for the transformation. For a "simple"
            scale correction, this is a float. For a "normalized" scale
            correction, this is a numpy array with the same shape as
            `cov` and containing the bin contents of the histogram.
        
        Returns
        -------
        numpy.ndarray
            The transformed covariance matrix.
        """
        if np.isscalar(param):
            return cov * param**2

        else:
            A = np.sum(param)
            y = param.reshape(-1, 1)
            delta = np.eye(len(y))
            jacobian = (delta * A - y) / A**2
            return jacobian @ cov @ jacobian.T

    def __repr__(self):
        s = f'--Systematic({self._name}, {self._label})--'
        s += f'\n\tFractional uncertainty: {self._std:.2%}'
        return s

    @property
    def name(self):
        return self._name
    
    @property
    def label(self):
        return self._label

    @property
    def std(self):
        return self._std

class DetectorSystematic(Systematic):
    """
    A Systematic object for a detector variation. Unlike the weight-
    based systematics, a detector variation is not a set of per-event
    weights: it is a separate simulated sample in which the same events
    are re-simulated with one aspect of the detector model changed.
    The uncertainty is therefore built by comparing the spectrum of
    each variation sample to that of a dedicated detector variation
    central value (CV) sample.

    The detector variation samples are generally produced separately
    from the nominal sample (different statistics, possibly different
    reconstruction version), so the comparison is done in terms of
    fractional shifts. For each variation `v` and bin `i`, the shift
    is (N_v,i / POT_v) / (N_cv,i / POT_cv) - 1, which is applied to the
    nominal spectrum of the parent Sample. Bins that are empty in the
    detector variation CV have no shift. A single variation is treated
    as a one-sided unisim (covariance = d d^T). Several variations of
    the same parameter (e.g. +/- 5% gain) are treated as universes of
    that parameter, and their covariance is averaged, as is done for
    the multisim weights.

    Attributes
    ----------
    _cv : tuple(pd.DataFrame, float)
        The selected events of the detector variation CV sample and its
        exposure in POT.
    _variations : list(tuple(pd.DataFrame, float))
        The selected events and exposure (POT) for each variation
        sample of this detector parameter.
    """
    def __init__(self, name, cv, variations, label=None):
        """
        Initializes the DetectorSystematic object.

        Parameters
        ----------
        name : str
            The name of the systematic uncertainty.
        cv : tuple(pd.DataFrame, float)
            The selected events of the detector variation CV sample and
            its exposure in POT.
        variations : list(tuple(pd.DataFrame, float))
            The selected events and exposure (POT) for each variation
            sample of this detector parameter.
        label : str, optional
            The label of the systematic uncertainty. The default is
            None.
        """
        super().__init__(name, None, label)
        self._cv = cv
        self._variations = variations

    def process(self, sample, mask, nuniv=None) -> None:
        """
        Processes the detector systematic uncertainty for the given
        sample for all configured Variables. The covariance matrix is
        expressed in the (unweighted) counts of the parent Sample, as
        for the other Systematic types, so that it is rescaled to the
        target exposure by `set_weight` in the same way.

        Parameters
        ----------
        sample : Sample
            The parent Sample object containing the nominal dataset.
        mask : pd.Series
            Unused. The `presel` mask has already been applied to the
            nominal, CV, and variation datasets.
        nuniv : int, optional
            Unused. The number of universes is the number of variation
            samples.

        Returns
        -------
        None.
        """
        self._covariances = dict()
        for name, variable in self._variables.items():
            for df, _ in [self._cv, *self._variations]:
                if name not in df.columns:
                    raise ValueError(f'Variable `{name}` not found in the detector variation samples for `{self._name}`.')
            bin_edges = list(variable._bin_edges.values())[0]
            nominal = DetectorSystematic.histogram(sample._data[name], bin_edges)
            cv = DetectorSystematic.histogram(self._cv[0][name], bin_edges) / self._cv[1]

            # Fractional shift of each variation relative to the
            # detector variation CV, with shape (nbins, nvariations).
            # This is applied to the nominal spectrum to give the
            # shift in the counts of the parent Sample.
            shifts = np.stack([DetectorSystematic.histogram(df[name], bin_edges) / pot for df, pot in self._variations], axis=1)
            ratio = np.divide(shifts, cv[:, np.newaxis], out=np.ones_like(shifts), where=cv[:, np.newaxis] > 0)
            diff = (ratio - 1) * nominal[:, np.newaxis]
            self._covariances[f'{self._name}_{name}'] = (diff @ diff.T) / diff.shape[1]

            # One-bin uncertainty
            diff = np.sum(diff, axis=0)
            self._std = np.sqrt((diff @ diff.T) / len(diff))
            self._std /= np.sum(nominal)

    @staticmethod
    def histogram(data, bin_edges) -> np.ndarray:
        """
        Histograms the data with the same convention as the other
        Systematic types (np.digitize, entries outside the binning are
        dropped).

        Parameters
        ----------
        data : pd.Series
            The data to histogram.
        bin_edges : numpy.ndarray
            The bin edges of the histogram.

        Returns
        -------
        numpy.ndarray
            The bin contents of the histogram.
        """
        bin_indices = np.digitize(data.to_numpy(), bin_edges) - 1
        valid_indices = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
        histogram = np.zeros(len(bin_edges) - 1)
        np.add.at(histogram, bin_indices[valid_indices], 1)
        return histogram
