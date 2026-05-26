**Project description**
*CI_Artifact Reduction*

Filtering CI Artifacts based on Correlation of Indepndent Components and audio 

*Method Overview*
1. EEG data is decomposed into independent components using ICA.
2. Each component is cross-correlated with the audio stimulus.
3. The peak correlation within an window (can be adjusted) is identified.
4. The SNR value is computed from the correlation peak.
5. Components with SNR values above a defined threshold are excluded.
6. The cleaned EEG signal is reconstructed from the remaining components.

```python
cleaned_eeg = ci_artifact_reduction(
    raw=raw,
    subject_id="S01",
    trial_id="T01",
    output_dir="./results",
    snr_threshold=6,
    fs_eeg=1000,
    attended_audio=attended_audio,
    plot=True,
    metadata=True
)
```

### Parameters

**raw** : mne.io.Raw  
&nbsp;&nbsp;&nbsp;&nbsp;EEG recording loaded with MNE-Python.

**subject_id** : str | int  
&nbsp;&nbsp;&nbsp;&nbsp;Identifier of the subject used for output file naming.

**trial_id** : str | int  
&nbsp;&nbsp;&nbsp;&nbsp;Identifier of the trial used for output file naming.

**output_dir** : str  
&nbsp;&nbsp;&nbsp;&nbsp;Directory where plots and metadata files are saved.

**snr_threshold** : float  
&nbsp;&nbsp;&nbsp;&nbsp;Independent components with SNR values above this threshold are removed.

**fs_eeg** : int  
&nbsp;&nbsp;&nbsp;&nbsp;Sampling frequency of the EEG recording in Hz.

**attended_audio** : np.ndarray  
&nbsp;&nbsp;&nbsp;&nbsp;1D array containing the attended audio signal.

**distraction_audio** : np.ndarray, optional  
&nbsp;&nbsp;&nbsp;&nbsp;1D array containing the distracting audio signal. Default is None.

**peak_win_negative** : float, optional  
&nbsp;&nbsp;&nbsp;&nbsp;Negative lag search window in seconds (default = 0.005).

**peak_win_pos** : float, optional  
&nbsp;&nbsp;&nbsp;&nbsp;Positive lag search window in seconds (default = 0.012).

**plot** : bool, optional  
&nbsp;&nbsp;&nbsp;&nbsp;If True, saves cross-correlation plots of all independent components (default = False).

**metadata** : bool, optional  
&nbsp;&nbsp;&nbsp;&nbsp;If True, saves metadata and summary statistics as CSV files (default = False).

---

### Returns

**cleaned_eeg** : np.ndarray  
&nbsp;&nbsp;&nbsp;&nbsp;EEG data after CI artifact removal.

---

### Raises

**ValueError**  
&nbsp;&nbsp;&nbsp;&nbsp;If `output_dir` is not specified.

**ValueError**  
&nbsp;&nbsp;&nbsp;&nbsp;If EEG and audio dimensions do not match.

---

### Warns

**UserWarning**  
&nbsp;&nbsp;&nbsp;&nbsp;If EEG sampling frequency is below 500 Hz.

---

```python
cleaned_eeg = ci_artifact_reduction(
    raw=raw,
    subject_id="S01",
    trial_id="T01",
    output_dir="./results",
    snr_threshold=6,
    fs_eeg=1000,
    attended_audio=attended_audio,
    plot=True,
    metadata=True
)
```

Installation
Install the latest release using pip:
```bash
pip install -i https://test.pypi.org/simple/ ci-artifact-reduction
```

<!-- pip install cors-ica -->