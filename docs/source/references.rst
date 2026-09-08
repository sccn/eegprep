.. _references:

==========
References
==========

Papers behind the methods EEGPrep implements. Each entry can be copied as APA or
BibTeX.

.. raw:: html

   <style>
   .eegprep-citation { margin-bottom: 1.1rem; }
   .eegprep-citation p { margin: 0.1rem 0; }
   .eegprep-cite-links { white-space: nowrap; margin-left: 0.15rem; }
   .eegprep-cite-copy {
     font: inherit; font-size: 0.82em; cursor: pointer;
     padding: 0.05rem 0.5rem; margin-left: 0.35rem;
     border: 1px solid currentColor; border-radius: 4px;
     background: transparent; color: inherit; opacity: 0.7;
     vertical-align: baseline; white-space: nowrap;
   }
   .eegprep-cite-copy:hover { opacity: 1; }
   </style>
   <script>
   document.addEventListener("click", function (event) {
     var button = event.target.closest(".eegprep-cite-copy");
     if (!button) { return; }
     var text = button.getAttribute("data-citation");
     var restore = button.textContent;
     var done = function () {
       button.textContent = "Copied";
       setTimeout(function () { button.textContent = restore; }, 1500);
     };
     if (navigator.clipboard && navigator.clipboard.writeText) {
       navigator.clipboard.writeText(text).then(done);
     } else {
       var area = document.createElement("textarea");
       area.value = text;
       document.body.appendChild(area);
       area.select();
       document.execCommand("copy");
       document.body.removeChild(area);
       done();
     }
   });
   </script>

.. raw:: html

   <div class="eegprep-citation">
     <p><strong>EEGPrep</strong></p>
     <p>
       Delorme, A., Ranganath, S., Kothe, C., Jaiswal, A., Makeig, S., &amp; Aristimunha, B. (2026). EEGPrep: a validated Python implementation of the EEGLAB preprocessing pipeline. arXiv:2607.16647.
       <span class="eegprep-cite-links"><a href="https://arxiv.org/abs/2607.16647">arXiv</a> | <a href="https://openalex.org/W7169837262">OpenAlex</a></span>
       <button class="eegprep-cite-copy" data-citation="Delorme, A., Ranganath, S., Kothe, C., Jaiswal, A., Makeig, S., &amp; Aristimunha, B. (2026). EEGPrep: a validated Python implementation of the EEGLAB preprocessing pipeline. arXiv:2607.16647.">Copy APA</button>
       <button class="eegprep-cite-copy" data-citation="@misc{delorme2026eegprep,&#10;  author = {Delorme, Arnaud and Ranganath, Suraj and Kothe, Christian and Jaiswal, Aman and Makeig, Scott and Aristimunha, Bruno},&#10;  title = {EEGPrep: a validated Python implementation of the EEGLAB preprocessing pipeline},&#10;  year = {2026},&#10;  eprint = {2607.16647},&#10;  archivePrefix = {arXiv},&#10;  doi = {10.48550/arXiv.2607.16647}&#10;}">Copy BibTeX</button>
     </p>
   </div>

.. raw:: html

   <div class="eegprep-citation">
     <p><strong>EEGLAB</strong></p>
     <p>
       Delorme, A., &amp; Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. Journal of Neuroscience Methods, 134(1), 9-21.
       <span class="eegprep-cite-links"><a href="https://pubmed.ncbi.nlm.nih.gov/15102499/">PubMed</a> | <a href="https://openalex.org/W2128495200">OpenAlex</a></span>
       <button class="eegprep-cite-copy" data-citation="Delorme, A., &amp; Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis. Journal of Neuroscience Methods, 134(1), 9-21.">Copy APA</button>
       <button class="eegprep-cite-copy" data-citation="@article{delorme2004eeglab,&#10;  author = {Delorme, Arnaud and Makeig, Scott},&#10;  title = {EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis},&#10;  journal = {Journal of Neuroscience Methods},&#10;  volume = {134},&#10;  number = {1},&#10;  pages = {9--21},&#10;  year = {2004},&#10;  doi = {10.1016/j.jneumeth.2003.10.009}&#10;}">Copy BibTeX</button>
     </p>
   </div>

.. raw:: html

   <div class="eegprep-citation">
     <p><strong>Artifact Subspace Reconstruction (ASR)</strong></p>
     <p>
       Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S., Jung, T.-P., &amp; Cauwenberghs, G. (2015). Real-time neuroimaging and cognitive monitoring using wearable dry EEG. IEEE Transactions on Biomedical Engineering, 62(11), 2553-2567.
       <span class="eegprep-cite-links"><a href="https://pubmed.ncbi.nlm.nih.gov/26415149/">PubMed</a> | <a href="https://openalex.org/W1936982107">OpenAlex</a></span>
       <button class="eegprep-cite-copy" data-citation="Mullen, T. R., Kothe, C. A. E., Chi, Y. M., Ojeda, A., Kerth, T., Makeig, S., Jung, T.-P., &amp; Cauwenberghs, G. (2015). Real-time neuroimaging and cognitive monitoring using wearable dry EEG. IEEE Transactions on Biomedical Engineering, 62(11), 2553-2567.">Copy APA</button>
       <button class="eegprep-cite-copy" data-citation="@article{mullen2015asr,&#10;  author = {Mullen, Tim R. and Kothe, Christian A. E. and Chi, Yu M. and Ojeda, Alejandro and Kerth, Trevor and Makeig, Scott and Jung, Tzyy-Ping and Cauwenberghs, Gert},&#10;  title = {Real-time neuroimaging and cognitive monitoring using wearable dry EEG},&#10;  journal = {IEEE Transactions on Biomedical Engineering},&#10;  volume = {62},&#10;  number = {11},&#10;  pages = {2553--2567},&#10;  year = {2015},&#10;  doi = {10.1109/TBME.2015.2481482}&#10;}">Copy BibTeX</button>
     </p>
   </div>

.. raw:: html

   <div class="eegprep-citation">
     <p><strong>ICLabel</strong></p>
     <p>
       Delorme, A., Truong, D., Pion-Tonachini, L., &amp; Makeig, S. (2024). Automatic EEG independent component classification using ICLabel in Python. In 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 4137-4141.
       <span class="eegprep-cite-links"><a href="https://ieeexplore.ieee.org/document/10822445/">IEEE Xplore</a> | <a href="https://openalex.org/W4406260175">OpenAlex</a></span>
       <button class="eegprep-cite-copy" data-citation="Delorme, A., Truong, D., Pion-Tonachini, L., &amp; Makeig, S. (2024). Automatic EEG independent component classification using ICLabel in Python. In 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 4137-4141.">Copy APA</button>
       <button class="eegprep-cite-copy" data-citation="@inproceedings{delorme2024iclabel,&#10;  author = {Delorme, Arnaud and Truong, Dung and Pion-Tonachini, Luca and Makeig, Scott},&#10;  title = {Automatic EEG Independent Component Classification Using ICLabel in Python},&#10;  booktitle = {2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},&#10;  pages = {4137--4141},&#10;  year = {2024},&#10;  doi = {10.1109/BIBM62325.2024.10822445}&#10;}">Copy BibTeX</button>
     </p>
   </div>

.. raw:: html

   <div class="eegprep-citation">
     <p><strong>ICA Artifact Rejection</strong></p>
     <p>
       Delorme, A., Sejnowski, T., &amp; Makeig, S. (2007). Enhanced detection of artifacts in EEG data using higher-order statistics and independent component analysis. NeuroImage, 34(4), 1443-1449.
       <span class="eegprep-cite-links"><a href="https://pubmed.ncbi.nlm.nih.gov/17188898/">PubMed</a> | <a href="https://openalex.org/W2149407814">OpenAlex</a></span>
       <button class="eegprep-cite-copy" data-citation="Delorme, A., Sejnowski, T., &amp; Makeig, S. (2007). Enhanced detection of artifacts in EEG data using higher-order statistics and independent component analysis. NeuroImage, 34(4), 1443-1449.">Copy APA</button>
       <button class="eegprep-cite-copy" data-citation="@article{delorme2007artifacts,&#10;  author = {Delorme, Arnaud and Sejnowski, Terrence and Makeig, Scott},&#10;  title = {Enhanced detection of artifacts in EEG data using higher-order statistics and independent component analysis},&#10;  journal = {NeuroImage},&#10;  volume = {34},&#10;  number = {4},&#10;  pages = {1443--1449},&#10;  year = {2007},&#10;  doi = {10.1016/j.neuroimage.2006.11.004}&#10;}">Copy BibTeX</button>
     </p>
   </div>

.. raw:: html

   <div class="eegprep-citation">
     <p><strong>EEG-BIDS</strong></p>
     <p>
       Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., &amp; Oostenveld, R. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6, 103.
       <span class="eegprep-cite-links"><a href="https://pubmed.ncbi.nlm.nih.gov/31239435/">PubMed</a> | <a href="https://openalex.org/W2956069845">OpenAlex</a></span>
       <button class="eegprep-cite-copy" data-citation="Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., &amp; Oostenveld, R. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6, 103.">Copy APA</button>
       <button class="eegprep-cite-copy" data-citation="@article{pernet2019eegbids,&#10;  author = {Pernet, Cyril R. and Appelhoff, Stefan and Gorgolewski, Krzysztof J. and Flandin, Guillaume and Phillips, Christophe and Delorme, Arnaud and Oostenveld, Robert},&#10;  title = {EEG-BIDS, an extension to the brain imaging data structure for electroencephalography},&#10;  journal = {Scientific Data},&#10;  volume = {6},&#10;  pages = {103},&#10;  year = {2019},&#10;  doi = {10.1038/s41597-019-0104-8}&#10;}">Copy BibTeX</button>
     </p>
   </div>
